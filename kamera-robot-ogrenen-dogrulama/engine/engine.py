"""Karar motoru: kamera olayına ne yapılacağına karar verir ve sonuçlardan öğrenir."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .config import EngineConfig
from .features import FeatureEncoder
from .labeling import label_from_robot
from .model import BayesianLogisticRegression, sigmoid
from .store import EventStore
from .types import Action, CameraEvent, Decision, Feedback, RobotVerdict

_DAY_SEC = 86400.0
_GRID = 512  # Thompson örneklemesi için logit dağılımının nicelik ızgarası


class DecisionEngine:
    """Kullanım:

        engine = DecisionEngine(EngineConfig(cameras=("K1", "K2")), store=EventStore("olaylar.db"))
        d = engine.decide(event, eta_min=3.5)                     # kameradan olay geldi
        engine.record_robot_result(d.event_id, RobotVerdict.NOT_FOUND, arrival_delay_min=4.0)
        engine.record_human_label(d.event_id, is_real=True)       # alarm cevabı veya klip incelemesi

    Karar adımları:
      1. Güvenlik kuralları izin verilen aksiyonları belirler (öğrenme bunları aşamaz).
      2. Model olayın gerçek olma olasılığını (p) ve bu tahmindeki belirsizliği verir.
      3. "Bakmalı mı?": bakmanın beklenen maliyeti yok saymaktan düşükse bakılır
         (isteğe bağlı: Thompson örneklemesiyle belirsiz bağlamlar daha sık denenir).
      4. Bakılacaksa robot mu, doğrudan alarm mı: beklenen maliyeti düşük olan.
      5. "Yok say" kararlarının küçük bir kısmında robot yine de gider (denetim). Böylece kör nokta
         oluşmaz ve sahada göremediğiniz kaçırılan olay sayısı tahmin edilebilir (estimated_missed).
    """

    def __init__(self, config: EngineConfig, store: EventStore | None = None, seed: int | None = None):
        self.cfg = config
        self.encoder = FeatureEncoder(config.cameras)
        self.model = BayesianLogisticRegression(*self.encoder.prior())
        self.store = store
        self.rng = np.random.default_rng(seed)
        self._z = np.array([NormalDist().inv_cdf((i + 0.5) / _GRID) for i in range(_GRID)])
        self._open: dict[str, tuple[CameraEvent, np.ndarray]] = {}  # karar verildi, sonucu bekleniyor
        self._audits: dict[str, float] = {}           # sonucu beklenen denetimler -> seçilme olasılığı
        self._audit_hits: list[tuple[float, float]] = []  # gerçek çıkan denetimler: (zaman, olasılık)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._t: list[float] = []
        self._unfitted = 0
        self._now = -math.inf
        if store is not None:
            self._restore(store)

    def decide(self, event: CameraEvent, eta_min: float | None) -> Decision:
        """eta_min: robotun olay yerine tahmini varış süresi (kuyruk dahil, dk). None: robot yok."""
        self._now = max(self._now, event.ts)
        x = self.encoder.encode(event)
        m, v = self.model.logit_moments(x)
        p_mean = float(sigmoid(m))

        allowed, rule = self.cfg.safety.allowed_actions(event)
        if eta_min is None:
            allowed = [a for a in allowed if a is not Action.DISPATCH] or [Action.ALARM]
        eta = 0.0 if eta_min is None else eta_min

        def cost(a, p):
            return self.cfg.costs.expected(a, event.cls, p, eta)

        reason = rule if len(allowed) == 1 else ("model" if rule is None else f"model ({rule})")
        informative = [a for a in allowed if a is not Action.IGNORE]  # sonucu öğrenilebilen aksiyonlar
        look = min(informative, key=lambda a: cost(a, p_mean))
        if Action.IGNORE not in allowed:
            action, probs = look, {look: 1.0}
        else:
            # Keşif yalnızca "bakmalı mı?" sorusunda gerekir: yok sayılan olayın sonucu hiç öğrenilemez.
            # Robot mu, insan mı sorusunda ise iki durumda da etiket geldiği için ortalama tahmin yeterli.
            learning = self.cfg.learning
            if learning.exploration == "thompson":
                draws = sigmoid(m + math.sqrt(v) * self._z)
                look_cost = np.min([cost(a, draws) for a in informative], axis=0)
                p_look = float(np.mean(look_cost < cost(Action.IGNORE, draws)))
            else:
                p_look = float(cost(look, p_mean) < cost(Action.IGNORE, p_mean))
            audit = learning.audit_rate if Action.DISPATCH in allowed else 0.0
            probs = {Action.IGNORE: (1.0 - p_look) * (1.0 - audit), look: p_look}
            probs[Action.DISPATCH] = probs.get(Action.DISPATCH, 0.0) + (1.0 - p_look) * audit
            if self.rng.random() < p_look:
                action = look
            elif self.rng.random() < audit:
                action, reason = Action.DISPATCH, "denetim"
            else:
                action = Action.IGNORE

        decision = Decision(
            event_id=event.event_id,
            action=action,
            p_real=p_mean,
            propensity=probs[action],
            reason=reason,
            expected_cost={a: float(cost(a, p_mean)) for a in allowed},
        )
        if action is not Action.IGNORE:
            self._open[event.event_id] = (event, x)
        if reason == "denetim":
            self._audits[event.event_id] = decision.propensity
        if self.store is not None:
            self.store.log_decision(event, decision)
        return decision

    def record_robot_result(self, event_id: str, verdict: RobotVerdict, arrival_delay_min: float) -> Feedback:
        """Robot incelemeyi bitirdiğinde çağrılır. needs_review=True ise klibi bir insana gösterin."""
        event, _ = self._open[event_id]
        fb = label_from_robot(event.cls, verdict, arrival_delay_min, self.cfg.learning)
        if self.store is not None:
            self.store.log_verdict(event_id, verdict, fb.needs_review)
        if fb.label is not None:
            self._add_label(event_id, fb.label, source="robot")
        elif not fb.needs_review:
            self._open.pop(event_id)  # etiketsiz kapandı
            self._audits.pop(event_id, None)
        return fb

    def record_human_label(self, event_id: str, is_real: bool) -> None:
        """Güvenlik ekibinin alarm sonrası cevabı veya kamera klibi incelemesi: en güvenilir etiket."""
        self._add_label(event_id, int(is_real), source="human")

    def add_example(self, event: CameraEvent, is_real: bool, source: str = "example") -> None:
        """Motorun kararından bağımsız etiketli olay.

        Gölge mod (mevcut kural tabanlı sistem karar verirken motor yalnızca izler ve öğrenir),
        geçmiş olay kayıtları ve kontrollü tatbikatlar (ör. duman testi) böyle eklenir.
        """
        self._now = max(self._now, event.ts)
        self._open[event.event_id] = (event, self.encoder.encode(event))
        if self.store is not None:
            self.store.log_example(event)
        self._add_label(event.event_id, int(is_real), source=source)

    def predict(self, event: CameraEvent) -> tuple[float, float]:
        """(p_gerçek, logit standart sapması): izleme ve açıklama için."""
        m, v = self.model.logit_moments(self.encoder.encode(event))
        return float(sigmoid(m)), math.sqrt(v)

    def estimated_missed(self, since_ts: float = -math.inf) -> float | None:
        """"Yok say" denen olaylar arasında kaç gerçek olay kaçtı? Denetimlerden tahmin (denetim yoksa None).

        Sahada kaçırılan olayları doğrudan göremezsiniz; tek pencere denetimlerdir. Denetim olasılığı
        a iken bulunan her gerçek olay, denetlenmeden geçen ortalama (1-a)/a gerçek olayı temsil eder.
        Denetim oranı düşükse tahmin kabadır: amaç kesin sayı değil, kör noktayı erken fark etmek.
        """
        if self.cfg.learning.audit_rate <= 0:
            return None
        return sum((1.0 - p) / p for ts, p in self._audit_hits if ts >= since_ts)

    def refit(self) -> None:
        """Modeli tüm etiketlerden yeniden kurar; eski etiketlerin ağırlığı zamanla yarıya iner."""
        self._unfitted = 0
        if not self._y:
            self.model.reset()
            return
        half_life = self.cfg.learning.half_life_days
        if half_life is None:
            w = np.ones(len(self._t))
        else:
            age = np.maximum(self._now - np.asarray(self._t), 0.0)
            w = 0.5 ** (age / (half_life * _DAY_SEC))
        self.model.fit(np.asarray(self._X), np.asarray(self._y), w)

    def _add_label(self, event_id: str, label: int, source: str) -> None:
        event, x = self._open.pop(event_id)
        audit_p = self._audits.pop(event_id, None)
        if audit_p is not None and label == 1:
            self._audit_hits.append((event.ts, audit_p))
        self._X.append(x)
        self._y.append(float(label))
        self._t.append(event.ts)
        if self.store is not None:
            self.store.log_label(event_id, label, source)
        self._unfitted += 1
        if self._unfitted >= self.cfg.learning.refit_every:
            self.refit()

    def _restore(self, store: EventStore) -> None:
        labeled, pending = store.history()
        for ev, label in labeled:
            if self.encoder.knows(ev):  # artık tanımlı olmayan kameraların kayıtları atlanır
                self._X.append(self.encoder.encode(ev))
                self._y.append(float(label))
                self._t.append(ev.ts)
                self._now = max(self._now, ev.ts)
        for ev in pending:
            if self.encoder.knows(ev):
                self._open[ev.event_id] = (ev, self.encoder.encode(ev))
                self._now = max(self._now, ev.ts)
        for event_id, ts, propensity, label in store.audits():
            if label == 1:
                self._audit_hits.append((ts, propensity))
            elif label is None and event_id in self._open:
                self._audits[event_id] = propensity
        self.refit()
