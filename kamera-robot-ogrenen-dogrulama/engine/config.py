"""Fabrikanın ayarlayacağı düğmeler: ödül/ceza (maliyet) tablosu, güvenlik kuralları, öğrenme ayarları."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import Action, CameraEvent


def _per_class(fire: float, smoke: float, smoking: float, open_door: float):
    return field(default_factory=lambda: {"fire": fire, "smoke": smoke, "smoking": smoking, "open_door": open_door})


@dataclass(frozen=True)
class CostModel:
    """Ödül/ceza tablosu. Birim önemli değil (TL, puan...), oranlar önemli.

    "Sorun varsa ödül, yoksa ceza" fikrinin maliyet karşılığı:
      robot gitti, olay yoktu     -> yol maliyeti (küçük ceza)
      robot gitti, olay gerçekti  -> kaçırma bedelinden kurtulduk (sınıfa göre büyük ödül)
      gidilmedi, olay gerçekti    -> kaçırma bedeli. Bu satır sahada GÖZLENEMEZ; bu yüzden
                                     ödülü değil, olayın gerçek olma olasılığını öğreniyoruz.
    """

    miss: dict[str, float] = _per_class(5000.0, 800.0, 40.0, 60.0)       # gerçek olayı kaçırmak
    false_alarm: dict[str, float] = _per_class(80.0, 60.0, 25.0, 20.0)   # insanları boşuna çağırmak
    delay_per_min: dict[str, float] = _per_class(150.0, 20.0, 1.0, 1.0)  # gerçek olayda her dakika gecikme
    trip_base: float = 4.0     # robotun bir görevinin sabit maliyeti (enerji, aşınma)
    trip_per_min: float = 1.0  # robotun meşgul kaldığı her dakika
    review: float = 2.0        # bir insanın kamera klibine bakması

    def expected(self, action: Action, cls: str, p, eta_min: float):
        """Olay p olasılıkla gerçekse aksiyonun beklenen maliyeti (p bir dizi de olabilir)."""
        if action is Action.IGNORE:
            return p * self.miss[cls]
        if action is Action.ALARM:
            return (1.0 - p) * self.false_alarm[cls]
        return self.trip_base + self.trip_per_min * eta_min + p * self.delay_per_min[cls] * eta_min


@dataclass(frozen=True)
class SafetyRules:
    """Öğrenmenin asla aşamayacağı sabit kurallar; model yalnızca bu sınırların içinde karar verir."""

    critical: tuple[str, ...] = ("fire", "smoke")
    never_ignore_conf: float = 0.25  # kritik sınıfta bu güvenin üstündeki tespit yok sayılamaz
    force_alarm_conf: dict[str, float] = field(default_factory=lambda: {"fire": 0.85, "smoke": 0.95})

    def allowed_actions(self, event: CameraEvent) -> tuple[list[Action], str | None]:
        """Olay için izin verilen aksiyonlar ve (varsa) bunu belirleyen kural."""
        if event.conf >= self.force_alarm_conf.get(event.cls, math.inf):
            return [Action.ALARM], "kural: yüksek güvenli kritik olay"
        if event.cls in self.critical and event.conf >= self.never_ignore_conf:
            return [Action.DISPATCH, Action.ALARM], "kural: kritik olay yok sayılamaz"
        return list(Action), None


@dataclass(frozen=True)
class LearningConfig:
    # "greedy": ortalama tahmine göre karar ver (önerilen; güven skorlarındaki çeşitlilik zaten doğal keşif sağlar)
    # "thompson": belirsizlik kadar dene; simülasyonda denetime göre ek fayda sağlamadı, seçenek olarak duruyor
    exploration: str = "greedy"
    audit_rate: float = 0.03  # "yok say" kararlarının bu kadarında robot yine de gider (kör nokta denetimi)
    labeling: str = "careful"      # "careful": geç varışta "bulunamadı" kanıt sayılmaz | "naive": bulunamadı = yanlış alarm
    # Robot bu süreden (dk) geç vardıysa "bulunamadı" sonucu etiket sayılmaz, klibe insan bakar
    freshness_min: dict[str, float] = _per_class(math.inf, 20.0, 2.5, 15.0)
    half_life_days: float | None = 30.0  # eski etiketlerin ağırlığı bu sürede yarıya iner; None: hiç unutma
    refit_every: int = 5                 # kaç yeni etikette bir model yeniden kurulsun


@dataclass(frozen=True)
class EngineConfig:
    cameras: tuple[str, ...]
    costs: CostModel = field(default_factory=CostModel)
    safety: SafetyRules = field(default_factory=SafetyRules)
    learning: LearningConfig = field(default_factory=LearningConfig)
