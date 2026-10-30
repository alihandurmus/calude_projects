"""Simülasyonda karşılaştırılan politikalar."""

from __future__ import annotations

from engine import (Action, CameraEvent, Decision, DecisionEngine, EngineConfig, Feedback, LearningConfig,
                    RobotVerdict, SafetyRules)


class FixedThresholdPolicy:
    """Bugünkü tipik kural tabanlı sistem: güven ≥ 0.80 alarm, 0.40-0.80 robot, altı yok say. Öğrenmez."""

    def __init__(self, low: float = 0.4, high: float = 0.8, safety: SafetyRules | None = None):
        self.low, self.high = low, high
        self.safety = safety or SafetyRules()

    def decide(self, event: CameraEvent, eta_min: float | None) -> Decision:
        allowed, _ = self.safety.allowed_actions(event)
        if eta_min is None:
            allowed = [a for a in allowed if a is not Action.DISPATCH] or [Action.ALARM]
        if event.conf >= self.high:
            want = Action.ALARM
        elif event.conf >= self.low:
            want = Action.DISPATCH
        else:
            want = Action.IGNORE
        action = want if want in allowed else allowed[0]
        return Decision(event.event_id, action, p_real=event.conf, propensity=1.0, reason="sabit eşik")

    def record_robot_result(self, event_id: str, verdict: RobotVerdict, arrival_delay_min: float) -> Feedback:
        # Robot bulamadıysa ya da bakamadıysa olay kapanır
        return Feedback(label=1 if verdict is RobotVerdict.CONFIRMED else 0, needs_review=False)

    def record_human_label(self, event_id: str, is_real: bool) -> None:
        pass


POLICY_NAMES = {
    "sabit": "Sabit eşik",
    "basit": "Basit öğrenen",
    "onerilen": "Önerilen",
    # Ablasyon: önerilen sistemden tek bir bileşeni çıkarılmış ya da değiştirilmiş halleri
    "onerilen-denetimsiz": "Denetimsiz",
    "onerilen-naif-etiket": "Naif etiket",
    "onerilen-unutmasiz": "Unutmasız",
    "onerilen-thompson": "Thompson",
}

_LEARNING = {
    # İlk fikrin doğrudan hali: ödül/ceza ile öğren, hep en iyi tahmine göre davran,
    # robot bulamadıysa "yanlış alarm" say, eski bilgiyi hiç unutma
    "basit": LearningConfig(audit_rate=0.0, labeling="naive", half_life_days=None),
    "onerilen": LearningConfig(),
    "onerilen-denetimsiz": LearningConfig(audit_rate=0.0),
    "onerilen-naif-etiket": LearningConfig(labeling="naive"),
    "onerilen-unutmasiz": LearningConfig(half_life_days=None),
    "onerilen-thompson": LearningConfig(exploration="thompson"),
}


def build_policy(name: str, cameras: tuple[str, ...], seed: int | None = None):
    if name == "sabit":
        return FixedThresholdPolicy()
    return DecisionEngine(EngineConfig(cameras=tuple(cameras), learning=_LEARNING[name]), seed=seed)
