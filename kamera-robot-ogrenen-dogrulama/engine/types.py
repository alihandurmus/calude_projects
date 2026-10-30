"""Karar motorunun ortak veri tipleri."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Kameraların tespit ettiği istenmeyen durum sınıfları
EVENT_CLASSES: tuple[str, ...] = ("fire", "smoke", "smoking", "open_door")
CLASS_NAMES_TR: dict[str, str] = {
    "fire": "Yangın",
    "smoke": "Duman",
    "smoking": "Sigara",
    "open_door": "Açık kapı",
}


class Action(str, Enum):
    IGNORE = "ignore"      # Sadece kaydet, kimseyi rahatsız etme
    DISPATCH = "dispatch"  # Robotu yakından bakmaya gönder
    ALARM = "alarm"        # Doğrudan güvenlik ekibine bildir


class RobotVerdict(str, Enum):
    CONFIRMED = "confirmed"        # Robot olayı gördü / ölçtü
    NOT_FOUND = "not_found"        # Robot baktı, bir şey bulamadı
    INCONCLUSIVE = "inconclusive"  # Robot bakamadı (yol kapalı, görüş engelli...)


@dataclass(frozen=True)
class CameraEvent:
    """Kameranın görüntü işleme modelinden gelen aday olay."""

    event_id: str
    camera_id: str
    cls: str              # EVENT_CLASSES içinden
    conf: float           # Kamera modelinin güven skoru (0-1)
    ts: float             # Olay zamanı, saniye (epoch veya simülasyon saati)
    hour: float           # Yerel saat (0-24): gece/gündüz bağlamı için
    track_sec: float = 5.0          # Tespit kaç saniyedir sürüyor (titreyen yanlış tespitler kısa olur)
    snapshot: str | None = None     # Görüntü yolu: kamera modeli yeniden eğitilirken kullanılır
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Decision:
    event_id: str
    action: Action
    p_real: float      # Modelin "olay gerçek" olasılığı tahmini (ortalama)
    propensity: float  # Bu aksiyonun seçilme olasılığı; politikayı sonradan tarafsız değerlendirmek için saklanır
    reason: str        # "model", "denetim" veya "kural: ..."
    expected_cost: dict[Action, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    """Robot sonucunun öğrenmeye dönüşmüş hali."""

    label: int | None   # 1: olay gerçek, 0: yanlış alarm, None: henüz bilinmiyor
    needs_review: bool  # True ise kamera klibi bir insana incelettirilmeli
