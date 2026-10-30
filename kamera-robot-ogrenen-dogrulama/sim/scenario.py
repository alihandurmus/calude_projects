"""Örnek fabrika senaryosu: 6 kamera, 1 robot. Sayılar temsilidir; kendi saha verinizle değiştirin."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    pos: tuple[float, float]  # fabrika planındaki konum (m)


@dataclass(frozen=True)
class Stream:
    """Bir kamerada bir sınıf için aday olay akışı: ya gerçek olaylar ya da yanlış tetiklenmeler."""

    camera: str
    cls: str
    real: bool
    day_rate: float            # 07-19 arası günde ortalama adet
    night_rate: float          # 19-07 arası
    conf: tuple[float, float]  # kamera güven skoru ~ Beta(a, b)
    start_day: float = 0.0
    end_day: float = math.inf
    # Tespitin sürdüğü saniye ~ LogNormal(log(medyan), sigma). Varsayılan: gerçek olaylar uzun
    # süre izlenir (25 sn), yanlış tetiklenmeler titrer (8 sn).
    track: tuple[float, float] | None = None

    def track_params(self) -> tuple[float, float]:
        if self.track is not None:
            return self.track
        return (25.0, 0.6) if self.real else (8.0, 0.8)


@dataclass(frozen=True)
class Scenario:
    cameras: tuple[Camera, ...]
    streams: tuple[Stream, ...]
    robot_base: tuple[float, float] = (100.0, 50.0)
    robot_speed_m_per_min: float = 42.0  # 0.7 m/s: insanlarla ortak alanda güvenli hız
    path_factor: float = 1.4             # koridorlar yüzünden yol kuş uçuşundan uzundur
    inspect_min: float = 1.0
    robot_detect_p: float = 0.95         # olay hâlâ oradaysa robotun görme olasılığı
    robot_false_confirm_p: float = 0.02  # olay yokken robotun "var" deme olasılığı
    # Olay bitmiş olsa da robotun iz bulma olasılığı (duman/koku sensörü, izmarit...)
    residual_p: dict[str, float] = field(
        default_factory=lambda: {"fire": 1.0, "smoke": 0.5, "smoking": 0.25, "open_door": 0.0})
    blocked_p: float = 0.02              # robotun olay yerine ulaşamaması veya bakamaması
    human_accuracy: float = 0.95
    review_delay_min: float = 10.0       # operatörün klibe bakıp sonuç vermesi
    alarm_response_min: float = 10.0     # güvenlik ekibinin alarm sonrası geri bildirim süresi
    drift: tuple[str, str, float] | None = None  # (kamera, sınıf, gün): raporda ayrıca izlenir


def remaining_duration_min(cls: str, rng: np.random.Generator) -> float:
    """Gerçek bir olay tespit edildikten sonra kaç dakika daha sürer?"""
    if cls == "fire":
        return math.inf               # yangın kendiliğinden kaybolmaz
    if cls == "smoke":
        return float(rng.exponential(30.0))
    if cls == "smoking":
        return float(rng.uniform(1.0, 6.0))  # sigara birkaç dakikada biter, kişi gider
    return float(rng.exponential(40.0))      # açık kapı: biri kapatana kadar


def default_scenario(blind_spot: bool = False) -> Scenario:
    """blind_spot=True: en zor durum. K6'daki gerçek sigara tespitleri kamerada yanlış
    tetiklenmelerle güven skoru dahil birebir aynı görünür; tek ipucu oradaki olayların sıklığıdır."""
    cameras = (
        Camera("K1", "Üretim Hattı A", (45.0, 25.0)),
        Camera("K2", "Üretim Hattı B", (155.0, 25.0)),
        Camera("K3", "Kaynak Atölyesi", (30.0, 85.0)),
        Camera("K4", "Kazan Dairesi", (170.0, 85.0)),
        Camera("K5", "Yükleme Rampası", (100.0, 95.0)),
        Camera("K6", "Depo Arkası", (190.0, 8.0)),
    )
    S = Stream
    streams = []
    for cam in ("K1", "K2"):  # iyi aydınlatılmış, güven skoru doğruya yakın kameralar
        streams += [
            S(cam, "smoking", True, 0.15, 0.05, (6, 3)),
            S(cam, "smoking", False, 0.8, 0.3, (3, 5)),
            S(cam, "smoke", True, 0.01, 0.01, (6, 3)),
            S(cam, "smoke", False, 0.3, 0.1, (3, 5)),
            S(cam, "fire", True, 0.004, 0.004, (8, 2)),
            S(cam, "fire", False, 0.05, 0.02, (3, 5)),
        ]
    streams += [
        # Kaynak atölyesi: mesai saatlerinde kıvılcım ve kaynak dumanı yangın/duman sanılıyor
        S("K3", "fire", False, 3.0, 0.1, (6, 3)),
        S("K3", "smoke", False, 2.0, 0.1, (5, 4)),
        S("K3", "fire", True, 0.006, 0.006, (8, 2)),
        S("K3", "smoke", True, 0.01, 0.01, (6, 3)),
        S("K3", "smoking", True, 0.05, 0.02, (5, 4)),
        S("K3", "smoking", False, 0.3, 0.1, (3, 5)),
        # Kazan dairesi: buhar gece gündüz duman sanılıyor
        S("K4", "smoke", False, 2.0, 2.0, (4, 4)),
        S("K4", "smoke", True, 0.01, 0.01, (6, 3)),
        S("K4", "fire", True, 0.004, 0.004, (8, 2)),
        S("K4", "fire", False, 0.1, 0.1, (3, 4)),
        # Yükleme rampası: gündüz güneş yansıması ve tırlar kapıyı açık gösteriyor;
        # gerçekten açık unutulan kapı ise çoğunlukla gece
        S("K5", "open_door", False, 3.0, 0.3, (5, 3)),
        S("K5", "open_door", True, 0.3, 1.0, (5, 2)),
        S("K5", "smoking", True, 0.4, 0.1, (5, 3)),
        S("K5", "smoking", False, 0.5, 0.1, (3, 5)),
        # Depo arkası: karanlık ve uzak, kamera düşük güven veriyor. Havalandırma çıkışının buharı
        # sigara dumanı gibi uzun süre görünüyor; kameranın hiçbir sinyali ikisini ayıramıyor.
        # 45. günden sonra sigara içenler burayı "kör nokta" olarak keşfediyor.
        S("K6", "smoking", True, 0.1, 0.05, (3, 5) if blind_spot else (3, 4), end_day=45),
        S("K6", "smoking", True, 2.0, 0.5, (3, 5) if blind_spot else (3, 4), start_day=45),
        S("K6", "smoking", False, 1.5, 1.0, (3, 5), track=(25.0, 0.6)),
        S("K6", "open_door", True, 0.05, 0.3, (4, 3)),
        S("K6", "open_door", False, 0.3, 0.3, (3, 5)),
    ]
    return Scenario(cameras, tuple(streams), drift=("K6", "smoking", 45.0))
