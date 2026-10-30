"""Olayı modelin anlayacağı sayısal vektöre çevirir ve başlangıç bilgisini (prior) tanımlar."""

from __future__ import annotations

import math

import numpy as np

from .types import EVENT_CLASSES, CameraEvent

NIGHT_START, NIGHT_END = 19.0, 7.0


def is_night(hour: float) -> bool:
    return hour >= NIGHT_START or hour < NIGHT_END


class FeatureEncoder:
    """Vektör blokları:
        [sınıf sabiti | sınıf × logit(güven) | kamera × sınıf | kamera × gece | log(iz süresi)]

    Başlangıç varsayımı "kameranın güven skoru doğrudur", yani ilk gün p(gerçek) ≈ güven.
    Saha etiketleri bu varsayımı kamera, sınıf ve saat bazında düzeltir; örneğin
    "kaynak atölyesindeki yangın tespitleri gündüz genelde kıvılcımdır".
    """

    def __init__(self, cameras: tuple[str, ...], classes: tuple[str, ...] = EVENT_CLASSES):
        self.cameras = tuple(cameras)
        self.classes = tuple(classes)
        self._cam = {c: i for i, c in enumerate(self.cameras)}
        self._cls = {k: i for i, k in enumerate(self.classes)}
        nk, nc = len(self.classes), len(self.cameras)
        self._o_slope = nk
        self._o_camcls = 2 * nk
        self._o_night = 2 * nk + nc * nk
        self._o_track = self._o_night + nc
        self.dim = self._o_track + 1

    def prior(self) -> tuple[np.ndarray, np.ndarray]:
        """(ortalama, varyans). Varyans, o ağırlığın veriyle ne kadar kolay değişebileceğidir."""
        mean = np.zeros(self.dim)
        var = np.empty(self.dim)
        var[: self._o_slope] = 2.0                    # sınıf sabitleri
        mean[self._o_slope : self._o_camcls] = 1.0    # güven skoru başlangıçta kalibre kabul edilir
        var[self._o_slope : self._o_camcls] = 0.5
        var[self._o_camcls : self._o_night] = 1.5     # kamera × sınıf düzeltmeleri
        var[self._o_night : self._o_track] = 1.0
        var[self._o_track] = 0.5
        return mean, var

    def knows(self, ev: CameraEvent) -> bool:
        return ev.camera_id in self._cam and ev.cls in self._cls

    def encode(self, ev: CameraEvent) -> np.ndarray:
        if not self.knows(ev):
            raise ValueError(f"Tanımsız kamera veya sınıf: {ev.camera_id!r} / {ev.cls!r}")
        c, k = self._cam[ev.camera_id], self._cls[ev.cls]
        conf = min(max(ev.conf, 0.01), 0.99)
        x = np.zeros(self.dim)
        x[k] = 1.0
        x[self._o_slope + k] = math.log(conf / (1.0 - conf))
        x[self._o_camcls + c * len(self.classes) + k] = 1.0
        x[self._o_night + c] = float(is_night(ev.hour))
        x[self._o_track] = math.log1p(ev.track_sec) - math.log1p(10.0)
        return x
