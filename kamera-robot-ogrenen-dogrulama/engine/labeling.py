"""Robot sonucunu eğitim etiketine çevirir."""

from __future__ import annotations

import math

from .config import LearningConfig
from .types import Feedback, RobotVerdict


def label_from_robot(cls: str, verdict: RobotVerdict, arrival_delay_min: float,
                     learning: LearningConfig) -> Feedback:
    """"Robot bulamadı" her zaman "olay yoktu" demek değildir.

    Sigara içen kişi robot varmadan gitmiş, kapı bu arada kapatılmış olabilir. Bunları yanlış alarm
    diye etiketlersek model gerçek olaylara güvenmemeyi öğrenir. Bu yüzden geç varılan ya da
    bakılamayan durumlarda kamera klibine bir insan bakar.
    """
    if verdict is RobotVerdict.CONFIRMED:
        return Feedback(label=1, needs_review=False)
    if learning.labeling == "naive":
        # Basit yaklaşım: bulunamadıysa yanlış alarmdır, bakılamadıysa unutulur
        return Feedback(label=0 if verdict is RobotVerdict.NOT_FOUND else None, needs_review=False)
    if verdict is RobotVerdict.INCONCLUSIVE or arrival_delay_min > learning.freshness_min.get(cls, math.inf):
        return Feedback(label=None, needs_review=True)
    return Feedback(label=0, needs_review=False)
