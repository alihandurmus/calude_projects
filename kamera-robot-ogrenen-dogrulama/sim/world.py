"""Ayrık olay simülasyonu: kameralar aday olay üretir, politika karar verir, robot gider, sonuç döner.

Sahaya çıkmadan önce karar motorunu, maliyet tablosunu ve güvenlik kurallarını burada deneyin.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from engine import Action, CameraEvent, CostModel, RobotVerdict

from .scenario import Scenario, remaining_duration_min

COST_KEYS = ("miss", "trip", "false_alarm", "delay", "review")


@dataclass(frozen=True)
class SimEvent:
    ev: CameraEvent
    real: bool
    remaining_min: float  # gerçek olay tespitten sonra kaç dakika daha sürecek
    u: np.ndarray         # olaya özel rastgele sayılar: tüm politikalar aynı "şansı" yaşar


def generate_events(sc: Scenario, days: int, seed: int) -> list[SimEvent]:
    rng = np.random.default_rng(seed)
    raw = []
    for s in sc.streams:
        for rate, start_h in ((s.day_rate, 7.0), (s.night_rate, 19.0)):
            for d in range(days):
                for t_h in d * 24 + start_h + rng.uniform(0.0, 12.0, rng.poisson(rate)):
                    if s.start_day <= t_h / 24 < min(s.end_day, days):
                        raw.append((t_h, s))
    raw.sort(key=lambda r: r[0])
    events = []
    for i, (t_h, s) in enumerate(raw):
        median, sigma = s.track_params()
        track = rng.lognormal(math.log(median), sigma)
        ev = CameraEvent(event_id=f"e{i:06d}", camera_id=s.camera, cls=s.cls, conf=float(rng.beta(*s.conf)),
                         ts=t_h * 3600.0, hour=t_h % 24, track_sec=float(track))
        remaining = remaining_duration_min(s.cls, rng) if s.real else 0.0
        events.append(SimEvent(ev, s.real, remaining, rng.random(4)))
    return events


class Robot:
    """Tek robot; görevleri sırayla yapar, boşta kalınca devriye merkezine döner."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.pos = sc.robot_base
        self.free_at = 0.0

    def travel_min(self, a, b) -> float:
        return math.dist(a, b) * self.sc.path_factor / self.sc.robot_speed_m_per_min

    def eta(self, target, t: float) -> float:
        start, pos = self._start(t)
        return start - t + self.travel_min(pos, target)

    def assign(self, target, t: float) -> tuple[float, float]:
        """Görevi kuyruğa ekler; (varış zamanı, robotun meşgul kalacağı dk) döner."""
        start, pos = self._start(t)
        travel = self.travel_min(pos, target)
        self.pos, self.free_at = target, start + travel + self.sc.inspect_min
        return start + travel, travel + self.sc.inspect_min

    def _start(self, t: float):
        start = max(t, self.free_at)
        back_home = start - self.free_at >= self.travel_min(self.pos, self.sc.robot_base)
        return start, (self.sc.robot_base if back_home else self.pos)


def robot_verdict(se: SimEvent, delay_min: float, sc: Scenario) -> RobotVerdict:
    u_detect, u_residual, u_blocked = se.u[0], se.u[1], se.u[2]
    if u_blocked < sc.blocked_p:
        return RobotVerdict.INCONCLUSIVE
    if not se.real:
        return RobotVerdict.CONFIRMED if u_detect < sc.robot_false_confirm_p else RobotVerdict.NOT_FOUND
    if delay_min <= se.remaining_min:  # olay hâlâ sürüyor
        return RobotVerdict.CONFIRMED if u_detect < sc.robot_detect_p else RobotVerdict.NOT_FOUND
    return RobotVerdict.CONFIRMED if u_residual < sc.residual_p[se.ev.cls] else RobotVerdict.NOT_FOUND


def human_says_real(se: SimEvent, sc: Scenario) -> bool:
    return se.real if se.u[3] < sc.human_accuracy else not se.real


@dataclass
class Result:
    name: str
    days: int
    period_days: int
    cost: dict[str, float] = field(default_factory=lambda: dict.fromkeys(COST_KEYS, 0.0))
    trips: int = 0
    trips_by: Counter = field(default_factory=Counter)  # (kamera, sınıf, olay gerçek mi) -> görev sayısı
    false_alarms: int = 0  # insanlara giden yanlış alarmlar
    reviews: int = 0
    real: Counter = field(default_factory=Counter)
    missed: Counter = field(default_factory=Counter)
    ignored_real: int = 0                   # "yok say" denen gerçek olaylar (sahada görünmez)
    estimated_missed: float | None = None   # motorun denetimlerden yaptığı tahmin
    drift_real: int = 0
    drift_missed: int = 0
    period_cost: list[float] = field(init=False)
    period_missed: list[int] = field(init=False)

    def __post_init__(self):
        n = math.ceil(self.days / self.period_days)
        self.period_cost = [0.0] * n
        self.period_missed = [0] * n

    @property
    def total_cost(self) -> float:
        return sum(self.cost.values())


def simulate(policy, sc: Scenario, events: list[SimEvent], days: int, costs: CostModel | None = None,
             period_days: int = 15, name: str = "") -> Result:
    """policy: decide / record_robot_result / record_human_label metotları olan herhangi bir nesne."""
    costs = costs or CostModel()
    cams = {c.id: c.pos for c in sc.cameras}
    robot = Robot(sc)
    res = Result(name, days, period_days)
    queue = [(se.ev.ts / 60.0, i, "camera", se) for i, se in enumerate(events)]
    heapq.heapify(queue)
    seq = len(queue)

    def push(t, kind, se):
        nonlocal seq
        heapq.heappush(queue, (t, seq, kind, se))
        seq += 1

    def period(se) -> int:
        return int(se.ev.ts / 86400.0 // period_days)

    def in_drift(se) -> bool:
        return sc.drift is not None and (se.ev.camera_id, se.ev.cls) == sc.drift[:2] \
            and se.ev.ts / 86400.0 >= sc.drift[2]

    def add_cost(kind, amount, se):
        res.cost[kind] += amount
        res.period_cost[period(se)] += amount

    def close(se, reported: bool, t: float):
        """Olayın son hali: insanlar haberdar oldu mu?"""
        cls = se.ev.cls
        if se.real and reported:
            add_cost("delay", costs.delay_per_min[cls] * (t - se.ev.ts / 60.0), se)
        elif se.real:
            res.missed[cls] += 1
            res.period_missed[period(se)] += 1
            res.drift_missed += in_drift(se)
            add_cost("miss", costs.miss[cls], se)
        elif reported:
            res.false_alarms += 1
            add_cost("false_alarm", costs.false_alarm[cls], se)

    while queue:
        t, _, kind, se = heapq.heappop(queue)
        ev = se.ev
        if kind == "camera":
            if se.real:
                res.real[ev.cls] += 1
                res.drift_real += in_drift(se)
            target = cams[ev.camera_id]
            d = policy.decide(ev, robot.eta(target, t))
            if d.action is Action.IGNORE:
                res.ignored_real += se.real
                close(se, False, t)
            elif d.action is Action.ALARM:
                close(se, True, t)
                push(t + sc.alarm_response_min, "alarm_feedback", se)
            else:
                arrival, busy = robot.assign(target, t)
                res.trips += 1
                res.trips_by[ev.camera_id, ev.cls, se.real] += 1
                add_cost("trip", costs.trip_base + costs.trip_per_min * busy, se)
                push(arrival, "robot", se)
        elif kind == "robot":
            delay = t - ev.ts / 60.0
            verdict = robot_verdict(se, delay, sc)
            fb = policy.record_robot_result(ev.event_id, verdict, delay)
            if verdict is RobotVerdict.CONFIRMED:
                close(se, True, t)
            elif fb.needs_review:
                res.reviews += 1
                add_cost("review", costs.review, se)
                push(t + sc.review_delay_min, "review", se)
            else:
                close(se, False, t)
        elif kind == "review":
            says_real = human_says_real(se, sc)
            policy.record_human_label(ev.event_id, says_real)
            close(se, says_real, t)
        else:  # alarm_feedback: güvenlik ekibi olay yerine gitti, sonucu bildirdi
            policy.record_human_label(ev.event_id, human_says_real(se, sc))
    if hasattr(policy, "estimated_missed"):
        res.estimated_missed = policy.estimated_missed()
    return res
