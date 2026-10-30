"""Simülasyonu tarayıcıda izlenebilir tek dosyalık bir HTML sayfasına dönüştürür.

    python gorsellestir.py                      # gorsel/simulasyon.html üretir; tarayıcıda açın
    python gorsellestir.py --seed 2 --days 150  # başka bir olay akışı

Sayfa aynı olay akışında üç politikayı (sabit eşik, basit öğrenen, önerilen) oynatır:
fabrika planında kameraların tespitleri, robotun görevleri ve her olayın sonucu.
Birim testleri de çalıştırılır ve sonuçları sayfaya eklenir.
"""

from __future__ import annotations

import os

# Küçük matrislerde çok iş parçacıklı BLAS hızlandırmaz, yavaşlatır. numpy yüklenmeden önce ayarlanmalı.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from engine import CLASS_NAMES_TR, EVENT_CLASSES, Action, RobotVerdict  # noqa: E402
from engine.features import is_night  # noqa: E402
from sim import POLICY_NAMES, build_policy, default_scenario, generate_events, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "gorsel" / "sablon.html"
PLACEHOLDER = "/*__VERI__*/null"
POLICIES = ("sabit", "basit", "onerilen")
ACTIONS = (Action.IGNORE, Action.DISPATCH, Action.ALARM)
OUTCOMES = ("cleared", "caught", "missed", "false_alarm")
VERDICTS = (RobotVerdict.CONFIRMED, RobotVerdict.NOT_FOUND, RobotVerdict.INCONCLUSIVE)
DAY_SEC = 86400.0

# Fabrika planında kameraların bulunduğu alanlar (x0, y0, x1, y1; metre). Tanımsız kameraya kutu çizilir.
ZONES = {
    "K1": (10, 6, 80, 44),
    "K2": (120, 6, 172, 44),
    "K3": (6, 64, 62, 96),
    "K4": (140, 64, 196, 96),
    "K5": (72, 74, 130, 98),
    "K6": (178, 2, 198, 40),
}


class BeliefProbe:
    """Politikayı sarar; her gün sonunda kör nokta tespitleri için sistemin ortalama tahminini kaydeder."""

    def __init__(self, policy, windows: list[list]):
        self.policy = policy
        self.windows = windows  # gün -> o güne yakın (±3 gün) kör nokta tespitleri
        self.series: list[float | None] = []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def decide(self, event, eta_min):
        self._catch_up(int(event.ts // DAY_SEC))
        return self.policy.decide(event, eta_min)

    def finish(self, days: int) -> list[float | None]:
        self._catch_up(days)
        return self.series

    def _catch_up(self, day: int) -> None:
        while len(self.series) < day:
            evs = self.windows[len(self.series)]
            if not evs:
                self.series.append(None)
            elif hasattr(self.policy, "predict"):
                self.series.append(round(statistics.fmean(self.policy.predict(e)[0] for e in evs), 4))
            else:  # sabit eşik kameranın kendi güven skoruna inanır
                self.series.append(round(statistics.fmean(e.conf for e in evs), 4))


class _Collect(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.rows: list[tuple[str, bool]] = []

    def _add(self, test, ok: bool) -> None:
        describe = getattr(test, "shortDescription", lambda: None)
        self.rows.append((describe() or str(test), ok))

    def addSuccess(self, test):
        super().addSuccess(test)
        self._add(test, True)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._add(test, False)

    def addError(self, test, err):
        super().addError(test, err)
        self._add(test, False)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._add(test, False)


def run_tests() -> dict:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = _Collect()
    t0 = time.perf_counter()
    suite.run(result)
    return {
        "passed": sum(ok for _, ok in result.rows),
        "total": len(result.rows),
        "seconds": round(time.perf_counter() - t0, 2),
        "items": [{"name": name, "ok": ok} for name, ok in result.rows],
    }


def drift_windows(events, sc, days: int) -> list[list]:
    """Her gün için kör nokta bağlamının (ör. Depo Arkası · gündüz · sigara) ±3 gündeki tespitleri."""
    cam, cls, _ = sc.drift
    by_day: list[list] = [[] for _ in range(days)]
    for se in events:
        ev = se.ev
        if ev.camera_id == cam and ev.cls == cls and not is_night(ev.hour):
            by_day[int(ev.ts // DAY_SEC)].append(ev)
    return [[ev for k in range(max(0, d - 3), min(days, d + 4)) for ev in by_day[k]] for d in range(days)]


def drift_truth(sc, days: int) -> list[float]:
    """Senaryodaki sıklıklara göre kör nokta tespitlerinin gerçekte ne kadarı gerçek olay?"""
    cam, cls, _ = sc.drift
    out = []
    for d in range(days):
        real = false = 0.0
        for s in sc.streams:
            if (s.camera, s.cls) == (cam, cls) and s.start_day <= d < s.end_day:
                if s.real:
                    real += s.day_rate
                else:
                    false += s.day_rate
        out.append(round(real / (real + false), 4) if real + false else 0.0)
    return out


def run_policy(name: str, sc, events, days: int, seed: int, windows: list[list], reasons: list[str]) -> dict:
    cams = tuple(c.id for c in sc.cameras)
    probe = BeliefProbe(build_policy(name, cams, seed), windows)
    trace: dict = {}
    simulate(probe, sc, events, days, trace=trace)
    belief = probe.finish(days)

    def reason_index(reason: str) -> int:
        if reason not in reasons:
            reasons.append(reason)
        return reasons.index(reason)

    run = {k: [] for k in ("a", "p", "r", "o", "c", "v", "w", "m")}
    missions = {k: [] for k in ("s", "ox", "oy", "tx", "ty", "ar", "dn", "ev")}
    daily = {k: [0] * days for k in ("missed", "caught", "fa", "trips")}
    for i, se in enumerate(events):
        rec = trace[se.ev.event_id]
        d = rec["decision"]
        run["a"].append(ACTIONS.index(d.action))
        run["p"].append(round(d.p_real, 3))
        run["r"].append(reason_index(d.reason))
        run["o"].append(OUTCOMES.index(rec["outcome"]))
        run["c"].append(round(rec["closed_at"], 2))
        run["v"].append(VERDICTS.index(rec["verdict"]) if "verdict" in rec else -1)
        run["w"].append(int(rec.get("review", False)))
        day = int(se.ev.ts // DAY_SEC)
        if "mission" in rec:
            m = rec["mission"]
            run["m"].append(len(missions["s"]))
            for key, value in (("s", m.start), ("ox", m.origin[0]), ("oy", m.origin[1]), ("tx", m.target[0]),
                               ("ty", m.target[1]), ("ar", m.arrival), ("dn", m.done)):
                missions[key].append(round(value, 2))
            missions["ev"].append(i)
            daily["trips"][day] += 1
        else:
            run["m"].append(-1)
        outcome = rec["outcome"]
        if outcome == "missed":
            daily["missed"][day] += 1
        elif outcome == "caught":
            daily["caught"][day] += 1
        elif outcome == "false_alarm":
            daily["fa"][day] += 1
    if missions["s"] != sorted(missions["s"]):
        raise RuntimeError("Görevler başlangıç zamanına göre sıralı olmalı")
    run["missions"] = missions
    run["daily"] = daily
    run["belief"] = belief
    return run


def first_catch_day(run: dict, events, sc, days: int) -> int | None:
    """Önerilen sistemin kör noktadaki gerçek olayların çoğunu yakalamaya başladığı ilk gün."""
    cam, cls, start = sc.drift
    real_by_day = [[0, 0] for _ in range(days)]  # [gerçek, yakalanan]
    for i, se in enumerate(events):
        if se.real and se.ev.camera_id == cam and se.ev.cls == cls:
            d = int(se.ev.ts // DAY_SEC)
            real_by_day[d][0] += 1
            real_by_day[d][1] += OUTCOMES[run["o"][i]] == "caught"
    for d in range(int(start), days - 5):
        real = sum(real_by_day[k][0] for k in range(d, d + 5))
        caught = sum(real_by_day[k][1] for k in range(d, d + 5))
        if real >= 3 and caught / real >= 0.5:
            return d
    return None


def build_data(seed: int, days: int, blind: bool) -> dict:
    sc = default_scenario(blind_spot=blind)
    events = generate_events(sc, days, seed)
    windows = drift_windows(events, sc, days)
    reasons: list[str] = []
    runs = {name: run_policy(name, sc, events, days, seed, windows, reasons) for name in POLICIES}

    at_9 = 9 * 60  # anlar sabah 09:00'dan başlar: gündüz vardiyası, olaylar yoğun
    moments = [{"t": at_9, "label": "Gün 0 · başlangıç"}]
    cam, cls, drift_day = sc.drift
    moments.append({"t": drift_day * 1440 + at_9, "label": f"Gün {drift_day:.0f} · kör nokta oluşuyor"})
    catch = first_catch_day(runs["onerilen"], events, sc, days)
    if catch is not None:
        moments.append({"t": catch * 1440 + at_9, "label": f"Gün {catch} · önerilen sistem fark ediyor"})
    if days > 100:
        moments.append({"t": 100 * 1440 + at_9, "label": "Gün 100 · öğrenmiş sistem"})

    cam_name = next(c.name for c in sc.cameras if c.id == cam)
    return {
        "meta": {
            "days": days, "seed": seed, "blind": blind, "events": len(events),
            "generated": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "python": platform.python_version(), "numpy": np.__version__,
        },
        "scenario": {
            "cameras": [{"id": c.id, "name": c.name, "x": c.pos[0], "y": c.pos[1],
                         "zone": ZONES.get(c.id, (c.pos[0] - 20, c.pos[1] - 12, c.pos[0] + 20, c.pos[1] + 12))}
                        for c in sc.cameras],
            "base": list(sc.robot_base), "speed": sc.robot_speed_m_per_min,
            "pathFactor": sc.path_factor, "inspect": sc.inspect_min,
            "drift": {"camera": cam, "cameraName": cam_name, "cls": cls, "clsName": CLASS_NAMES_TR[cls],
                      "day": drift_day},
        },
        "classes": [{"key": k, "name": CLASS_NAMES_TR[k]} for k in EVENT_CLASSES],
        "policies": [{"key": k, "name": POLICY_NAMES[k]} for k in POLICIES],
        "reasons": reasons,
        "events": {
            "t": [round(se.ev.ts / 60.0, 2) for se in events],
            "cam": [next(i for i, c in enumerate(sc.cameras) if c.id == se.ev.camera_id) for se in events],
            "cls": [EVENT_CLASSES.index(se.ev.cls) for se in events],
            "conf": [round(se.ev.conf, 3) for se in events],
            "real": [int(se.real) for se in events],
        },
        "runs": runs,
        "beliefTruth": drift_truth(sc, days),
        "moments": moments,
        "start": moments[1]["t"],
        "tests": run_tests(),
    }


def render(data: dict, fragment: bool) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"Şablonda {PLACEHOLDER} bulunamadı: {TEMPLATE}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace(PLACEHOLDER, payload)
    if fragment:
        return page
    return ('<!doctype html>\n<html lang="tr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simülasyonu tarayıcıda izlenebilir bir HTML sayfasına dönüştürür")
    ap.add_argument("--seed", type=int, default=0, help="olay akışı tohumu (varsayılan 0)")
    ap.add_argument("--days", type=int, default=120, help="simüle edilecek gün sayısı (varsayılan 120)")
    ap.add_argument("--blind", action="store_true", help="zor kör nokta senaryosu")
    ap.add_argument("--out", type=Path, default=ROOT / "gorsel" / "simulasyon.html", help="çıktı dosyası")
    ap.add_argument("--fragment", action="store_true",
                    help="sayfa iskeleti olmadan yaz (claude.ai Artifact olarak yayınlamak için)")
    args = ap.parse_args()
    if args.days <= 46:
        ap.error("--days en az 47 olmalı (kör nokta 45. günde başlıyor)")

    data = build_data(args.seed, args.days, args.blind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(data, args.fragment), encoding="utf-8")
    tests = data["tests"]
    print(f"Testler: {tests['passed']}/{tests['total']} geçti")
    print(f"Yazıldı: {args.out} ({args.out.stat().st_size / 1024:.0f} KB). Tarayıcıda açın.")


if __name__ == "__main__":
    main()
