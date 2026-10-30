"""Politikaları aynı olay akışında karşılaştırır ve modelin neler öğrendiğini gösterir.

    python run_simulation.py                # 3 politika, 120 gün, 5 farklı rastgele olay akışı
    python run_simulation.py --ablation     # önerilen sistemin bileşenlerini tek tek çıkarır
    python run_simulation.py --blind        # en zor kör nokta: K6'da gerçek ve yanlış tespit aynı görünür
    python run_simulation.py --days 180 --seeds 10
"""

from __future__ import annotations

import os

# Küçük matrislerde çok iş parçacıklı BLAS hızlandırmaz, yavaşlatır. numpy yüklenmeden önce ayarlanmalı.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import statistics  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from engine import CLASS_NAMES_TR, EVENT_CLASSES  # noqa: E402
from engine.features import is_night  # noqa: E402
from sim import POLICY_NAMES, build_policy, default_scenario, generate_events, simulate  # noqa: E402

MAIN = ("sabit", "basit", "onerilen")
ABLATION = ("onerilen", "onerilen-denetimsiz", "onerilen-naif-etiket", "onerilen-unutmasiz", "onerilen-thompson")
RECENT_DAYS = 60

# Modelin öğrendiklerini göstermek için örnek bağlamlar: (kamera, sınıf, gece mi)
CONTEXTS = (
    ("K3", "fire", False),
    ("K3", "fire", True),
    ("K4", "smoke", False),
    ("K5", "open_door", False),
    ("K5", "open_door", True),
    ("K6", "smoking", False),
    ("K1", "smoking", False),
)


def tr(x: float, nd: int = 0) -> str:
    """Türkçe sayı biçimi: 12.345,6"""
    return f"{x:,.{nd}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float) -> str:
    return f"%{x:.0f}"


def in_context(se, ctx) -> bool:
    cam, cls, night = ctx
    return se.ev.camera_id == cam and se.ev.cls == cls and is_night(se.ev.hour) == night


def run_seed(job):
    names, days, seed, period, blind = job
    sc = default_scenario(blind_spot=blind)
    cams = tuple(c.id for c in sc.cameras)
    events = generate_events(sc, days, seed)
    recent = [se for se in events if se.ev.ts >= (days - RECENT_DAYS) * 86400.0]
    groups = [[se for se in recent if in_context(se, ctx)] for ctx in CONTEXTS]
    truth = [(statistics.fmean(se.ev.conf for se in g), statistics.fmean(se.real for se in g)) if g else None
             for g in groups]
    results, learned = {}, {}
    for name in names:
        policy = build_policy(name, cams, seed)
        results[name] = simulate(policy, sc, events, days, period_days=period, name=name)
        if hasattr(policy, "predict"):
            learned[name] = [statistics.fmean(policy.predict(se.ev)[0] for se in g) if g else None for g in groups]
    return results, learned, truth


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]

    def line(r):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))

    print(f"\n{title}")
    print(line(header))
    print("-" * len(line(header)))
    for r in rows:
        print(line(r))


def critical_false_trips(r) -> int:
    return sum(v for (_, cls, real), v in r.trips_by.items() if cls in ("fire", "smoke") and not real)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kamera + robot doğrulama: politika karşılaştırması (simülasyon)")
    ap.add_argument("--days", type=int, default=120, help="simüle edilecek gün sayısı (varsayılan 120)")
    ap.add_argument("--seeds", type=int, default=5, help="kaç farklı rastgele olay akışı denensin (varsayılan 5)")
    ap.add_argument("--period", type=int, default=15, help="dönemsel tablo için gün aralığı (varsayılan 15)")
    ap.add_argument("--ablation", action="store_true", help="önerilen sistemin bileşenlerini tek tek çıkar")
    ap.add_argument("--blind", action="store_true", help="K6'da gerçek ve yanlış tespitleri birebir aynı göster")
    args = ap.parse_args()
    if args.days <= RECENT_DAYS:
        ap.error(f"--days en az {RECENT_DAYS + 1} olmalı")

    names = ABLATION if args.ablation else MAIN
    jobs = [(names, args.days, seed, args.period, args.blind) for seed in range(args.seeds)]
    with ProcessPoolExecutor() as pool:
        runs = list(pool.map(run_seed, jobs))
    res = {n: [run[0][n] for run in runs] for n in names}
    days = args.days

    def avg(n, f):
        return statistics.fmean(f(r) for r in res[n])

    def sd(n, f):
        return statistics.stdev(f(r) for r in res[n]) if len(res[n]) > 1 else 0.0

    def cells(f, fmt):
        return [fmt(avg(n, f)) for n in names]

    def estimate(n):
        est = [r.estimated_missed for r in res[n]]
        if any(e is None for e in est):
            return "-"
        return f"{tr(statistics.fmean(est))} ({tr(avg(n, lambda r: r.ignored_real))})"

    sc = default_scenario(blind_spot=args.blind)
    print(f"Senaryo: {len(sc.cameras)} kamera, 1 robot, {days} gün, {args.seeds} farklı olay akışının ortalaması"
          + (" | ZOR KÖR NOKTA (--blind)" if args.blind else ""))
    header = ["", *[POLICY_NAMES[n] for n in names]]
    rows = [
        ["Toplam maliyet (puan)",
         *[f"{tr(avg(n, lambda r: r.total_cost))} ±{tr(sd(n, lambda r: r.total_cost))}" for n in names]],
        ["  kaçırılan olaylar", *cells(lambda r: r.cost["miss"], tr)],
        ["  robot görevleri", *cells(lambda r: r.cost["trip"], tr)],
        ["  insanlara yanlış alarm", *cells(lambda r: r.cost["false_alarm"], tr)],
        ["  gecikme + klip inceleme", *cells(lambda r: r.cost["delay"] + r.cost["review"], tr)],
        ["Robot görevi / gün", *cells(lambda r: r.trips / days, lambda x: tr(x, 1))],
        ["  yangın/duman yanlış tespitine giden",
         *cells(lambda r: 100 * critical_false_trips(r) / max(r.trips, 1), pct)],
        ["İnsanlara yanlış alarm / gün", *cells(lambda r: r.false_alarms / days, lambda x: tr(x, 1))],
        ["Klip incelemesi / gün", *cells(lambda r: r.reviews / days, lambda x: tr(x, 1))],
        ["Kaçırılan gerçek olay (adet)", *cells(lambda r: sum(r.missed.values()), tr)],
        *[[f"  {CLASS_NAMES_TR[k]} (kaçırılan / gerçek)",
           *[f"{tr(avg(n, lambda r, k=k: r.missed[k]), 1)} / {tr(avg(n, lambda r, k=k: r.real[k]), 1)}"
             for n in names]] for k in EVENT_CLASSES],
    ]
    if sc.drift is not None:
        cam = next(c.name for c in sc.cameras if c.id == sc.drift[0])
        rows.append([f"Kör nokta: {cam} {CLASS_NAMES_TR[sc.drift[1]].lower()}, {sc.drift[2]:.0f}. gün sonrası yakalanan",
                     *cells(lambda r: 100 * (1 - r.drift_missed / max(r.drift_real, 1)), pct)])
    rows.append(["Denetimden kaçırma tahmini (gerçek)", *[estimate(n) for n in names]])
    print_table("ÖZET", header, rows)
    print("Not: 'kaçırılan' sayılarını sahada göremezsiniz; yalnızca simülasyon bilir. Sahadaki tek pencere denetimlerdir.\n"
          "     Denetim tahmini kabadır: %3 denetimde bulunan her gerçek olay ~32 olayı temsil eder ve robotun\n"
          "     yanlışlıkla 'var' dediği durumlar tahmini yukarı çeker (yani tahmin temkinli tarafta kalır).")

    n_periods = len(res[names[0]][0].period_missed)
    rows = []
    for i in range(n_periods):
        a, b = i * args.period, min((i + 1) * args.period, days)
        mark = " *" if sc.drift is not None and a <= sc.drift[2] < b else ""
        rows.append([f"{a}-{b}. gün{mark}", *[tr(avg(n, lambda r, i=i: r.period_missed[i]), 1) for n in names]])
    print_table(f"{args.period} GÜNLÜK DÖNEMLERDE KAÇIRILAN GERÇEK OLAY", header, rows)
    if sc.drift is not None:
        print(f"* {sc.drift[2]:.0f}. günden itibaren sigara içenler depo arkasını kullanmaya başlıyor.")

    learners = [n for n in names if n in runs[0][1]]
    cam_names = {c.id: c.name for c in sc.cameras}
    rows = []
    for j, (cam, cls, night) in enumerate(CONTEXTS):
        truth = [run[2][j] for run in runs if run[2][j] is not None]
        if not truth:
            continue
        label = f"{cam_names[cam]} · {CLASS_NAMES_TR[cls].lower()} · {'gece' if night else 'gündüz'}"
        rows.append([label, pct(100 * statistics.fmean(t[0] for t in truth)),
                     pct(100 * statistics.fmean(t[1] for t in truth)),
                     *[pct(100 * statistics.fmean(run[1][n][j] for run in runs if run[1][n][j] is not None))
                       for n in learners]])
    print_table(f"MODEL NE ÖĞRENDİ? (son {RECENT_DAYS} gündeki olaylar, ortalama p(gerçek))",
                ["Bağlam", "Kamera güveni", "Gerçek oran", *[POLICY_NAMES[n] for n in learners]], rows)


if __name__ == "__main__":
    main()
