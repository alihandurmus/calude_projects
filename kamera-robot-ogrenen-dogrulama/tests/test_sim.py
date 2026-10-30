"""Simülasyon testleri. Çalıştırmak için proje klasöründe:  python -m unittest"""

import unittest

from sim import build_policy, default_scenario, generate_events, simulate

DAYS = 10


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = default_scenario()
        cls.cams = tuple(c.id for c in cls.sc.cameras)
        cls.events = generate_events(cls.sc, DAYS, seed=0)

    def test_short_run_is_consistent(self):
        """Simülasyon sayıları tutarlı: kaçırılan ≤ gerçek, maliyetlerin toplamı tutuyor"""
        for name in ("sabit", "basit", "onerilen"):
            with self.subTest(policy=name):
                r = simulate(build_policy(name, self.cams, seed=0), self.sc, self.events, DAYS)
                self.assertGreater(r.trips, 0)
                self.assertEqual(r.trips, sum(r.trips_by.values()))
                for cls, n in r.missed.items():
                    self.assertLessEqual(n, r.real[cls])
                self.assertAlmostEqual(r.total_cost, sum(r.period_cost))
                self.assertEqual(sum(r.missed.values()), sum(r.period_missed))

    def test_same_seed_same_result(self):
        """Aynı tohumla simülasyon aynı sonucu veriyor"""
        a = simulate(build_policy("onerilen", self.cams, seed=3), self.sc, self.events, DAYS)
        b = simulate(build_policy("onerilen", self.cams, seed=3), self.sc, generate_events(self.sc, DAYS, 0), DAYS)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.missed, b.missed)


if __name__ == "__main__":
    unittest.main()
