"""Karar motoru testleri. Çalıştırmak için proje klasöründe:  python -m unittest"""

import csv
import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from engine import (Action, CameraEvent, DecisionEngine, EngineConfig, EventStore, Feedback, LearningConfig,
                    RobotVerdict)
from engine.labeling import label_from_robot
from engine.model import BayesianLogisticRegression, sigmoid

CAMS = ("K1", "K2")
DAY = 86400.0


def event(i, cls="smoking", conf=0.5, ts=0.0, hour=12.0, cam="K1", track=10.0) -> CameraEvent:
    return CameraEvent(f"e{i}", cam, cls, conf, ts, hour, track)


def label_all(engine, events, is_real) -> None:
    """Kritik sınıfta (duman) olaylar yok sayılmadığı için her olay açık kalır ve etiketlenebilir."""
    for ev in events:
        engine.decide(ev, eta_min=3.0)
        engine.record_human_label(ev.event_id, is_real(ev))


class ModelTests(unittest.TestCase):
    def test_recovers_true_weights(self):
        """Model, bilinen bir olasılık ilişkisini veriden doğru öğreniyor"""
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
        y = (rng.random(4000) < sigmoid(X @ np.array([-1.0, 2.0]))).astype(float)
        model = BayesianLogisticRegression(np.zeros(2), np.full(2, 10.0))
        model.fit(X, y)
        np.testing.assert_allclose(model.mean, [-1.0, 2.0], atol=0.15)
        self.assertTrue(np.all(np.diag(model.cov) < 0.01))

    def test_no_data_means_prior(self):
        """Hiç etiket yokken model başlangıç bilgisine döner"""
        model = BayesianLogisticRegression(np.array([0.5, 1.0]), np.array([2.0, 3.0]))
        model.fit(np.zeros((0, 2)), np.zeros(0))
        np.testing.assert_allclose(model.mean, [0.5, 1.0])
        np.testing.assert_allclose(np.diag(model.cov), [2.0, 3.0])


class DecisionTests(unittest.TestCase):
    def test_untrained_engine_trusts_camera_confidence(self):
        """Eğitilmemiş motor kameranın güven skorunu doğru kabul ediyor"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS))
        for conf in (0.2, 0.5, 0.8):
            self.assertAlmostEqual(engine.predict(event(0, conf=conf))[0], conf, places=6)

    def test_learns_camera_specific_false_alarms(self):
        """Aynı sınıfta kameraya özgü yanlış alarmları ayırt ediyor (buhar ve gerçek duman)"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS), seed=0)
        rng = np.random.default_rng(0)
        evs = [event(i, cls="smoke", conf=float(rng.uniform(0.3, 0.9)), cam="K1" if i % 2 else "K2")
               for i in range(200)]
        label_all(engine, evs, lambda ev: ev.camera_id == "K2")  # K1'deki "duman" hep buhar çıkıyor
        engine.refit()
        self.assertLess(engine.predict(event(998, cls="smoke", conf=0.6, cam="K1"))[0], 0.1)
        self.assertGreater(engine.predict(event(999, cls="smoke", conf=0.6, cam="K2"))[0], 0.9)

    def test_safety_rules_override_model(self):
        """Güvenlik kuralları modelden üstün: duman yok sayılmaz, yüksek güvenli yangın hep alarm"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS), seed=0)
        label_all(engine, [event(i, cls="smoke", conf=0.4) for i in range(60)], lambda ev: False)
        self.assertLess(engine.predict(event(999, cls="smoke", conf=0.4))[0], 0.05)
        actions = {engine.decide(event(1000 + i, cls="smoke", conf=0.4), eta_min=3.0).action for i in range(200)}
        self.assertNotIn(Action.IGNORE, actions)  # model "buhar" dese de kritik olay yok sayılmaz
        fire = engine.decide(event(2000, cls="fire", conf=0.9), eta_min=3.0)
        self.assertEqual(fire.action, Action.ALARM)
        self.assertEqual(fire.propensity, 1.0)

    def test_no_robot_means_no_dispatch(self):
        """Robot müsait değilken robota görev verilmiyor"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS), seed=0)
        for i, conf in enumerate(np.linspace(0.05, 0.95, 50)):
            for cls in ("fire", "smoke", "smoking", "open_door"):
                d = engine.decide(event(f"{cls}{i}", cls=cls, conf=float(conf)), eta_min=None)
                self.assertNotEqual(d.action, Action.DISPATCH)

    def test_audit_sends_robot_to_some_ignored_events(self):
        """Denetim, yok sayılan olayların bir kısmına robot gönderiyor; kaçırma tahmini doğru"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS, learning=LearningConfig(audit_rate=0.5)), seed=0)
        decisions = [engine.decide(event(i, conf=0.05), eta_min=3.0) for i in range(400)]
        audits = [d for d in decisions if d.reason == "denetim"]
        self.assertTrue(150 < len(audits) < 250, len(audits))
        self.assertTrue(all(d.action is Action.DISPATCH and d.propensity == 0.5 for d in audits))
        ignored = [d for d in decisions if d.action is Action.IGNORE]
        self.assertTrue(all(d.propensity == 0.5 for d in ignored))
        for d in audits[:10]:  # denetimde bulunan her gerçek olay (1-a)/a = 1 kaçırılan olayı temsil eder
            engine.record_human_label(d.event_id, True)
        self.assertAlmostEqual(engine.estimated_missed(), 10.0)

    def test_estimate_needs_audits(self):
        """Denetim kapalıyken kaçırma tahmini üretilmiyor"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS, learning=LearningConfig(audit_rate=0.0)))
        self.assertIsNone(engine.estimated_missed())

    def test_thompson_explores_when_uncertain(self):
        """Thompson seçeneği belirsiz durumda keşif yapıyor, varsayılan karar yapmıyor"""
        def dispatches(exploration):
            cfg = EngineConfig(cameras=CAMS, learning=LearningConfig(exploration=exploration, audit_rate=0.0))
            engine = DecisionEngine(cfg, seed=1)
            return sum(engine.decide(event(i, conf=0.1), eta_min=3.0).action is not Action.IGNORE
                       for i in range(300))

        self.assertGreater(dispatches("thompson"), 0)
        self.assertEqual(dispatches("greedy"), 0)

    def test_old_labels_fade(self):
        """Eski etiketler zamanla unutuluyor, modelin belirsizliği yeniden artıyor"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS, learning=LearningConfig(half_life_days=10)), seed=0)
        label_all(engine, [event(i, cls="smoke", conf=0.5) for i in range(40)], lambda ev: False)
        engine.refit()
        probe = event(999, cls="smoke", conf=0.5)
        sd_fresh = engine.predict(probe)[1]
        engine.decide(event(1000, cls="smoke", conf=0.5, ts=100 * DAY), eta_min=3.0)  # 100 gün geçti
        engine.refit()
        self.assertGreater(engine.predict(probe)[1], 1.5 * sd_fresh)


class LabelingTests(unittest.TestCase):
    def test_not_found_is_evidence_only_when_robot_arrived_in_time(self):
        '''"Robot bulamadı" yalnızca robot zamanında vardıysa kanıt sayılıyor'''
        careful, naive = LearningConfig(), LearningConfig(labeling="naive")
        late, fresh = 10.0, 1.0
        cases = [
            ("smoking", RobotVerdict.NOT_FOUND, late, careful, Feedback(None, True)),
            ("smoking", RobotVerdict.NOT_FOUND, fresh, careful, Feedback(0, False)),
            ("smoking", RobotVerdict.NOT_FOUND, late, naive, Feedback(0, False)),
            ("fire", RobotVerdict.NOT_FOUND, late, careful, Feedback(0, False)),  # yangın kendiliğinden geçmez
            ("smoking", RobotVerdict.CONFIRMED, late, careful, Feedback(1, False)),
            ("open_door", RobotVerdict.INCONCLUSIVE, fresh, careful, Feedback(None, True)),
            ("open_door", RobotVerdict.INCONCLUSIVE, fresh, naive, Feedback(None, False)),
        ]
        for cls, verdict, delay, cfg, expected in cases:
            with self.subTest(cls=cls, verdict=verdict, delay=delay, labeling=cfg.labeling):
                self.assertEqual(label_from_robot(cls, verdict, delay, cfg), expected)

    def test_review_label_reaches_model(self):
        """Operatörün klip incelemesi modele ulaşıyor"""
        engine = DecisionEngine(EngineConfig(cameras=CAMS, learning=LearningConfig(refit_every=1)), seed=0)
        ev = event(1, cls="smoke", conf=0.5)
        engine.decide(ev, eta_min=3.0)
        fb = engine.record_robot_result(ev.event_id, RobotVerdict.INCONCLUSIVE, arrival_delay_min=2.0)
        self.assertEqual(fb, Feedback(None, True))
        before = engine.predict(ev)[0]
        engine.record_human_label(ev.event_id, True)
        self.assertGreater(engine.predict(ev)[0], before)


class StoreTests(unittest.TestCase):
    def test_restart_restores_model_and_pending_events(self):
        """Servis yeniden başlayınca model ve bekleyen olaylar kayıttan geri geliyor"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "olaylar.db")
            store = EventStore(path)
            engine = DecisionEngine(EngineConfig(cameras=CAMS), store=store, seed=0)
            label_all(engine, [event(i, cls="smoke", conf=0.6) for i in range(30)], lambda ev: ev.event_id < "e2")
            engine.decide(event(100, cls="smoke", conf=0.6), eta_min=3.0)  # sonucu henüz gelmedi
            engine.refit()
            probe = event(999, cls="smoke", conf=0.6)
            before = engine.predict(probe)
            store.close()

            store = EventStore(path)
            again = DecisionEngine(EngineConfig(cameras=CAMS), store=store, seed=0)
            np.testing.assert_allclose(again.predict(probe), before, atol=1e-6)
            again.record_human_label("e100", True)  # yeniden başladıktan sonra da etiketlenebilir
            self.assertEqual(again.store.export_training_manifest(os.path.join(tmp, "m.csv")), 31)
            store.close()

    def test_shadow_mode_learns_from_examples(self):
        """Gölge modda ve tatbikatlardan gelen örneklerle öğreniyor"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "olaylar.db")
            store = EventStore(path)
            cfg = EngineConfig(cameras=CAMS, learning=LearningConfig(audit_rate=0.0, refit_every=1))
            engine = DecisionEngine(cfg, store=store, seed=0)
            ev = event(1, cls="smoking", conf=0.05)
            self.assertEqual(engine.decide(ev, eta_min=3.0).action, Action.IGNORE)  # motor sadece izliyor
            before = engine.predict(ev)[0]
            engine.add_example(ev, is_real=True)       # mevcut sistemin robotu gitti, sigara varmış
            engine.add_example(event(2, cls="fire", conf=0.6), is_real=True, source="tatbikat")
            self.assertGreater(engine.predict(ev)[0], before)
            after = engine.predict(ev)
            store.close()

            store = EventStore(path)
            again = DecisionEngine(cfg, store=store, seed=0)
            np.testing.assert_allclose(again.predict(ev), after, atol=1e-6)
            self.assertEqual(store.export_training_manifest(os.path.join(tmp, "m.csv")), 2)
            store.close()

    def test_manifest_marks_hard_examples(self):
        """Kamera modelini yeniden eğitmek için zor örnekler doğru işaretleniyor"""
        with tempfile.TemporaryDirectory() as tmp:
            store = EventStore()
            engine = DecisionEngine(EngineConfig(cameras=CAMS), store=store, seed=0)
            for ev, real in ((event(1, cls="smoke", conf=0.8), False),   # kamera emin, olay yok
                             (event(2, cls="smoke", conf=0.3), True),    # kamera emin değil, olay var
                             (event(3, cls="smoke", conf=0.8), True)):
                engine.decide(replace(ev, snapshot=f"/goruntu/{ev.event_id}.jpg"), eta_min=3.0)
                engine.record_human_label(ev.event_id, real)
            path = os.path.join(tmp, "m.csv")
            store.export_training_manifest(path)
            with open(path, encoding="utf-8") as f:
                rows = {r["event_id"]: r for r in csv.DictReader(f)}
            self.assertEqual(rows["e1"]["kind"], "hard_negative")
            self.assertEqual(rows["e2"]["kind"], "hard_positive")
            self.assertEqual(rows["e3"]["kind"], "easy")
            self.assertEqual(rows["e1"]["snapshot"], "/goruntu/e1.jpg")
            store.close()


if __name__ == "__main__":
    unittest.main()
