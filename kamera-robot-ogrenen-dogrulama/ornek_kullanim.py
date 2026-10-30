"""Karar motorunun gerçek sisteme nasıl bağlanacağını gösteren küçük örnek.

    python ornek_kullanim.py

Gerçek sistemde kamera olayları kamera analitiğinden (ör. MQTT), robot sonuçları robot filosundan
(ör. ROS 2 action sonucu), operatör cevapları da izleme ekranından gelir. Burada mesajları elle üretiyoruz.
"""

from __future__ import annotations

import os
import tempfile

from engine import Action, CameraEvent, DecisionEngine, EngineConfig, EventStore, RobotVerdict

out_dir = tempfile.mkdtemp(prefix="dogrulama_")
engine = DecisionEngine(EngineConfig(cameras=("K1", "K6")), store=EventStore(os.path.join(out_dir, "olaylar.db")),
                        seed=0)


def on_camera_event(msg: dict, robot_eta_min: float | None) -> Action:
    """Kamera analitiğinden olay geldi. robot_eta_min: filo yöneticisinin verdiği tahmini varış süresi."""
    ev = CameraEvent(event_id=msg["id"], camera_id=msg["camera"], cls=msg["class"], conf=msg["conf"],
                     ts=msg["ts"], hour=msg["hour"], track_sec=msg["track_sec"], snapshot=msg.get("snapshot"))
    d = engine.decide(ev, eta_min=robot_eta_min)
    print(f"{ev.event_id}: {ev.camera_id} {ev.cls:9s} güven={ev.conf:.2f} -> {d.action.value:8s} "
          f"p(gerçek)={d.p_real:.2f}  [{d.reason}]")
    return d.action  # DISPATCH: robota görev gönder, ALARM: güvenlik ekibine bildir


def on_robot_result(event_id: str, verdict: RobotVerdict, arrival_delay_min: float) -> None:
    fb = engine.record_robot_result(event_id, verdict, arrival_delay_min)
    if verdict is RobotVerdict.CONFIRMED:
        print("   robot doğruladı -> güvenlik ekibine bildir")
    elif fb.needs_review:
        print(f"   robot {arrival_delay_min:.0f} dk sonra vardı, bir şey göremedi -> klibi operatöre gönder")
    else:
        print("   robot: sorun yok, devriyeye dönüyor")


def on_operator_feedback(event_id: str, is_real: bool) -> None:
    engine.record_human_label(event_id, is_real)
    print(f"   operatör: {'olay gerçek' if is_real else 'yanlış alarm'}")


T0 = 14 * 3600.0  # bugün 14:00 (gerçek sistemde epoch saniye)

if on_camera_event({"id": "olay-1", "camera": "K6", "class": "smoking", "conf": 0.45, "ts": T0, "hour": 14.0,
                    "track_sec": 30.0, "snapshot": "/kayit/olay-1.jpg"}, robot_eta_min=4.0) is Action.DISPATCH:
    on_robot_result("olay-1", RobotVerdict.NOT_FOUND, arrival_delay_min=4.2)
    on_operator_feedback("olay-1", is_real=True)  # klipte sigara içen biri görülüyor

on_camera_event({"id": "olay-2", "camera": "K1", "class": "fire", "conf": 0.91, "ts": T0 + 600, "hour": 14.2,
                 "track_sec": 12.0}, robot_eta_min=2.0)
on_operator_feedback("olay-2", is_real=False)  # kaynakçı çalışıyormuş

if on_camera_event({"id": "olay-3", "camera": "K1", "class": "smoke", "conf": 0.50, "ts": T0 + 900, "hour": 14.3,
                    "track_sec": 6.0}, robot_eta_min=2.0) is Action.DISPATCH:
    on_robot_result("olay-3", RobotVerdict.NOT_FOUND, arrival_delay_min=2.1)  # buhar çıkmış

on_camera_event({"id": "olay-4", "camera": "K1", "class": "smoking", "conf": 0.10, "ts": T0 + 1200, "hour": 14.3,
                 "track_sec": 3.0}, robot_eta_min=2.0)

# Kontrollü duman tatbikatı: motorun kararından bağımsız etiketli örnek
engine.add_example(CameraEvent("tatbikat-1", "K6", "smoke", 0.35, T0 + 3600, 15.0, 40.0), is_real=True,
                   source="tatbikat")
engine.refit()

manifest = os.path.join(out_dir, "egitim_listesi.csv")
n = engine.store.export_training_manifest(manifest)
print(f"\nKayıtlar: {out_dir}\nKamera modelini yeniden eğitmek için {n} doğrulanmış örnek: {manifest}")
