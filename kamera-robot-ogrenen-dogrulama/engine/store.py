"""Olay, karar ve doğrulama kayıtları (SQLite).

İki öğrenme döngüsünün ortak veri kaynağı:
  1) Karar modeli bu etiketlerden yeniden kurulur (servis yeniden başlasa da öğrendikleri kaybolmaz).
  2) Kamera modelini yeniden eğitmek için robotun doğruladığı "zor örnekler" buradan dışa aktarılır.
"""

from __future__ import annotations

import csv
import json
import sqlite3

from .types import CameraEvent, Decision, RobotVerdict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    ts           REAL NOT NULL,
    hour         REAL NOT NULL,
    camera_id    TEXT NOT NULL,
    cls          TEXT NOT NULL,
    conf         REAL NOT NULL,
    track_sec    REAL NOT NULL,
    snapshot     TEXT,
    bbox         TEXT,
    action       TEXT NOT NULL,   -- ignore / dispatch / alarm; motor dışı etiketli örneklerde 'example'
    p_real       REAL,
    propensity   REAL,
    reason       TEXT NOT NULL,
    verdict      TEXT,
    needs_review INTEGER NOT NULL DEFAULT 0,
    label        INTEGER,
    label_source TEXT
)
"""
_EVENT_COLS = "event_id, camera_id, cls, conf, ts, hour, track_sec, snapshot, bbox"


class EventStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    def log_decision(self, ev: CameraEvent, d: Decision) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO events (event_id, ts, hour, camera_id, cls, conf, track_sec, snapshot,"
                " bbox, action, p_real, propensity, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ev.event_id, ev.ts, ev.hour, ev.camera_id, ev.cls, ev.conf, ev.track_sec, ev.snapshot,
                 json.dumps(ev.bbox) if ev.bbox else None, d.action.value, d.p_real, d.propensity, d.reason),
            )

    def log_example(self, ev: CameraEvent) -> None:
        """Motorun karar vermediği olayı kaydeder; gölge modda karar zaten kayıtlıysa ona dokunmaz."""
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO events (event_id, ts, hour, camera_id, cls, conf, track_sec, snapshot, bbox,"
                " action, reason) VALUES (?,?,?,?,?,?,?,?,?,'example','örnek')",
                (ev.event_id, ev.ts, ev.hour, ev.camera_id, ev.cls, ev.conf, ev.track_sec, ev.snapshot,
                 json.dumps(ev.bbox) if ev.bbox else None),
            )

    def log_verdict(self, event_id: str, verdict: RobotVerdict, needs_review: bool) -> None:
        with self.db:
            self.db.execute("UPDATE events SET verdict = ?, needs_review = ? WHERE event_id = ?",
                            (verdict.value, int(needs_review), event_id))

    def log_label(self, event_id: str, label: int, source: str) -> None:
        with self.db:
            self.db.execute("UPDATE events SET label = ?, label_source = ?, needs_review = 0 WHERE event_id = ?",
                            (label, source, event_id))

    def history(self) -> tuple[list[tuple[CameraEvent, int]], list[CameraEvent]]:
        """(etiketli olaylar, sonucu hâlâ beklenen olaylar): motor yeniden başlarken kullanılır."""
        labeled = [(self._event(r[:-1]), r[-1]) for r in self.db.execute(
            f"SELECT {_EVENT_COLS}, label FROM events WHERE label IS NOT NULL ORDER BY ts")]
        pending = [self._event(r) for r in self.db.execute(
            f"SELECT {_EVENT_COLS} FROM events WHERE label IS NULL AND action != 'ignore'"
            " AND (verdict IS NULL OR needs_review = 1) ORDER BY ts")]
        return labeled, pending

    def audits(self) -> list[tuple[str, float, float, int | None]]:
        """Denetim kararları: (event_id, ts, seçilme olasılığı, etiket)."""
        return self.db.execute(
            "SELECT event_id, ts, propensity, label FROM events WHERE reason = 'denetim' ORDER BY ts").fetchall()

    def export_training_manifest(self, path: str, hard_margin: float = 0.5) -> int:
        """Doğrulanmış olayları kamera modelini yeniden eğitmek için CSV'ye yazar; satır sayısını döner.

        kind sütunu:
          hard_negative  kamera emindi ama olay yoktu (ör. buharı duman sandı): en değerli örnekler
          hard_positive  kamera emin değildi ama olay vardı (ör. karanlıkta sigara)
          easy           kameranın zaten doğru bildiği örnekler
        """
        rows = self.db.execute(
            "SELECT event_id, ts, camera_id, cls, conf, snapshot, bbox, label, label_source"
            " FROM events WHERE label IS NOT NULL ORDER BY ts").fetchall()
        with open(path, "w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(["event_id", "ts", "camera_id", "cls", "conf", "snapshot", "bbox", "label",
                          "label_source", "kind"])
            for r in rows:
                conf, label = r[4], r[7]
                if label == 0 and conf >= hard_margin:
                    kind = "hard_negative"
                elif label == 1 and conf < hard_margin:
                    kind = "hard_positive"
                else:
                    kind = "easy"
                out.writerow([*r, kind])
        return len(rows)

    @staticmethod
    def _event(row) -> CameraEvent:
        event_id, camera_id, cls, conf, ts, hour, track_sec, snapshot, bbox = row
        return CameraEvent(event_id, camera_id, cls, conf, ts, hour, track_sec, snapshot,
                           tuple(json.loads(bbox)) if bbox else None)
