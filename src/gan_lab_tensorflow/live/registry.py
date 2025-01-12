"""A tiny SQLite registry of training runs.

Every saved run captures its configuration, seed, final metrics, the full
streamed metric history, and a serialized generator (see
``engine.LiveGan.export_state``). That makes a run *reproducible* -- you can
list what you ran, compare final coverage/MMD across runs, and reload any
generator to serve fresh samples from it without retraining.

Deliberately dependency-free: the standard-library ``sqlite3`` module only. A
fresh connection is opened per call so the registry is safe to touch from the
web thread and the training thread alike.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

_SUMMARY_COLS = ["id", "created", "dataset", "loss", "seed", "step", "mmd", "coverage", "precision", "collapsed"]


class RunRegistry:
    def __init__(self, path: str | Path = "runs.db"):
        self.path = str(path)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    loss TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    mmd REAL NOT NULL,
                    coverage REAL NOT NULL,
                    precision REAL NOT NULL,
                    collapsed INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )

    def save(self, *, dataset: str, loss: str, seed: int, frame: dict, metrics_history: list, state: dict) -> int:
        """Persist a run and return its new id."""
        created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs
                  (created, dataset, loss, seed, step, mmd, coverage, precision, collapsed, state_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created, dataset, loss, int(seed), int(frame["step"]),
                    float(frame["mmd"]), float(frame["coverage"]), float(frame["precision"]),
                    1 if frame.get("collapsed") else 0,
                    json.dumps(state), json.dumps(metrics_history),
                ),
            )
            return int(cur.lastrowid)

    def list(self, limit: int = 50) -> list[dict]:
        """Run summaries, newest first (no heavy state/history payloads)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_SUMMARY_COLS)} FROM runs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._summary(r) for r in rows]

    def get(self, run_id: int) -> dict | None:
        """A full run including the generator state and metric history."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),)).fetchone()
        if row is None:
            return None
        out = self._summary(row)
        out["state"] = json.loads(row["state_json"])
        out["metrics"] = json.loads(row["metrics_json"])
        return out

    def delete(self, run_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
            return cur.rowcount > 0

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict:
        d = {k: row[k] for k in _SUMMARY_COLS}
        d["collapsed"] = bool(d["collapsed"])
        return d
