"""Tests for the SQLite run registry and the model-serving path."""

from __future__ import annotations

import numpy as np

from gan_lab_tensorflow.live.engine import LiveGan, LiveGanConfig, sample_generator
from gan_lab_tensorflow.live.registry import RunRegistry


def _run(steps: int = 120, **cfg):
    gan = LiveGan(LiveGanConfig(**cfg))
    for _ in range(steps):
        gan.train_step()
    return gan


def _save(reg: RunRegistry, gan: LiveGan) -> int:
    frame = gan.snapshot().to_dict()
    return reg.save(
        dataset=gan.config.dataset, loss=gan.config.loss, seed=gan.config.seed,
        frame=frame, metrics_history=[{"s": 0, "mmd": 0.4}, {"s": 50, "mmd": 0.2}],
        state=gan.export_state(),
    )


def test_save_list_get_roundtrip(tmp_path):
    reg = RunRegistry(tmp_path / "runs.db")
    gan = _run(dataset="mixture", loss="wasserstein", seed=3)
    rid = _save(reg, gan)

    summaries = reg.list()
    assert len(summaries) == 1
    s = summaries[0]
    assert s["id"] == rid and s["dataset"] == "mixture" and s["loss"] == "wasserstein"
    assert isinstance(s["collapsed"], bool)

    full = reg.get(rid)
    assert full is not None
    assert "state" in full and "metrics" in full
    assert len(full["metrics"]) == 2
    assert full["state"]["noise_dim"] == gan.config.noise_dim


def test_get_missing_returns_none(tmp_path):
    reg = RunRegistry(tmp_path / "runs.db")
    assert reg.get(999) is None


def test_delete(tmp_path):
    reg = RunRegistry(tmp_path / "runs.db")
    rid = _save(reg, _run())
    assert reg.delete(rid) is True
    assert reg.get(rid) is None
    assert reg.delete(rid) is False


def test_list_orders_newest_first(tmp_path):
    reg = RunRegistry(tmp_path / "runs.db")
    a = _save(reg, _run(seed=1))
    b = _save(reg, _run(seed=2))
    ids = [r["id"] for r in reg.list()]
    assert ids == [b, a]


def test_serving_reloads_generator(tmp_path):
    """A saved run can be reloaded and sampled -- the /generate contract."""
    reg = RunRegistry(tmp_path / "runs.db")
    gan = _run(300, dataset="mixture", loss="wasserstein", seed=5)
    rid = _save(reg, gan)

    state = reg.get(rid)["state"]
    served = sample_generator(state, 200, seed=1)
    assert len(served["points"]) == 200
    assert len(served["extent"]) == 4
    assert all(np.isfinite(p).all() for p in served["points"])

    # Deterministic for a fixed seed, varied otherwise.
    again = sample_generator(state, 200, seed=1)
    assert served["points"] == again["points"]
    other = sample_generator(state, 200, seed=2)
    assert other["points"] != served["points"]

    # Served samples should resemble what the live generator produces.
    live = np.array(gan.snapshot().to_dict()["fake"])
    srv = np.array(served["points"])
    assert np.allclose(live.mean(axis=0), srv.mean(axis=0), atol=1.5)
