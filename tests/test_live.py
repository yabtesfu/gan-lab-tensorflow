"""Tests for the real-time observatory engine and session control.

These exercise the dependency-light NumPy backend only (no TensorFlow, no web
server), so they run in CI on the same footing as the rest of the suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from gan_lab_tensorflow.live.engine import LiveGan, LiveGanConfig
from gan_lab_tensorflow.live.session import DEMO_COLLAPSE, TrainingSession


def _train(config: LiveGanConfig, steps: int) -> LiveGan:
    gan = LiveGan(config)
    for _ in range(steps):
        gan.train_step()
    return gan


# -- learning ---------------------------------------------------------------
def test_training_reduces_mmd_on_quadratic():
    """The generator should measurably approach the target distribution."""
    gan = LiveGan(LiveGanConfig(dataset="quadratic", loss="vanilla", seed=1))
    start = gan.snapshot().mmd
    for _ in range(900):
        gan.train_step()
    end = gan.snapshot().mmd
    assert end < start * 0.7
    assert end < 0.3


def test_wasserstein_covers_the_mixture():
    """Wasserstein on the 3-mode mixture should reach high coverage."""
    gan = _train(LiveGanConfig(dataset="mixture", loss="wasserstein", seed=42), 1200)
    frame = gan.snapshot()
    assert frame.coverage > 0.85


# -- determinism ------------------------------------------------------------
def test_runs_are_deterministic():
    """Same seed + config + step count => identical generated points.

    This is what makes Demo Mode reproducible; snapshots must not perturb the
    training RNG.
    """
    a = _train(LiveGanConfig(seed=5), 200)
    b = _train(LiveGanConfig(seed=5), 200)
    # Snapshot on `a` a few extra times: it must not change the trajectory.
    for _ in range(5):
        a.snapshot()
    a.train_step()
    b.train_step()
    assert np.allclose(np.array(a.snapshot().fake_points), np.array(b.snapshot().fake_points))


# -- steering ---------------------------------------------------------------
def test_nonstructural_change_keeps_weights():
    gan = _train(LiveGanConfig(learning_rate=1e-3), 50)
    before = gan.step
    rebuilt = gan.apply_config(LiveGanConfig(learning_rate=5e-3))
    assert rebuilt is False
    assert gan.step == before  # training continues, no reset


def test_structural_change_rebuilds():
    gan = _train(LiveGanConfig(dataset="mixture"), 50)
    rebuilt = gan.apply_config(LiveGanConfig(dataset="sine"))
    assert rebuilt is True
    assert gan.step == 0


def test_config_clamps_and_validates():
    cfg = LiveGanConfig(learning_rate=99.0, d_steps=42).clamp()
    assert cfg.learning_rate <= 1e-1
    assert cfg.d_steps <= 5
    with pytest.raises(ValueError):
        LiveGanConfig(loss="nonsense").clamp()
    with pytest.raises(ValueError):
        LiveGanConfig(dataset="nope").clamp()


# -- telemetry frame --------------------------------------------------------
def test_snapshot_shape_and_serialisation():
    frame = LiveGan(LiveGanConfig()).snapshot()
    payload = frame.to_dict()
    assert payload["type"] == "frame"
    assert payload["grid"]["n"] == 32
    assert len(payload["grid"]["values"]) == 32 * 32
    assert len(payload["real"]) == LiveGan._VIZ_POINTS
    assert len(payload["extent"]) == 4
    for key in ("step", "genLoss", "discLoss", "mmd", "coverage", "precision", "collapsed"):
        assert key in payload


# -- collapse detection -----------------------------------------------------
def test_demo_preset_collapses():
    """The deterministic demo preset must actually trigger the collapse alarm."""
    gan = _train(DEMO_COLLAPSE, 1400)
    frame = None
    for _ in range(5):  # the detector needs a short streak of bad frames
        frame = gan.snapshot()
    assert frame.coverage < 0.5
    assert frame.collapsed is True


# -- session control (no background thread) ---------------------------------
def test_session_controls_are_thread_safe_api():
    session = TrainingSession(LiveGanConfig(seed=3))
    assert session.state()["running"] is False
    session.play()
    assert session.state()["running"] is True
    assert session.toggle() is False  # flips back to paused

    session.apply_control({"action": "config", "value": {"loss": "vanilla", "learning_rate": 2e-3}})
    state = session.state()
    assert state["config"]["loss"] == "vanilla"
    assert abs(state["config"]["learning_rate"] - 2e-3) < 1e-9


def test_session_load_demo():
    session = TrainingSession()
    session.load_demo()
    cfg = session.state()["config"]
    assert cfg["dataset"] == "mixture"
    assert cfg["loss"] == "vanilla"
    assert session.latest_frame is not None
