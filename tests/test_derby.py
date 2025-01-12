"""Tests for the A/B derby: two engines racing on the same target."""

from __future__ import annotations

from gan_lab_tensorflow.live.derby import DerbySession


def test_combined_frame_shape():
    session = DerbySession(dataset="mixture", seed=7)
    frame = session._combined()
    assert frame["type"] == "derby"
    for side in ("left", "right"):
        for key in ("fake", "grid", "mmd", "coverage", "collapsed", "loss"):
            assert key in frame[side]
    assert frame["left"]["loss"] == "vanilla"
    assert frame["right"]["loss"] == "wasserstein"
    assert len(frame["extent"]) == 4


def test_identical_start_same_seed():
    """Same seed => identical initial generators (the 'fair race' premise)."""
    session = DerbySession(dataset="mixture", seed=7)
    frame = session._combined()
    assert frame["left"]["fake"] == frame["right"]["fake"]


def test_wasserstein_beats_vanilla_on_the_same_target():
    """After training, the vanilla side covers far less than the Wasserstein one."""
    session = DerbySession(dataset="mixture", seed=7)
    for _ in range(1800):
        session._left.train_step()
        session._right.train_step()
    frame = session._combined()
    assert frame["right"]["coverage"] > frame["left"]["coverage"] + 0.3


def test_controls_and_dataset_switch():
    session = DerbySession()
    assert session.state()["running"] is False
    session.play()
    assert session.state()["running"] is True
    assert session.toggle() is False

    session.set_dataset("ring")
    st = session.state()
    assert st["dataset"] == "ring"
    assert st["running"] is False  # switching target re-arms, paused

    session.set_dataset("bogus")  # ignored
    assert session.state()["dataset"] == "ring"
