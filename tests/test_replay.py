import pytest

from gan_lab_tensorflow.replay import ReplayBuffer


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer[int](capacity=3, seed=1)
    buffer.add_many([1, 2, 3, 4])
    assert len(buffer) == 3
    assert buffer.sample(10) == [2, 3, 4]


def test_replay_buffer_rejects_bad_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer[int](capacity=0)

