import pytest

from gan_lab_tensorflow.augment import AugmentationPolicy, adapt_probability, augment_points


def test_augment_points_is_deterministic_with_seed():
    points = [(1.0, 2.0), (3.0, 4.0)]
    policy = AugmentationPolicy(probability=1.0, jitter_std=0.01)

    assert augment_points(points, policy, seed=7) == augment_points(points, policy, seed=7)
    assert augment_points(points, policy, seed=7) != points


def test_augmentation_policy_validates_scale_range():
    with pytest.raises(ValueError):
        AugmentationPolicy(x_scale_min=1.2, x_scale_max=0.8).validate()


def test_adapt_probability_moves_toward_target_accuracy():
    assert adapt_probability(0.1, discriminator_real_accuracy=0.9) == pytest.approx(0.12)
    assert adapt_probability(0.1, discriminator_real_accuracy=0.2) == pytest.approx(0.08)
    assert adapt_probability(0.1, discriminator_real_accuracy=0.6) == pytest.approx(0.1)
