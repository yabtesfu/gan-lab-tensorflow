from gan_lab_tensorflow.data import sample_curve
from gan_lab_tensorflow.metrics import coverage_ratio, moment_distance


def test_moment_distance_zero_for_same_points():
    points = sample_curve(30, seed=10)
    assert moment_distance(points, points) == 0


def test_coverage_ratio_stays_in_unit_interval():
    real = sample_curve(40, seed=1)
    generated = sample_curve(40, seed=2)
    ratio = coverage_ratio(real, generated)
    assert 0 <= ratio <= 1

