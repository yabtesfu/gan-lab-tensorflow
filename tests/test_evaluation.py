from gan_lab_tensorflow.data import sample_curve
from gan_lab_tensorflow.evaluation import maximum_mean_discrepancy, nearest_neighbor_precision


def test_mmd_zero_for_identical_samples():
    points = sample_curve(8, seed=4)
    assert maximum_mean_discrepancy(points, points) == 0


def test_nearest_neighbor_precision_bounds():
    real = sample_curve(12, seed=2)
    generated = sample_curve(12, seed=3)
    precision = nearest_neighbor_precision(real, generated, radius=10.0)
    assert 0 <= precision <= 1

