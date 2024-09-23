import pytest

from gan_lab_tensorflow.conditional import ConditionalSpec, attach_labels, one_hot


def test_one_hot_encodes_labels():
    assert one_hot([0, 2], 3) == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_attach_labels_extends_samples():
    samples = [(0.1, 0.2)]
    assert attach_labels(samples, [1], 2) == [(0.1, 0.2, 0.0, 1.0)]


def test_conditional_spec_validates_class_count():
    with pytest.raises(ValueError):
        ConditionalSpec(noise_dim=4, num_classes=1).validate()

