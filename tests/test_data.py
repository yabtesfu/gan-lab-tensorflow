from gan_lab_tensorflow.data import batches, sample_curve, sample_mixture, summarize


def test_sample_curve_is_reproducible():
    left = sample_curve(5, seed=7)
    right = sample_curve(5, seed=7)
    assert left == right


def test_batches_keep_tail_items():
    items = sample_curve(5, seed=1)
    chunks = list(batches(items, 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_mixture_summary_has_expected_count():
    summary = summarize(sample_mixture(25, seed=3))
    assert summary.count == 25

