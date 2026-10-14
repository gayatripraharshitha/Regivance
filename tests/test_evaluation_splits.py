
import pytest

from src.evaluation.splits import generate_walk_forward_splits


def test_generates_twelve_chronological_folds():
    splits = generate_walk_forward_splits(n_samples=2000)

    assert len(splits) == 12

    for split in splits:
        assert split.train.stop - split.train.start == 1260
        assert split.validation.stop - split.validation.start == 252
        assert split.test.stop - split.test.start == 21

        # The five-observation gaps must be preserved.
        assert split.validation.start - split.train.stop == 5
        assert split.test.start - split.validation.stop == 5

        # All windows must be ordered and non-overlapping.
        assert split.train.stop <= split.validation.start
        assert split.validation.stop <= split.test.start


def test_test_blocks_cover_the_final_period_without_overlap():
    n_samples = 2000
    splits = generate_walk_forward_splits(n_samples=n_samples)

    assert splits[0].test.start == n_samples - 252
    assert splits[-1].test.stop == n_samples

    for previous, current in zip(splits, splits[1:]):
        assert previous.test.stop == current.test.start


def test_rejects_insufficient_data():
    with pytest.raises(ValueError, match="Not enough observations"):
        generate_walk_forward_splits(n_samples=1773)


def test_rejects_overlapping_test_configuration():
    with pytest.raises(ValueError, match="test_size must equal step_size"):
        generate_walk_forward_splits(
            n_samples=2000,
            test_size=42,
            step_size=21,
        )
