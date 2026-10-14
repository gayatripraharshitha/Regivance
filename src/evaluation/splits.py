
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    """Position-based train, validation, and test windows for one fold."""

    fold: int
    train: slice
    validation: slice
    test: slice


def generate_walk_forward_splits(
    n_samples: int,
    train_size: int = 1260,
    validation_size: int = 252,
    test_size: int = 21,
    step_size: int = 21,
    final_test_size: int = 252,
    embargo: int = 5,
) -> list[WalkForwardSplit]:
    """
    Generate chronological walk-forward splits.

    The last final_test_size observations form the evaluation period.
    Each test block is evaluated once. Training and validation windows
    move forward by step_size after each fold.

    A gap of `embargo` observations separates:
      - training from validation
      - validation from testing

    Parameters are measured in rows, not calendar days.
    Slices follow normal Python conventions: start included, stop excluded.
    """
    positive_parameters = {
        "n_samples": n_samples,
        "train_size": train_size,
        "validation_size": validation_size,
        "test_size": test_size,
        "step_size": step_size,
        "final_test_size": final_test_size,
    }

    for name, value in positive_parameters.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    if not isinstance(embargo, int) or isinstance(embargo, bool) or embargo < 0:
        raise ValueError("embargo must be a non-negative integer.")

    # Disjoint test blocks ensure every test observation is scored once.
    if test_size != step_size:
        raise ValueError(
            "test_size must equal step_size so test blocks do not overlap "
            "or leave gaps."
        )

    if final_test_size % test_size != 0:
        raise ValueError(
            "final_test_size must be divisible by test_size."
        )

    first_test_start = n_samples - final_test_size

    required_before_test = (
        train_size + validation_size + 2 * embargo
    )

    if first_test_start < required_before_test:
        minimum_samples = final_test_size + required_before_test
        raise ValueError(
            f"Not enough observations for these settings. "
            f"Need at least {minimum_samples}, received {n_samples}."
        )

    splits = []

    for fold_number, test_start in enumerate(
        range(first_test_start, n_samples, step_size),
        start=1,
    ):
        test_stop = test_start + test_size

        validation_stop = test_start - embargo
        validation_start = validation_stop - validation_size

        train_stop = validation_start - embargo
        train_start = train_stop - train_size

        splits.append(
            WalkForwardSplit(
                fold=fold_number,
                train=slice(train_start, train_stop),
                validation=slice(validation_start, validation_stop),
                test=slice(test_start, test_stop),
            )
        )

    return splits
