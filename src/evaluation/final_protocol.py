"""Prospective evaluation boundaries; observed development data is never held out retroactively."""
import hashlib

import numpy as np
import pandas as pd

from src.evaluation.splits import generate_walk_forward_splits


def history_digest(returns, through):
    """Hash parsed history so append-only updates preserve the research vintage."""
    history = returns.loc[:pd.Timestamp(through)]
    return hashlib.sha256(history.to_csv(float_format="%.12g").encode()).hexdigest()


def final_evaluation_plan(raw_index, config):
    """Select first N post-freeze origins only once all N forward targets exist.

    Returns a blocked status without inventing future rows. A trailing window
    must never silently move the final evaluation to different dates.
    """
    if not isinstance(raw_index, pd.DatetimeIndex) or raw_index.hasnans:
        raise ValueError("Raw dates must be a non-missing DatetimeIndex.")
    if raw_index.has_duplicates or not raw_index.is_monotonic_increasing:
        raise ValueError("Raw dates must be unique and chronological.")
    ev, final = config["evaluation"], config["final_evaluation"]
    boundary = max(pd.Timestamp(final["freeze_date"]), pd.Timestamp(final["exposed_data_through"]))
    future_positions = np.flatnonzero(raw_index > boundary)
    required = final["observations"] + ev["horizon"]
    if len(future_positions) < required:
        return {"status": "awaiting_future_data", "boundary": str(boundary.date()),
                "available_post_freeze_observations": len(future_positions),
                "required_post_freeze_observations": required,
                "missing_observations": required - len(future_positions)}
    first = int(future_positions[0])
    stop = first + final["observations"]
    splits = generate_walk_forward_splits(
        stop, ev["train_size"], ev["validation_size"], ev["test_size"],
        ev["step_size"], final["observations"], ev["embargo"],
    )
    for split in splits:
        if split.train.stop - 1 + ev["horizon"] >= split.validation.start:
            raise ValueError("Training targets overlap validation.")
        if split.validation.stop - 1 + ev["horizon"] >= split.test.start:
            raise ValueError("Validation targets overlap testing.")
    return {"status": "ready", "first_position": first, "stop_position": stop,
            "raw_stop_position": stop + ev["horizon"], "splits": splits,
            "start_date": str(raw_index[first].date()),
            "end_date": str(raw_index[stop-1].date())}
