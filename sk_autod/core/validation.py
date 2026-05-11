from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def validate_curves(train_loss: Sequence[float], val_loss: Sequence[float]) -> None:
    if train_loss is None or val_loss is None:
        raise ValueError("train_loss and val_loss are required")

    if len(train_loss) == 0 or len(val_loss) == 0:
        raise ValueError("train_loss and val_loss must not be empty")

    if len(train_loss) != len(val_loss):
        raise ValueError("train_loss and val_loss must have the same length")

    if len(train_loss) < 5:
        raise ValueError("Need at least 5 epochs of data")

    if any(not isfinite(float(value)) for value in train_loss) or any(
        not isfinite(float(value)) for value in val_loss
    ):
        raise ValueError("Curves must contain only finite numeric values")
