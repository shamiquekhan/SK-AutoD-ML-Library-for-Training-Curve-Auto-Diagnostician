from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sk_autod.utils.stats import diff, ema, mean, std


@dataclass(slots=True)
class PreprocessedCurves:
    train_loss: list[float]
    val_loss: list[float]
    train_smooth: list[float]
    val_smooth: list[float]
    train_delta: list[float]
    val_delta: list[float]
    gap: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class Preprocessor:
    def __init__(self, ema_alpha: float = 0.3) -> None:
        self.ema_alpha = ema_alpha

    def preprocess(self, train_loss, val_loss) -> PreprocessedCurves:
        train = [float(value) for value in train_loss]
        val = [float(value) for value in val_loss]
        train_smooth = ema(train, self.ema_alpha)
        val_smooth = ema(val, self.ema_alpha)

        return PreprocessedCurves(
            train_loss=train,
            val_loss=val,
            train_smooth=train_smooth,
            val_smooth=val_smooth,
            train_delta=diff(train_smooth),
            val_delta=diff(val_smooth),
            gap=[val_item - train_item for train_item, val_item in zip(train_smooth, val_smooth)],
            metadata={
                "train_mean": mean(train),
                "val_mean": mean(val),
                "train_std": std(train),
                "val_std": std(val),
            },
        )
