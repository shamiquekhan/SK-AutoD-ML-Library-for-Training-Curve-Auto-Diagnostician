from __future__ import annotations

from statistics import mean

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class DataLeakageDetector(BaseDetector):
    name = "Data leakage"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        warmup = max(1, len(curves.train_loss) // 4)
        train_tail = curves.train_loss[warmup:]
        val_tail = curves.val_loss[warmup:]
        if len(train_tail) < 2:
            return []

        gap = mean(train_tail) - mean(val_tail)
        consistent = all(train > val for train, val in zip(train_tail, val_tail))
        if gap > 0.1 and consistent:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.HIGH,
                    confidence=0.83,
                    message="Validation loss is consistently lower than training loss.",
                    recommendation="Check the train/validation split and preprocessing for leakage.",
                    epoch=len(curves.train_loss) - 1,
                    metadata={"mean_gap": gap},
                )
            ]
        return []
