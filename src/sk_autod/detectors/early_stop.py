from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class EarlyStopDetector(BaseDetector):
    name = "Missed early stopping"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.val_loss) < 5:
            return []

        best_epoch = min(range(len(curves.val_loss)), key=curves.val_loss.__getitem__)
        final_epoch = len(curves.val_loss) - 1
        tail_start = int(len(curves.val_loss) * 0.9)

        if best_epoch < tail_start and final_epoch > best_epoch and curves.val_loss[-1] > curves.val_loss[best_epoch] * 1.05:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.WARNING,
                    confidence=0.78,
                    message="Validation loss had an earlier minimum that was not used.",
                    recommendation="Enable EarlyStopping with restore_best_weights=True.",
                    epoch=best_epoch,
                    metadata={"best_epoch": best_epoch},
                )
            ]

        return []