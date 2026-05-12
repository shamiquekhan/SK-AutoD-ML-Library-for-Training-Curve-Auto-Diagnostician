from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class UnderfittingDetector(BaseDetector):
    name = "Underfitting"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        improvement = curves.train_smooth[0] - curves.train_smooth[-1]
        final_train = curves.train_smooth[-1]
        final_val = curves.val_smooth[-1]

        if improvement < 0.5 and final_train > 0.8 and final_val > 0.8:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.HIGH,
                    confidence=0.81,
                    message="Both train and validation loss stay high with little improvement.",
                    recommendation="Increase model capacity or train longer with better features.",
                    epoch=len(curves.train_smooth) - 1,
                )
            ]
        return []
