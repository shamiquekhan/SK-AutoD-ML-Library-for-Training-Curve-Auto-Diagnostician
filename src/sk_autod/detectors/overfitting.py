from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class OverfittingDetector(BaseDetector):
    name = "Classic overfitting"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.train_smooth) < 5:
            return []

        train_window = curves.train_smooth[-3:]
        val_window = curves.val_smooth[-3:]
        if train_window[0] > train_window[-1] and val_window[0] < val_window[-1]:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.CRITICAL,
                    confidence=0.92,
                    message="Validation loss increased while training loss decreased.",
                    recommendation="Use dropout, regularization, or reduce model capacity.",
                    epoch=len(curves.train_smooth) - 1,
                )
            ]
        return []
