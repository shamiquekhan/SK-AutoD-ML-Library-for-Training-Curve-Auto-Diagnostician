from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class DyingReLUDetector(BaseDetector):
    name = "Dying ReLU proxy"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.train_smooth) < 5:
            return []

        plateau_end = max(2, len(curves.train_smooth) // 3)
        plateau = curves.train_smooth[:plateau_end]
        variation = max(plateau) - min(plateau)
        final_loss = curves.train_smooth[-1]

        if variation < 0.15 and final_loss > 0.7:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.HIGH,
                    confidence=0.77,
                    message="Loss plateaus early and remains high.",
                    recommendation="Use leaky ReLU, improve initialization, or normalize inputs.",
                    epoch=plateau_end - 1,
                    metadata={"plateau_variation": variation},
                )
            ]
        return []
