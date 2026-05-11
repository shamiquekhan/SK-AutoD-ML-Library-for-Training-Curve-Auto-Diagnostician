from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class NoisyTrainingDetector(BaseDetector):
    name = "Noisy training"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        deltas = curves.train_delta
        if len(deltas) < 4:
            return []

        sign_changes = 0
        for previous, current in zip(deltas, deltas[1:]):
            if previous == 0 or current == 0:
                continue
            if (previous > 0) != (current > 0):
                sign_changes += 1

        ratio = sign_changes / max(1, len(deltas) - 1)
        if ratio >= 0.4:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.MEDIUM,
                    confidence=0.74,
                    message="Loss flips direction frequently, suggesting unstable training.",
                    recommendation="Increase batch size, reduce the learning rate, or clip gradients.",
                    epoch=len(curves.train_smooth) - 1,
                    metadata={"sign_change_ratio": ratio},
                )
            ]
        return []
