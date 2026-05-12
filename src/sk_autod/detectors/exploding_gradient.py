from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class ExplodingGradientDetector(BaseDetector):
    name = "Exploding gradient"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        for index in range(1, len(curves.train_loss)):
            previous = curves.train_loss[index - 1]
            current = curves.train_loss[index]
            if previous > 0 and current >= previous * 3:
                return [
                    Finding(
                        detector_name=self.name,
                        severity=Severity.CRITICAL,
                        confidence=0.95,
                        message="Training loss spiked sharply in a single epoch.",
                        recommendation="Use gradient clipping and reduce the learning rate.",
                        epoch=index,
                        metadata={"spike_ratio": current / previous},
                    )
                ]
        return []
