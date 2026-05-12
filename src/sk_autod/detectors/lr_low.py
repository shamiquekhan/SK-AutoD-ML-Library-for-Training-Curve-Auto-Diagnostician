from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class LRLowDetector(BaseDetector):
    name = "LR too low"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.train_smooth) < 5:
            return []

        overall_improvement = curves.train_smooth[0] - curves.train_smooth[-1]
        per_epoch = overall_improvement / max(1, len(curves.train_smooth) - 1)

        if overall_improvement < 0.4 and curves.train_smooth[-1] > 0.5 and per_epoch < 0.001:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.MEDIUM,
                    confidence=0.76,
                    message="Loss is decreasing too slowly to be efficient.",
                    recommendation="Increase the learning rate or use a warmup schedule.",
                    epoch=len(curves.train_smooth) - 1,
                    metadata={"per_epoch_improvement": per_epoch},
                )
            ]
        return []
