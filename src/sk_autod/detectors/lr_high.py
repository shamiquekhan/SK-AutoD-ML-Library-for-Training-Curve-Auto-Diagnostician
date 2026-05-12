from __future__ import annotations

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class LRHighDetector(BaseDetector):
    name = "LR too high"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        deltas = [current - previous for previous, current in zip(curves.train_loss, curves.train_loss[1:])]
        if len(deltas) < 4:
            return []

        sign_flips = sum(1 for previous, current in zip(deltas, deltas[1:]) if previous == 0 or current == 0 or (previous > 0) != (current > 0))
        has_uptrend = curves.train_loss[-1] > curves.train_loss[0] * 0.5

        if sign_flips >= 2 and has_uptrend:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.HIGH,
                    confidence=0.86,
                    message="Loss oscillates without a stable downward trend.",
                    recommendation="Reduce the learning rate or add warmup.",
                    epoch=len(curves.train_smooth) - 1,
                    metadata={"sign_flips": int(sign_flips)},
                )
            ]
        return []
