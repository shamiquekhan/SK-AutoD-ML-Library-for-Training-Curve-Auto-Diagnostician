from __future__ import annotations

from statistics import mean

from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class LabelNoiseDetector(BaseDetector):
    name = "Label noise floor"

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.train_smooth) < 6:
            return []

        tail_start = max(1, int(len(curves.train_smooth) * 0.6))
        train_tail = curves.train_smooth[tail_start:]
        val_tail = curves.val_smooth[tail_start:]
        tail_slope = train_tail[0] - train_tail[-1]
        tail_gap = abs(mean(train_tail) - mean(val_tail))

        if tail_slope < 0.05 and train_tail[-1] > 0.7 and tail_gap < 0.2:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.MEDIUM,
                    confidence=0.72,
                    message="Loss has flattened above a suspiciously high floor.",
                    recommendation="Audit labels, class balance, and consider label smoothing.",
                    epoch=len(curves.train_smooth) - 1,
                    metadata={"tail_gap": tail_gap},
                )
            ]
        return []
