"""Example: Creating a custom detector for SK-AutoD.

This example shows how to subclass BaseDetector to create your own
diagnostic detector that integrates with the SK-AutoD pipeline.
"""

from sk_autod import diagnose
from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.detectors.base import BaseDetector


class PlateauDetector(BaseDetector):
    """Detects when loss plateaus for too long without improvement.

    This is a custom detector that flags when the training loss
    hasn't improved by at least `min_improvement` over `window_size` epochs.
    """

    name = "Training plateau"

    def __init__(self, window_size: int = 10, min_improvement: float = 0.01):
        self.window_size = window_size
        self.min_improvement = min_improvement

    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if len(curves.train_smooth) < self.window_size:
            return []

        # Check if training loss has plateaued
        window = curves.train_smooth[-self.window_size :]
        improvement = window[0] - window[-1]

        if improvement < self.min_improvement:
            return [
                Finding(
                    detector_name=self.name,
                    severity=Severity.MEDIUM,
                    confidence=0.75,
                    message=f"Training loss plateaued for {self.window_size} epochs "
                    f"(improvement: {improvement:.4f} < {self.min_improvement})",
                    recommendation="Consider increasing learning rate or using "
                    "learning rate scheduling to break out of the plateau.",
                    epoch=len(curves.train_smooth) - self.window_size,
                    metadata={"improvement": improvement, "window_size": self.window_size},
                )
            ]
        return []


def main() -> None:
    # Example: Training that plateaus
    train_loss = [
        2.5, 2.0, 1.7, 1.5, 1.4, 1.35, 1.33, 1.32, 1.31, 1.30,
        1.30, 1.29, 1.29, 1.28, 1.28,  # Plateau starts here
    ]
    val_loss = [
        2.6, 2.1, 1.8, 1.6, 1.5, 1.45, 1.43, 1.42, 1.41, 1.40,
        1.40, 1.39, 1.39, 1.38, 1.38,
    ]

    # Use custom detector alongside built-in detectors
    report = diagnose(train_loss, val_loss, detectors=[PlateauDetector()])

    print("Custom Detector Results:")
    print(report.summary())

    # Access detailed findings
    for finding in report.findings:
        print(f"\n{finding.detector_name}:")
        print(f"  Confidence: {finding.confidence:.0%}")
        print(f"  Recommendation: {finding.recommendation}")


if __name__ == "__main__":
    main()
