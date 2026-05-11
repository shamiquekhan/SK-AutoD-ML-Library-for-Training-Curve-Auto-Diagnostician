"""Compatibility alias for noisy_training detector."""
from sk_autod.detectors.noisy_training import NoisyTrainingDetector

__all__ = ["NoisyDetector", "NoisyTrainingDetector"]


# Alias for compatibility with architecture guide references
NoisyDetector = NoisyTrainingDetector
