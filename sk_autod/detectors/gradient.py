"""Compatibility alias for exploding_gradient detector."""
from sk_autod.detectors.exploding_gradient import ExplodingGradientDetector

__all__ = ["GradientExplosionDetector", "ExplodingGradientDetector"]


# Alias for compatibility with architecture guide references
GradientExplosionDetector = ExplodingGradientDetector
