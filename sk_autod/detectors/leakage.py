"""Compatibility alias for data_leakage detector."""
from sk_autod.detectors.data_leakage import DataLeakageDetector

__all__ = ["LeakageDetector", "DataLeakageDetector"]


# Alias for compatibility with architecture guide references
LeakageDetector = DataLeakageDetector
