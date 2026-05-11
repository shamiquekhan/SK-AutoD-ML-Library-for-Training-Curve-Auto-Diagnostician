from sk_autod.core.preprocessing import Preprocessor
from sk_autod.detectors.lr_high import LRHighDetector


def test_lr_high_detector_fires_on_oscillation():
    curves = Preprocessor().preprocess(
        [2.3, 1.7, 1.9, 1.6, 1.8, 1.5],
        [2.4, 1.8, 2.0, 1.7, 1.9, 1.6],
    )

    findings = LRHighDetector().detect(curves)

    assert len(findings) == 1
