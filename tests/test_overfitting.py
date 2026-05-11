from sk_autod.core.preprocessing import Preprocessor
from sk_autod.detectors.overfitting import OverfittingDetector


def test_overfitting_detector_fires_on_diverging_curves():
    curves = Preprocessor().preprocess(
        [2.3, 1.9, 1.4, 0.9, 0.5],
        [2.4, 2.0, 2.1, 2.5, 3.0],
    )

    findings = OverfittingDetector().detect(curves)

    assert len(findings) == 1
