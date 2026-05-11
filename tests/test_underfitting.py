from sk_autod.core.preprocessing import Preprocessor
from sk_autod.detectors.underfitting import UnderfittingDetector


def test_underfitting_detector_fires_on_flat_high_curves():
    curves = Preprocessor().preprocess(
        [1.9, 1.8, 1.8, 1.7, 1.7],
        [2.0, 1.9, 1.9, 1.8, 1.8],
    )

    findings = UnderfittingDetector().detect(curves)

    assert len(findings) == 1
