from __future__ import annotations

from sk_autod.core.models import DiagnosisReport, Finding
from sk_autod.core.preprocessing import Preprocessor
from sk_autod.core.validation import validate_curves
from sk_autod.detectors.lr_high import LRHighDetector
from sk_autod.detectors.overfitting import OverfittingDetector
from sk_autod.detectors.underfitting import UnderfittingDetector
from sk_autod.detectors.early_stop import EarlyStopDetector


def diagnose(train_loss, val_loss, detectors=None) -> DiagnosisReport:
    validate_curves(train_loss, val_loss)

    curves = Preprocessor().preprocess(train_loss, val_loss)
    active_detectors = detectors or [
        OverfittingDetector(),
        LRHighDetector(),
        EarlyStopDetector(),
        UnderfittingDetector(),
    ]

    findings: list[Finding] = []
    for detector in active_detectors:
        findings.extend(detector.detect(curves))

    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WARNING": 3, "INFO": 4}
    findings.sort(key=lambda item: (severity_rank[item.severity.value], -item.confidence))
    return DiagnosisReport(findings=findings)
