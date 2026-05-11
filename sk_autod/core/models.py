from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(slots=True)
class Finding:
    detector_name: str
    severity: Severity
    confidence: float
    message: str
    recommendation: str
    epoch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiagnosisReport:
    findings: list[Finding] = field(default_factory=list)

    def summary(self) -> str:
        if not self.findings:
            return "✅ No issues found."

        lines = ["SK-AutoD Diagnosis Report:"]
        for finding in self.findings:
            lines.append(
                f"- [{finding.severity}] {finding.detector_name}: {finding.message}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [
                {
                    "detector_name": finding.detector_name,
                    "severity": finding.severity.value,
                    "confidence": finding.confidence,
                    "message": finding.message,
                    "recommendation": finding.recommendation,
                    "epoch": finding.epoch,
                    "metadata": finding.metadata,
                }
                for finding in self.findings
            ]
        }
