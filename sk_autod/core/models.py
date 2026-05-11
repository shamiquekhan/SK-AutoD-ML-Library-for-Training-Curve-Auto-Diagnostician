from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
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

    def to_html(self) -> str:
        if not self.findings:
            return "<html><body><p>✅ No issues found.</p></body></html>"

        cards = []
        for finding in self.findings:
            cards.append(
                "<div class='finding'>"
                f"<h3>{escape(finding.detector_name)}</h3>"
                f"<p><strong>Severity:</strong> {escape(finding.severity.value)}</p>"
                f"<p><strong>Confidence:</strong> {finding.confidence:.0%}</p>"
                f"<p><strong>Problem:</strong> {escape(finding.message)}</p>"
                f"<p><strong>Fix:</strong> {escape(finding.recommendation)}</p>"
                "</div>"
            )

        return (
            "<html><head><meta charset='utf-8'>"
            "<title>SK-AutoD Diagnosis Report</title>"
            "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.5}"
            ".finding{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0}</style>"
            "</head><body><h1>SK-AutoD Diagnosis Report</h1>"
            + "".join(cards)
            + "</body></html>"
        )
