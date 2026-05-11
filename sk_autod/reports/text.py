from __future__ import annotations

from sk_autod.core.models import DiagnosisReport


def render_text(report: DiagnosisReport) -> str:
    if not report.findings:
        return "✅ No issues found."

    lines = ["SK-AutoD Diagnosis Report"]
    for finding in report.findings:
        lines.append(f"[{finding.severity}] {finding.detector_name}")
        lines.append(f"  Confidence: {finding.confidence:.0%}")
        lines.append(f"  Problem: {finding.message}")
        lines.append(f"  Fix: {finding.recommendation}")
    return "\n".join(lines)
