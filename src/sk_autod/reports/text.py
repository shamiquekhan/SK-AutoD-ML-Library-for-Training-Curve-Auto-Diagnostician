from __future__ import annotations

from sk_autod.core.models import DiagnosisReport, Severity


SEVERITY_EMOJIS = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.WARNING: "🟡",
    Severity.INFO: "🔵",
}


def _format_boxed_title(title: str) -> str:
    width = max(len(title), 55)
    border = "═" * width
    centered_title = title.center(width)
    return f"{border}\n{centered_title}\n{border}"


def _format_confidence(confidence: float) -> str:
    return f"{confidence:.0%}" if confidence <= 1 else f"{confidence:.0f}%"


def render_text(report: DiagnosisReport) -> str:
    if not report.findings:
        return "✅ No issues found."

    lines = [_format_boxed_title("SK-AutoD Diagnosis Report")]
    for finding in report.findings:
        lines.append("")
        emoji = SEVERITY_EMOJIS.get(finding.severity, "")
        header = f"{emoji} {finding.severity.value}: {finding.detector_name}"
        lines.append(header)
        lines.append(f"   Detected at epoch {finding.epoch if finding.epoch is not None else 'n/a'} ({_format_confidence(finding.confidence)} confidence)")
        lines.append(f"   {finding.message}")
        lines.append("")
        lines.append(f"   Fix: {finding.recommendation}")

    lines.append("")
    lines.append(_format_boxed_title(f"Summary: {len(report.findings)} issue{'s' if len(report.findings) != 1 else ''} found."))
    return "\n".join(lines)

