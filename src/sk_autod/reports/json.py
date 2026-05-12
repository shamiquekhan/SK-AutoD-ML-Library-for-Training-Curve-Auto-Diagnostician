from __future__ import annotations

import json

from sk_autod.core.models import DiagnosisReport


def render_json(report: DiagnosisReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
