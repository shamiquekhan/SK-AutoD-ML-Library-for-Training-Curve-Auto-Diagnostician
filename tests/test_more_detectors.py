from sk_autod import diagnose, render_json


def test_json_rendering_contains_findings_array():
    report = diagnose(
        [2.3, 2.0, 1.8, 1.7, 5.8],
        [2.4, 2.1, 1.9, 1.8, 5.5],
    )

    output = render_json(report)

    assert '"findings"' in output


def test_exploding_gradient_is_detected():
    report = diagnose(
        [2.3, 2.1, 2.0, 6.5, 6.3],
        [2.4, 2.2, 2.1, 6.8, 6.7],
    )

    assert any(f.detector_name == "Exploding gradient" for f in report.findings)
