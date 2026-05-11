from sk_autod import diagnose


def test_diagnose_returns_report():
    report = diagnose(
        [2.3, 1.8, 1.3, 0.9, 0.5],
        [2.4, 2.0, 2.1, 2.5, 3.0],
    )

    assert report.findings
