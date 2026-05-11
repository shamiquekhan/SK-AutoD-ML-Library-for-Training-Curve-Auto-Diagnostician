"""Example of using SK-AutoD after a PyTorch training loop.

This example keeps the focus on the diagnostics API and uses synthetic losses
so it can run without a full model training setup.
"""

from sk_autod import diagnose


def main() -> None:
    train_loss = [1.8, 1.5, 1.2, 1.0, 0.8, 0.7, 0.62]
    val_loss = [1.9, 1.6, 1.3, 1.15, 1.12, 1.18, 1.34]

    report = diagnose(train_loss, val_loss)
    print(report.summary())


if __name__ == "__main__":
    main()
