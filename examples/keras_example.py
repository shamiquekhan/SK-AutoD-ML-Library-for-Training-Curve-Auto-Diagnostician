"""Example of using SK-AutoD after a Keras training loop.

Pass `history.history['loss']` and `history.history['val_loss']` to diagnose().
"""

from sk_autod import diagnose


def main() -> None:
    train_loss = [1.7, 1.3, 1.08, 0.95, 0.88, 0.84, 0.81]
    val_loss = [1.8, 1.35, 1.14, 1.10, 1.12, 1.18, 1.23]

    report = diagnose(train_loss, val_loss)
    print(report.summary())


if __name__ == "__main__":
    main()
