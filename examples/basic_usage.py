from sk_autod import diagnose


def main() -> None:
    train_loss = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
    val_loss = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

    report = diagnose(train_loss, val_loss)
    print(report.summary())


if __name__ == "__main__":
    main()
