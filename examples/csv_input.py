"""Example: Loading training curves from CSV files.

This example demonstrates how to diagnose training curves stored
in CSV files, a common workflow when analyzing exported training logs.
"""

import csv
from pathlib import Path

from sk_autod import diagnose


def load_curve_from_csv(filepath: str, column: str = "loss") -> list[float]:
    """Load a loss curve from a CSV file.

    Args:
        filepath: Path to the CSV file
        column: Name of the column containing loss values

    Returns:
        List of loss values as floats
    """
    values: list[float] = []
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(float(row[column]))
    return values


def main() -> None:
    # Create a sample CSV file for demonstration
    sample_csv = Path(__file__).parent / "sample_training_log.csv"

    # Generate sample data
    train_losses = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
    val_losses = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

    # Write sample CSV
    with open(sample_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (train, val) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i, train, val])

    print(f"Created sample CSV: {sample_csv}")

    # Load curves from CSV
    train_loss = load_curve_from_csv(str(sample_csv), column="train_loss")
    val_loss = load_curve_from_csv(str(sample_csv), column="val_loss")

    print(f"Loaded {len(train_loss)} epochs from CSV")

    # Diagnose
    report = diagnose(train_loss, val_loss)
    print("\nDiagnosis Report:")
    print(report.summary())

    # Export results to JSON for downstream processing
    result = report.to_dict()
    print(f"\nFound {len(result['findings'])} issues")

    # Clean up sample file
    sample_csv.unlink()
    print(f"\nCleaned up sample CSV")


if __name__ == "__main__":
    main()
