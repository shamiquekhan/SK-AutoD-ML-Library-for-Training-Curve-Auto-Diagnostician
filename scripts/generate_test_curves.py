#!/usr/bin/env python
"""Generate synthetic training curves for testing and demos.

Run with: python scripts/generate_test_curves.py
"""

import json
from pathlib import Path


def generate_curve(length: int, pattern: str) -> tuple[list[float], list[float]]:
    """Generate synthetic training curves."""
    if pattern == "converging":
        train = [2.5 * (0.95**i) + 0.01 for i in range(length)]
        val = [2.6 * (0.94**i) + 0.02 for i in range(length)]
    elif pattern == "overfitting":
        train = [2.5 * (0.9**i) for i in range(length)]
        val = [2.6 * (0.9**i) for i in range(length)]
        pivot = int(length * 0.4)
        for i in range(pivot, length):
            val[i] = val[pivot - 1] + 0.05 * (i - pivot)
    elif pattern == "exploding":
        train = [2.5 - i * 0.1 for i in range(length)]
        val = [2.6 - i * 0.1 for i in range(length)]
        # Add explosion at 60%
        explosion_idx = int(length * 0.6)
        for i in range(explosion_idx, length):
            train[i] = train[i - 1] * 3
            val[i] = val[i - 1] * 3
    elif pattern == "plateau":
        train = [2.5 * (0.95**i) for i in range(length)]
        val = [2.6 * (0.95**i) for i in range(length)]
        # Plateau at 50%
        plateau_val = train[int(length * 0.5)]
        for i in range(int(length * 0.5), length):
            train[i] = plateau_val
            val[i] = plateau_val + 0.1
    elif pattern == "noisy":
        import random
        random.seed(42)
        train = [2.5 * (0.95**i) + random.uniform(-0.2, 0.2) for i in range(length)]
        val = [2.6 * (0.94**i) + random.uniform(-0.2, 0.2) for i in range(length)]
    else:
        train = [2.5 - i * 0.02 for i in range(length)]
        val = [2.6 - i * 0.015 for i in range(length)]
    
    return train, val


def main() -> None:
    output_dir = Path(__file__).parent.parent / "outputs" / "test_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patterns = ["converging", "overfitting", "exploding", "plateau", "noisy"]
    lengths = [20, 50, 100]
    
    for pattern in patterns:
        for length in lengths:
            train, val = generate_curve(length, pattern)
            filename = f"{pattern}_{length}epochs.json"
            filepath = output_dir / filename
            
            with open(filepath, "w") as f:
                json.dump({
                    "pattern": pattern,
                    "length": length,
                    "train_loss": train,
                    "val_loss": val,
                }, f, indent=2)
            
            print(f"Generated: {filename}")
    
    print(f"\nAll curves saved to: {output_dir}")


if __name__ == "__main__":
    main()
