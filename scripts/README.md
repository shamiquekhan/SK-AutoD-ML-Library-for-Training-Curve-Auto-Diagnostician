# Scripts

Utility scripts for SK-AutoD development and maintenance.

## Available Scripts

### `generate_test_curves.py`

Generate synthetic training curves for testing and demos.

```bash
python scripts/generate_test_curves.py
```

Creates JSON files in `outputs/test_curves/` with patterns:
- `converging` - Smooth convergence
- `overfitting` - Train falls, val rises
- `exploding` - Loss explodes mid-training
- `plateau` - Loss plateaus early
- `noisy` - High variance training

## Adding New Scripts

Place utility scripts here following the naming convention:
- `generate_*.py` - Data generation scripts
- `validate_*.py` - Validation scripts
- `release_*.py` - Release automation
