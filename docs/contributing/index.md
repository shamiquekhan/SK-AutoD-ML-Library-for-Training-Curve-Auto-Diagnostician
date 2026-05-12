---
title: Contributing
nav_order: 5
---

# Contributing

Contributions are welcome — bug reports, new detectors, integrations, and docs improvements.

## Getting started

```bash
git clone https://github.com/shamiquekhan/SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician.git
cd sk-autod
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Code style

```bash
make lint     # runs ruff + black
make format   # auto-fix formatting
```

## Adding a new detector

1. Open an issue first to discuss the pattern you want to detect
2. Create `sk_autod/detectors/your_detector.py` subclassing `BaseDetector` 
3. Write unit tests in `tests/test_your_detector.py` with fixture curves that definitely trigger it and curves that definitely should not
4. Register it in `sk_autod/detectors/__init__.py` 
5. Document it in `docs/detectors/index.md` 
6. Open a pull request

## Areas we'd love help with

- Threshold tuning on real Kaggle/HuggingFace training logs
- Framework integrations (Weights & Biases, MLflow, Kubeflow)
- Jupyter magic command (`%autod train_loss val_loss`)
- VS Code extension
