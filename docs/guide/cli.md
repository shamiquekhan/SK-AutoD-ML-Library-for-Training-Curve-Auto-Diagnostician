---
title: CLI usage
parent: Guide
nav_order: 3
---

# CLI usage

SK-AutoD includes a command-line interface installed alongside the package.

## Basic diagnosis

```bash
sk-autod diagnose \
  --train-loss 2.3 1.9 1.4 0.9 0.5 0.3 0.15 \
  --val-loss   2.4 2.0 1.8 1.9 2.3 2.8 3.4
```

## From CSV files

```bash
sk-autod diagnose \
  --train-file train_losses.csv \
  --val-file   val_losses.csv
```

## Output formats

```bash
# Default: formatted text
sk-autod diagnose --train-loss 2.3 1.9 ... --output text

# JSON (for piping to other tools)
sk-autod diagnose --train-loss 2.3 1.9 ... --output json

# HTML report saved to file
sk-autod diagnose --train-loss 2.3 1.9 ... --output html --out report.html
```

## Pipe-friendly

```bash
echo "2.3 1.9 1.4 0.9" | sk-autod diagnose --train-loss -
```

## Version and help

```bash
sk-autod --version
sk-autod diagnose --help
```
