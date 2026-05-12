---
title: Installation
parent: Guide
nav_order: 1
---

# Installation

## From PyPI (recommended)

```bash
pip install sk-autod
```

Requires Python 3.10+.

## From source (development)

```bash
git clone https://github.com/shamiquekhan/SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician.git
cd sk-autod
pip install -e ".[dev]"
```

## Optional dependencies

Visualization support (for HTML reports in v0.2+):

```bash
pip install sk-autod[viz]
```

This adds `matplotlib` and `plotly`.

## Verifying the install

```bash
python -c "from sk_autod import diagnose; print('SK-AutoD ready')"
```

Or via the CLI:

```bash
sk-autod --version
```
