# Installation

## From PyPI (once published)

```bash
pip install sk-autod
```

## From Source (recommended for now)

```bash
git clone https://github.com/shamiquekhan/SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician.git
cd SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician
pip install -e ".[dev]"
```

## Verify the Install

```bash
python -c "from sk_autod import diagnose; print('SK-AutoD ready')"
```

## Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For running examples with numpy
pip install -e ".[examples]"
```
