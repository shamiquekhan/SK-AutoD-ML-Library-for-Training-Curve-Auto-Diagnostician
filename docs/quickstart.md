# Quickstart

## Basic Usage

```python
from sk_autod import diagnose

train_loss = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val_loss = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

report = diagnose(train_loss, val_loss)
print(report.summary())
```

Output:
```
SK-AutoD Diagnosis Report:
- [Severity.CRITICAL] Classic overfitting: Validation loss increased while training loss decreased.
- [Severity.HIGH] Dying ReLU proxy: Loss plateaus early and remains high.
```

## Output Formats

### Text Summary

```python
print(report.summary())
```

### JSON (for logging/APIs)

```python
import json
data = report.to_dict()
print(json.dumps(data, indent=2))
```

### HTML (for reports)

```python
html = report.to_html()
with open("diagnosis_report.html", "w") as f:
    f.write(html)
```

## Using Specific Detectors

```python
from sk_autod import diagnose
from sk_autod.detectors import OverfittingDetector, LRHighDetector

# Only run specific detectors
report = diagnose(train_loss, val_loss, detectors=[
    OverfittingDetector(),
    LRHighDetector(),
])
```

## Quick One-Liner Check

```python
from sk_autod import quick_check

quick_check(train_loss, val_loss)
# Output: "🔴 CRITICAL: Classic overfitting detected"
```

## Notebook Integration

For Jupyter workflows, call `diagnose()` after training:

```python
# After model.fit() or training loop
history = model.history
report = diagnose(history['loss'], history['val_loss'])
report.to_html()  # Renders nicely in notebooks
```

## What to Expect

- `diagnose()` returns a report object
- `summary()` prints a human-readable diagnosis
- `to_dict()` returns JSON-friendly output
- `to_html()` returns a standalone HTML page
- Findings are sorted by severity (CRITICAL → INFO)
