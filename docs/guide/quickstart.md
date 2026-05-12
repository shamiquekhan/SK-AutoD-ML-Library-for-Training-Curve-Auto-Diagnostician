---
title: Quickstart
parent: Guide
nav_order: 2
---

# Quickstart

## Basic diagnosis

```python
from sk_autod import diagnose

train_loss = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val_loss   = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

report = diagnose(train_loss, val_loss)
print(report.summary())
```

## One-liner

```python
from sk_autod import quick_check
print(quick_check(train_loss, val_loss))
# → "[CRITICAL] Classic overfitting at epoch 4 (94% confidence)"
```

## JSON output

```python
import json
print(json.dumps(report.to_dict(), indent=2))
```

## In a training loop

```python
train_losses, val_losses = [], []

for epoch in range(100):
    # ... your training step ...
    train_losses.append(train_loss_this_epoch)
    val_losses.append(val_loss_this_epoch)

# Check periodically
if epoch % 10 == 0 and epoch >= 5:
    from sk_autod import quick_check
    print(quick_check(train_losses, val_losses))
```

## Custom detectors

```python
from sk_autod import diagnose
from sk_autod.detectors.base import BaseDetector
from sk_autod.core.report import Finding, Severity

class MyDetector(BaseDetector):
    name = "my_custom_check"

    def detect(self, report):
        # example: flag if final train loss > 1.0
        if report.train_loss[-1] > 1.0:
            return [Finding(
                pattern="high_final_loss",
                severity=Severity.MEDIUM,
                confidence=0.8,
                epoch=len(report.train_loss),
                description="Final training loss still above 1.0.",
                fix="Train longer or increase model capacity."
            )]
        return []

report = diagnose(train_loss, val_loss, detectors=[MyDetector()])
```

## Examples

Three patterns to try:

```python
from sk_autod import diagnose

# 1. Clean training — no issues expected
diagnose([2.3,1.9,1.4,0.9,0.5,0.3,0.15], [2.4,2.0,1.6,1.4,1.3,1.2,1.2]).summary()

# 2. Classic overfitting
diagnose([2.3,1.9,1.4,0.9,0.5,0.3,0.15], [2.4,2.0,1.8,1.9,2.3,2.8,3.4]).summary()

# 3. Oscillating loss (LR too high)
diagnose([2.3,1.8,1.6,1.9,1.5,1.7,1.4], [2.5,2.0,1.9,2.1,1.8,2.0,1.9]).summary()
```
