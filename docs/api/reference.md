---
title: API reference
nav_order: 4
---

# API reference

## `diagnose()` 

The primary entry point.

```python
from sk_autod import diagnose

report = diagnose(
    train_loss,          # list[float] — required
    val_loss=None,       # list[float] | None
    train_acc=None,      # list[float] | None
    val_acc=None,        # list[float] | None
    smooth=True,         # apply EMA smoothing before detection
    detectors=None,      # list[BaseDetector] | None — override default set
)
```

Returns a `DiagnosisReport`.

---

## `quick_check()` 

Returns a single summary string of the top-severity finding.

```python
from sk_autod import quick_check

msg = quick_check(train_loss, val_loss)
# → "[CRITICAL] Classic overfitting at epoch 4 (94% confidence)"
```

---

## `DiagnosisReport` 

The object returned by `diagnose()`.

| Method / attribute | Description |
|---|---|
| `.findings` | `list[Finding]` sorted by severity |
| `.summary()` | Prints formatted diagnosis to stdout |
| `.to_dict()` | Returns JSON-serialisable dict |
| `.to_html()` | Returns standalone HTML string (v0.2+) |
| `.train_loss` | The input train loss array |
| `.val_loss` | The input val loss array (or `None`) |
| `.epochs` | Number of epochs |

---

## `Finding` 

Each entry in `report.findings`.

| Field | Type | Description |
|---|---|---|
| `pattern` | `str` | Internal pattern ID e.g. `"classic_overfitting"` |
| `severity` | `Severity` | `CRITICAL`, `HIGH`, `MEDIUM`, `WARNING`, `INFO` |
| `confidence` | `float` | 0.0–1.0 |
| `epoch` | `int \| None` | Epoch where issue was detected |
| `description` | `str` | Human-readable explanation |
| `fix` | `str` | Concrete fix recommendation |

---

## `AutoDCallback` (v0.3+)

In-training callback compatible with PyTorch and Keras training loops.

```python
from sk_autod import AutoDCallback

cb = AutoDCallback(
    min_epochs=10,      # don't run before this epoch
    print_live=True,    # print warnings in real time
)

for epoch in range(100):
    # ... training step ...
    cb.on_epoch_end(epoch, train_losses, val_losses)
```

---

## `BaseDetector` 

Abstract class for custom detectors.

```python
from sk_autod.detectors.base import BaseDetector
from sk_autod.core.report import Finding

class MyDetector(BaseDetector):
    name = "my_detector"   # used in Finding.pattern

    def detect(self, report: DiagnosisReport) -> list[Finding]:
        ...
        return []   # or [Finding(...)]
```
