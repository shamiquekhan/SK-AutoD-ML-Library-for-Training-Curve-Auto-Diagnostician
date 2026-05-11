# API Reference

## `sk_autod.diagnose(train_loss, val_loss, detectors=None)`

Runs the full diagnostics pipeline and returns a `DiagnosisReport`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `train_loss` | `Sequence[float]` | Training loss per epoch (≥5 values) |
| `val_loss` | `Sequence[float]` | Validation loss per epoch (same length as train_loss) |
| `detectors` | `list[BaseDetector] \| None` | Custom detectors (default: all 10 built-in) |

### Returns

`DiagnosisReport` - Container with findings and output methods.

### Example

```python
from sk_autod import diagnose

train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

report = diagnose(train, val)
print(report.summary())
```

---

## `DiagnosisReport`

The main result object.

### Methods

| Method | Returns | Description |
|---|---|---|
| `summary()` | `str` | Human-readable text report |
| `to_dict()` | `dict[str, Any]` | JSON-serializable dictionary |
| `to_html()` | `str` | Standalone HTML report |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `findings` | `list[Finding]` | List of detected issues, sorted by severity |

---

## `Finding`

A single diagnostic issue.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `detector_name` | `str` | Name of the detector that found this |
| `severity` | `Severity` | CRITICAL, HIGH, MEDIUM, WARNING, or INFO |
| `confidence` | `float` | 0.0–1.0 confidence score |
| `message` | `str` | Human-readable problem description |
| `recommendation` | `str` | Suggested fix |
| `epoch` | `int \| None` | Epoch where issue was detected |
| `metadata` | `dict[str, Any]` | Detector-specific data |

---

## `Severity`

Enum for issue severity levels.

```python
from sk_autod import Severity

# Values (in order of importance)
Severity.CRITICAL  # Training is broken, stop immediately
Severity.HIGH      # Major issue, likely hurts performance
Severity.MEDIUM    # Noticeable issue, should address
Severity.WARNING   # Minor issue, good to know
Severity.INFO      # Informational, no action needed
```

---

## `quick_check(train_loss, val_loss)`

Returns a short one-line diagnosis for notebooks and terminals.

```python
from sk_autod import quick_check

quick_check(train, val)
# Output: "🔴 CRITICAL: Classic overfitting detected at epoch 4"
```

---

## `AutoDCallback`

Collects losses during training and prints live diagnostics once enough epochs have been observed.

```python
from sk_autod import AutoDCallback

callback = AutoDCallback()
model.fit(X, y, callbacks=[callback])
```

---

## `BaseDetector`

Abstract base class for creating custom detectors.

```python
from sk_autod.detectors import BaseDetector
from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.core.models import Finding, Severity

class MyDetector(BaseDetector):
    name = "My custom issue"
    
    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        if curves.train_smooth[-1] > curves.train_smooth[0]:
            return [Finding(
                detector_name=self.name,
                severity=Severity.WARNING,
                confidence=0.8,
                message="Training loss increased over time",
                recommendation="Check learning rate schedule",
            )]
        return []
```

---

## Built-in Detectors

| Detector | Severity | Description |
|---|---|---|
| `OverfittingDetector` | CRITICAL | Val rises while train falls |
| `ExplodingGradientDetector` | CRITICAL | Loss spikes exponentially |
| `LRHighDetector` | HIGH | Learning rate too high (oscillations) |
| `LRLowDetector` | MEDIUM | Learning rate too low (slow convergence) |
| `UnderfittingDetector` | HIGH | Loss stays high |
| `DyingReLUDetector` | HIGH | Loss plateaus early |
| `NoisyTrainingDetector` | WARNING | High variance in training |
| `DataLeakageDetector` | CRITICAL | Suspiciously low validation loss |
| `EarlyStopDetector` | WARNING | Missed early stopping opportunity |
| `LabelNoiseDetector` | MEDIUM | Noise floor in loss |
