# Detectors

SK-AutoD uses rule-based detectors to identify common training pathologies.

## Current Detectors

### 🔴 CRITICAL

| Detector | Trigger Condition | Recommendation |
|---|---|---|
| **Classic overfitting** | Val rises while train falls for 3+ epochs | Add dropout (p=0.3–0.5), L2 regularization, or reduce model capacity |
| **Exploding gradient** | Loss spikes exponentially | Reduce learning rate, add gradient clipping |
| **Data leakage proxy** | Val loss suspiciously lower than train | Check for data leakage, verify train/val split |

### 🟠 HIGH

| Detector | Trigger Condition | Recommendation |
|---|---|---|
| **LR too high** | Oscillating loss pattern | Reduce LR by 5–10×, add warmup schedule |
| **Underfitting** | Loss stays high, minimal improvement | Increase model capacity, train longer, reduce regularization |
| **Dying ReLU proxy** | Loss plateaus early and stays flat | Use LeakyReLU, check initialization, normalize inputs |

### 🟡 MEDIUM/WARNING

| Detector | Trigger Condition | Recommendation |
|---|---|---|
| **LR too low** | Very slow convergence | Increase LR by 2–5× |
| **Noisy training** | High variance between epochs | Increase batch size, add smoothing, check data quality |
| **Missed early stopping** | Earlier val minimum not used | Enable EarlyStopping with restore_best_weights=True |
| **Label noise floor** | Loss doesn't improve beyond threshold | Check for label errors, clean dataset |

## Design Principles

- **Stateless**: No internal state (all data passed in)
- **Pure functions**: Same input → same output
- **Isolated**: No side effects, no external dependencies
- **Fast**: <1ms per detector on typical 100-epoch curve

## Creating Custom Detectors

```python
from sk_autod.detectors import BaseDetector
from sk_autod.core.models import Finding, Severity
from sk_autod.core.preprocessing import PreprocessedCurves

class MyDetector(BaseDetector):
    name = "My custom issue"
    
    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        # Access preprocessed data
        train_smooth = curves.train_smooth
        val_smooth = curves.val_smooth
        train_delta = curves.train_delta
        
        # Your detection logic here
        if some_condition:
            return [Finding(
                detector_name=self.name,
                severity=Severity.HIGH,
                confidence=0.85,
                message="Description of the issue",
                recommendation="How to fix it",
                epoch=affected_epoch,
            )]
        
        return []
```

### Available Preprocessed Data

| Field | Type | Description |
|---|---|---|
| `train_loss` | `list[float]` | Original training loss |
| `val_loss` | `list[float]` | Original validation loss |
| `train_smooth` | `list[float]` | EMA-smoothed training loss |
| `val_smooth` | `list[float]` | EMA-smoothed validation loss |
| `train_delta` | `list[float]` | Epoch-to-epoch changes in train |
| `val_delta` | `list[float]` | Epoch-to-epoch changes in val |
| `gap` | `list[float]` | val_smooth - train_smooth |
| `metadata` | `dict` | Statistics (mean, std, etc.) |
