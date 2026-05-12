---
title: Detectors
nav_order: 3
has_children: false
---

# Detectors

SK-AutoD includes 10 built-in detectors covering the most common ML training pathologies.
Each detector returns zero or one `Finding` with a severity, confidence score, root cause description, and fix recommendation.

---

## Classic overfitting

**Severity:** 🔴 Critical

**Signal:** Val loss rises while train loss keeps falling.

**Logic:** Finds the first epoch where `val_loss` delta turns positive for 3+ consecutive epochs while `train_loss` delta remains negative. Confidence scales with the magnitude of the divergence gap.

**Fix:** Add dropout (p=0.3–0.5), L2 regularisation, or reduce model capacity.

---

## Exploding gradient

**Severity:** 🔴 Critical

**Signal:** Loss spikes >300% in a single epoch.

**Logic:** Checks each epoch for `loss[i] > loss[i-1] × 3`. Confirms it's a single-point spike rather than a gradual rise.

**Fix:** Gradient clipping (`max_norm=1.0`), reduce learning rate.

---

## LR too high

**Severity:** 🟠 High

**Signal:** Loss oscillates without a clear downward trend.

**Logic:** Rolling variance of `train_loss` over a 5-epoch window exceeds 2× baseline, and loss is not trending down. Also detects the overshoot-then-partial-recovery pattern.

**Fix:** Reduce LR by 5–10×. Add a warmup schedule.

---

## LR too low

**Severity:** 🟡 Medium

**Signal:** Loss decreases extremely slowly across all epochs.

**Logic:** Overall slope of `train_loss` is shallower than −0.001/epoch across the first 50% of training. Also checks whether accuracy is climbing at all.

**Fix:** Increase LR. Use a warmup-then-cosine-decay schedule.

---

## Underfitting

**Severity:** 🟠 High

**Signal:** Both train and val loss plateau at high values with no divergence.

**Logic:** Final `train_loss` above threshold (default 0.5 for classification), slope in last 20% of epochs < 0.001, no divergence between train/val.

**Fix:** Increase model capacity, train longer, lower LR.

---

## Dying ReLU proxy

**Severity:** 🟠 High

**Signal:** Loss flatlines very early at a high value — not due to convergence.

**Logic:** Loss plateau before epoch 10 (or before 20% of total epochs), final loss not near 0, no oscillation ruling out LR issues.

**Fix:** He initialisation, leaky ReLU or ELU, batch normalisation.

---

## Noisy training

**Severity:** 🟡 Medium

**Signal:** Jagged loss curve with frequent up-down flips throughout training.

**Logic:** Counts sign changes in the loss delta array. If >40% of epochs flip direction, the training is flagged as unstable.

**Fix:** Increase batch size, add gradient clipping, reduce LR.

---

## Data leakage proxy

**Severity:** 🟠 High

**Signal:** Val loss is consistently lower than train loss (beyond the warm-up period).

**Logic:** Mean of `(train_loss − val_loss)` after warm-up epochs > 0.1 consistently.

**Fix:** Audit train/val split. Check for preprocessing applied to both splits before splitting.

---

## Missed early stopping

**Severity:** 🔵 Warning

**Signal:** Val loss had a clear minimum earlier in training that was not used as the checkpoint.

**Logic:** Finds global minimum of `val_loss`. If not in the last 10% of training, and final val loss > min × 1.05, the detector fires.

**Fix:** Use an `EarlyStopping` callback (patience=3–5). Restore best weights.

---

## Label noise floor

**Severity:** 🟡 Medium

**Signal:** Training loss never drops below a suspiciously high floor despite sufficient training.

**Logic:** Loss flattens above the expected floor (0.1 for clean multi-class classification) when training is long enough to have converged.

**Fix:** Audit labels, use label smoothing (ε=0.1), investigate class imbalance.

---

## Writing a custom detector

All built-in detectors subclass `BaseDetector`:

```python
from sk_autod.detectors.base import BaseDetector
from sk_autod.core.report import Finding, Severity

class SlowStartDetector(BaseDetector):
    name = "slow_start"

    def detect(self, report):
        # Check if loss barely moved in first 3 epochs
        if len(report.train_loss) < 3:
            return []

        initial_drop = report.train_loss[0] - report.train_loss[2]
        if initial_drop < 0.05:
            return [Finding(
                pattern="slow_start",
                severity=Severity.MEDIUM,
                confidence=0.75,
                epoch=1,
                description="Loss barely changed in first 3 epochs.",
                fix="Check your data loader, loss function, and initial LR."
            )]
        return []
```

Then pass it to `diagnose()`:

```python
report = diagnose(train_loss, val_loss, detectors=[SlowStartDetector()])
```
