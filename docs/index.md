---
title: SK-AutoD
layout: home
nav_order: 1
---

# SK-AutoD 🩺
{: .fs-9 }

Auto-diagnose your ML training curves in seconds.
{: .fs-6 .fw-300 }

[Get started](guide/quickstart/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/shamiquekhan/sk-autod){: .btn .fs-5 .mb-4 .mb-md-0 }

---

Stop manually eyeballing loss curves. SK-AutoD analyses your training data and instantly
detects 10+ common pathologies — overfitting, exploding gradients, learning rate issues, and more.

## The one-liner

```python
from sk_autod import diagnose

report = diagnose(
    train_loss=[2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15],
    val_loss  =[2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
)
report.summary()
```

**Output:**

```
═══════════════════════════════════════════════════════
  SK-AutoD Diagnosis Report
═══════════════════════════════════════════════════════

🔴 CRITICAL: Classic overfitting
   Detected at epoch 4 (94% confidence)
   Val loss rose while train loss fell for 3+ consecutive epochs.

   Fix: Add dropout (p=0.3–0.5), L2 regularisation, or reduce model capacity.

═══════════════════════════════════════════════════════
Summary: 1 critical issue found.
```

## What it detects

| Detector | Severity |
|---|---|
| Classic overfitting | 🔴 Critical |
| Exploding gradient | 🔴 Critical |
| LR too high | 🟠 High |
| Underfitting | 🟠 High |
| Dying ReLU proxy | 🟠 High |
| Data leakage proxy | 🟠 High |
| LR too low | 🟡 Medium |
| Noisy training | 🟡 Medium |
| Label noise floor | 🟡 Medium |
| Missed early stopping | 🔵 Warning |

## Install

```bash
pip install sk-autod
```

## Philosophy

- **Rule-based, not ML-based** — interpretable, debuggable, works offline
- **Zero config** — sensible defaults, works out of the box
- **Fast** — diagnoses 1000 curves in under 200 ms
- **Fail gracefully** — unknown patterns produce no false alarms
