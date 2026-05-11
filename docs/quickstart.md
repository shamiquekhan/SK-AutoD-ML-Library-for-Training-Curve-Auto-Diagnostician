# Quickstart

```python
from sk_autod import diagnose

train_loss = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val_loss = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

report = diagnose(train_loss, val_loss)
print(report.summary())
```

## What to expect

- `diagnose()` returns a report object
- `summary()` prints a human-readable diagnosis
- `to_dict()` returns JSON-friendly output

## Notebook tip

For Jupyter workflows, call `diagnose()` after training and pass the collected `loss` and `val_loss` arrays.
