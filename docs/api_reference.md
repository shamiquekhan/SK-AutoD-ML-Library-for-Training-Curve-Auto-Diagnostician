# API Reference

## `sk_autod.diagnose(train_loss, val_loss)`

Runs the full diagnostics pipeline and returns a `DiagnosisReport`.

### Parameters

- `train_loss`: sequence of training loss values
- `val_loss`: sequence of validation loss values

### Returns

- `DiagnosisReport`

## `DiagnosisReport`

Methods:

- `summary()`
- `to_dict()`
- `to_html()`

## `quick_check(train_loss, val_loss)`

Returns a short one-line diagnosis for notebooks and terminals.

## `AutoDCallback`

Collects losses during training and prints live diagnostics once enough epochs have been observed.
