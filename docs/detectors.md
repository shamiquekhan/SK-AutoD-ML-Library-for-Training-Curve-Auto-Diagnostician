# Detectors

SK-AutoD uses rule-based detectors to identify common training pathologies.

## Current detectors

- Classic overfitting
- Exploding gradient
- LR too high
- LR too low
- Underfitting
- Dying ReLU proxy
- Noisy training
- Data leakage proxy
- Missed early stopping
- Label noise floor

## Design notes

- Each detector is stateless
- Each detector returns zero or more findings
- Findings include severity, confidence, and a fix recommendation
- The runner deduplicates overlapping findings before formatting the report

## Extending the system

Create a new detector by subclassing `BaseDetector` and implementing `detect()`.
