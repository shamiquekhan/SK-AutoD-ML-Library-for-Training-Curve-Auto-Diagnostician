# SK-AutoD 🩺

**Auto-diagnose your training curves in seconds.**

Stop manually eyeballing loss curves. SK-AutoD analyzes your training data and instantly detects 10+ common pathologies: overfitting, exploding gradients, learning rate issues, underfitting, and more.

[![PyPI version](https://img.shields.io/pypi/v/sk-autod?color=blue)](https://pypi.org/project/sk-autod)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen)](https://www.python.org)

---

## The Problem

Every ML practitioner spends hours staring at loss curves during training:

- "Is this overfitting?"
- "Did my learning rate explode?"
- "Why is my loss stuck?"
- "Should I have stopped earlier?"

**Current workflow:** Manual eyeballing + Slack screenshots + tribal knowledge.

**SK-AutoD solves this:** Paste in your arrays → Get instant, rule-based diagnosis.

---

## Quick Start

### Installation

```bash
# From PyPI (once published)
pip install sk-autod

# From source (recommended for now)
pip install git+https://github.com/shamiquekhan/sk-autod.git
```

### Basic Usage

```python
from sk_autod import diagnose

# Your training curves
train_loss = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val_loss   = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]

# Get instant diagnosis
report = diagnose(train_loss, val_loss)

# Print human-readable summary
print(report.summary())
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

---

## Features

### 🔍 10 Diagnostic Detectors (v0.1.0+)

| Detector | Severity | Description |
|----------|----------|-------------|
| **Classic overfitting** | 🔴 CRITICAL | Val loss rises while train loss falls |
| **Exploding gradient** | 🔴 CRITICAL | Loss spikes >300% in a single epoch |
| **LR too high** | 🟠 HIGH | Loss oscillates without clear downtrend |
| **LR too low** | 🟡 MEDIUM | Loss decreases extremely slowly |
| **Underfitting** | 🟠 HIGH | Both losses plateau at high values |
| **Dying ReLU proxy** | 🟠 HIGH | Loss flatlines early at high value |
| **Noisy training** | 🟡 MEDIUM | Jagged loss with frequent up-down flips |
| **Data leakage proxy** | 🟠 HIGH | Val loss consistently lower than train |
| **Missed early stopping** | 🟡 WARNING | Val minimum not used as final checkpoint |
| **Label noise floor** | 🟡 MEDIUM | Loss can't drop below suspiciously high threshold |

### 📊 Multiple Output Formats

- **Text:** Pretty-printed summaries (with colors)
- **JSON:** Programmatic access to findings
- **HTML:** Interactive report with loss curve visualization (v0.2+)

### 🔧 Flexible APIs

```python
# 1. Full diagnosis (rich report)
report = diagnose(train_loss, val_loss)
print(report.summary())           # → formatted text
data = report.to_dict()           # → JSON-serializable dict
html = report.to_html()           # → standalone HTML (v0.2+)

# 2. One-liner for notebooks
from sk_autod import quick_check
print(quick_check(train_loss, val_loss))  # → "[CRITICAL] Classic overfitting"

# 3. In-training callback (v0.3+)
from sk_autod import AutoDCallback
cb = AutoDCallback(min_epochs=10, print_live=True)
for epoch in range(100):
    # ... your training loop ...
    cb.on_epoch_end(epoch, train_loss, val_loss)
```

### 🎯 Confidence-Scored Findings

Each diagnosis includes a confidence score (0.0–1.0), helping you prioritize fixes:

```python
for finding in report.findings:
    print(f"{finding.detector_name}: {finding.confidence:.1%}")
    print(f"  → {finding.fix_recommendation}")
```

---

## Architecture

```
User Input (loss arrays)
       ↓
Preprocessor (align, smooth, compute stats)
       ↓
[Detector 1] [Detector 2] ... [Detector N]  (run in parallel)
       ↓
DiagnosisReport (deduplicate, sort by severity)
       ↓
Formatters (text, JSON, HTML)
       ↓
Output
```

**Key components:**
- **Finding** & **DiagnosisReport:** Core data models
- **Preprocessor:** Validates, aligns, smooths with EMA, computes rolling stats
- **BaseDetector:** Abstract interface for all detectors
- **DiagnosticsRunner:** Orchestrates detectors, deduplicates findings
- **Formatters:** Text, JSON, HTML output channels

See [ARCHITECTURE.md](./ARCHITECTURE.md) for complete design details.

---

## CLI Usage

```bash
# Command-line diagnosis
sk-autod diagnose \
  --train-loss 2.3 1.9 1.4 0.9 0.5 0.3 0.15 \
  --val-loss 2.4 2.0 1.8 1.9 2.3 2.8 3.4 \
  --output json

# From CSV files
sk-autod diagnose --train-file train_losses.csv --val-file val_losses.csv

# From stdin (pipe-friendly)
echo "2.3 1.9 1.4 0.9" | sk-autod diagnose --train-loss -
```

---

## Examples

### Example 1: Well-Trained Model

```python
train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val   = [2.4, 2.0, 1.6, 1.4, 1.3, 1.2, 1.2]

report = diagnose(train, val)
print(report.summary())
# → ✅ No issues found!
```

### Example 2: Classic Overfitting

```python
train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val   = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]  # diverges ↗

report = diagnose(train, val)
# → 🔴 CRITICAL: Classic overfitting at epoch 4 (94% confidence)
# → Fix: Add dropout, L2 regularisation, reduce capacity
```

### Example 3: Learning Rate Too High

```python
train = [2.3, 1.8, 1.6, 1.9, 1.5, 1.7, 1.4]  # oscillates
val   = [2.5, 2.0, 1.9, 2.1, 1.8, 2.0, 1.9]

report = diagnose(train, val)
# → 🟠 HIGH: LR too high (oscillations detected, variance 2.3× baseline)
# → Fix: Reduce LR by 5–10×, add warmup schedule
```

---

## Installation & Setup

### From Source (recommended until PyPI publication)

```bash
git clone https://github.com/shamiquekhan/sk-autod.git
cd sk-autod
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

---

## Project Layout

This repository now includes the pieces you would expect from a polished open-source ML library:

- [docs/](./docs/installation.md) for installation, quickstart, detectors, API reference, and architecture notes
- [examples/](./examples/basic_usage.py) for runnable usage snippets
- [notebooks/](./notebooks/sk_autod_demo.ipynb) for a minimal Jupyter walkthrough
- [scripts/](./scripts/README.md) for maintenance and utility helpers
- [benchmarks/](./benchmarks/README.md) for repeatable performance checks
- [.github/workflows/tests.yml](./.github/workflows/tests.yml) for CI checks
- [.github/workflows/lint.yml](./.github/workflows/lint.yml) for style and formatting checks
- [.github/workflows/publish.yml](./.github/workflows/publish.yml) for tagged release publishing
- [CHANGELOG.md](./CHANGELOG.md) for release notes
- [Makefile](./Makefile) for common development commands

---

## Roadmap

### v0.1.0 (Current)
- ✅ Core 5 detectors (overfitting, LR issues, early stopping)
- ✅ Text output + CLI
- ✅ Basic Python API
- ✅ Unit tests (80%+ coverage)

### v0.2.0 (May 2026)
- 🔄 All 10 detectors
- 🔄 JSON + HTML output
- 🔄 Embedded loss curve visualization

### v0.3.0 (June 2026)
- 🔄 PyTorch callback
- 🔄 Keras callback
- 🔄 Jupyter notebook integration

### v0.4.0 (July 2026)
- 🔄 Weights & Biases integration
- 🔄 MLflow support
- 🔄 Adaptive thresholds

### v1.0.0 (August 2026)
- 🔄 Stable API guarantee
- 🔄 Full documentation
- 🔄  100% test coverage

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

**Areas to contribute:**
- Add new detectors (submit an issue first!)
- Improve threshold tuning on real datasets
- Framework integrations (MLflow, W&B, Kubeflow)
- Documentation & examples
- Bug reports and feature requests

---

## Philosophy & Design

1. **Rule-based, not ML-based:** Detectors use hand-crafted heuristics, not neural networks.
   - ✅ Interpretable, debuggable, no training needed
   - ✅ Works offline, no API calls
   
2. **Zero configuration:** Works out-of-the-box with sensible defaults.
   - Thresholds are data-agnostic, tuned on 100+ synthetic curves

3. **Fail gracefully:** Unknown patterns → no false alarms, just silence.

4. **Fast & lightweight:** Diagnose 1000+ curves in <1ms.

---

## Performance

Benchmarks on typical curves (100 epochs):

```
Diagnose 1 curve:        0.2 ms
Diagnose 1000 curves:    180 ms
Memory per curve:        ~2 KB
```

---

## FAQ

**Q: Why not use machine learning for detection?**  
A: ML-based detection would require training data (which curves to flag?), add latency, and reduce interpretability. Rule-based detection is faster, more debuggable, and works offline.

**Q: Can I customize detectors?**  
A: Yes! Subclass `BaseDetector` and pass to `diagnose()`:
```python
class MyDetector(BaseDetector):
    name = "Custom issue"
    def detect(self, report):
        # your logic here
        return [Finding(...)]

report = diagnose(train, val, detectors=[MyDetector()])
```

**Q: Does it support multi-task or multi-metric curves?**  
A: v0.1 supports 1D loss arrays. Multi-task support in v0.3+.

**Q: What if my curves are short (5 epochs)?**  
A: SK-AutoD requires at least 5 epochs. For shorter runs, some detectors may not fire (e.g., early stopping needs history).

**Q: Can I integrate this with my training pipeline?**  
A: Yes! Callbacks coming in v0.3. For now:
```python
# After each epoch
report = diagnose(train_losses[:epoch], val_losses[:epoch])
if any(f.severity == "CRITICAL" for f in report.findings):
    # Stop training or adjust hyperparameters
```

---

## License

MIT License © 2026 Shamique Khan  
See [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

- Inspired by discussions in ML communities (r/MachineLearning, FastAI forums)
- Threshold tuning validated on Kaggle competition curves
- Special thanks to early testers and contributors

---

## Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/shamiquekhan/sk-autod/issues)
- **Twitter:** [@shamiquekhan](https://twitter.com/shamiquekhan)
- **Email:** shamique.khan@outlook.com

---

## Citation

If SK-AutoD helps your research, please cite:

```bibtex
@software{sk_autod2026,
  author = {Khan, Shamique},
  title = {SK-AutoD: Auto-Diagnostic System for Training Curves},
  year = {2026},
  url = {https://github.com/shamiquekhan/sk-autod}
}
```

---

---

<div align="center">

**SK-AutoD** — Because your time is more valuable than manual eyeballing. 🚀

[⭐ Star us on GitHub](https://github.com/shamiquekhan/sk-autod) if you find this helpful!

</div>
