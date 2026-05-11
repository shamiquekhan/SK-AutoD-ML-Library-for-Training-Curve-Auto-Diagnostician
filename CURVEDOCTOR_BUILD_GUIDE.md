# SK-AutoD: Training Curve Auto-Diagnostician

## Complete Build & Launch Guide

**Version:** 0.1.0 (MVP)  
**Timeline:** 4 weeks  
**Target Users:** ML engineers, researchers, practitioners doing manual curve eyeballing  

---

## Problem Statement

Every ML practitioner spends hours staring at loss curves during training:
- "Is this overfitting?"
- "Did my learning rate explode?"
- "Should I have stopped earlier?"
- "Why is my loss stuck?"

**Current workflow:** Manual eyeballing + Slack screenshots + tribal knowledge.  
**CurveDoctor solves this:** Paste in your arrays → Get instant, rule-based diagnosis.

---

## Project Overview

**SK-AutoD** is a Python library that analyzes training curves (loss, accuracy, metrics) and automatically detects 10+ distinct training pathologies:

- **Classic overfitting** — val loss rises while train loss falls
- **Exploding gradient** — loss spikes >300% in one epoch
- **Learning rate too high** — oscillations without convergence
- **Learning rate too low** — glacially slow improvement
- **Underfitting** — both losses plateau at high values
- **Dying ReLU proxy** — loss flatlines early, high value
- **Noisy/unstable training** — jagged loss throughout
- **Data leakage** — val loss consistently lower than train
- **Missed early stopping** — didn't use the best checkpoint
- **Label noise floor** — loss can't drop below a threshold

Each detection includes:
- **Root cause diagnosis** with confidence
- **Fix recommendations** (e.g., "Reduce LR by 5–10×", "Add dropout p=0.3–0.5")
- **Severity level** (CRITICAL / HIGH / MEDIUM / WARNING)

---

## Architecture & Data Flow

```
User Input (arrays)
       ↓
Preprocessor (align, smooth, compute stats)
       ↓
DiagnosticsRunner (fire all detectors in parallel)
       ↓
[Detector 1] [Detector 2] ... [Detector N]
       ↓
DiagnosisReport (deduplicate, sort by severity)
       ↓
Formatters → Text | JSON | HTML
       ↓
Output (summary, dict, HTML report)
```

### Core Components

#### 1. **Data Models** (`sk_autod/models.py`)

```python
@dataclass
class Finding:
    """A single diagnostic finding."""
    detector_name: str          # "Classic overfitting"
    severity: Severity          # CRITICAL, HIGH, MEDIUM, WARNING
    signature: str              # "Val loss rises while train loss falls"
    logic: str                  # "Detect epoch where val_loss delta > 0..."
    fix_recommendation: str     # "Add dropout (p=0.3–0.5), L2 regularisation..."
    confidence: float           # 0.0–1.0
    affected_epoch: int | None  # Which epoch triggered this finding
    metadata: dict              # detector-specific data

@dataclass
class DiagnosisReport:
    """Collection of findings + utilities."""
    findings: list[Finding]
    train_loss: np.ndarray
    val_loss: np.ndarray
    processed_at: datetime
    
    def summary(self) -> str:
        """Human-readable text report."""
    
    def to_dict(self) -> dict:
        """JSON-serializable dict."""
    
    def to_html(self) -> str:
        """Standalone HTML report with chart."""
```

#### 2. **Preprocessor** (`sk_autod/preprocessor.py`)

Normalizes, smooths, and enriches raw input:

```python
class Preprocessor:
    def __init__(self, ema_alpha=0.3):
        self.ema_alpha = ema_alpha
    
    def preprocess(self, train_loss, val_loss):
        # Validate: non-empty, numeric, same length
        # Align: pad shorter array if needed
        # Smooth: exponential moving average
        # Compute: delta arrays, rolling stats (mean, std, min/max)
        # Return: enriched arrays + metadata
```

**Output:** Normalized arrays + precomputed rolling statistics.

#### 3. **Base Detector Interface** (`sk_autod/detectors/base.py`)

```python
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Template for all detectors."""
    
    name: str                  # "Overfitting"
    default_severity: Severity # CRITICAL
    
    @abstractmethod
    def detect(self, report: DiagnosisReport) -> list[Finding]:
        """
        Analyze preprocessed curves.
        Return list of findings (0+ per detector).
        """
        pass
```

Each detector is **stateless** and **deterministic** — same input → same output.

#### 4. **10 Detector Implementations** (`sk_autod/detectors/*.py`)

Each detector inherits `BaseDetector` and implements `detect()`:

| Detector | File | Logic |
|----------|------|-------|
| **Classic Overfitting** | `overfitting.py` | Val ↑ while train ↓ for 3+ epochs |
| **Exploding Gradient** | `gradient.py` | Loss spike >300% in 1 epoch |
| **LR Too High** | `lr_high.py` | High variance + no downtrend |
| **LR Too Low** | `lr_low.py` | Slope < -0.001/epoch in first 50% |
| **Underfitting** | `underfitting.py` | Both losses plateau high + no divergence |
| **Dying ReLU Proxy** | `dying_relu.py` | Flatline before epoch 10 |
| **Noisy Training** | `noisy.py` | >40% sign flips in deltas |
| **Data Leakage** | `leakage.py` | Train loss consistently > val loss |
| **Missed Early Stop** | `early_stop.py` | Val min not in last 10% |
| **Label Noise Floor** | `label_noise.py` | Loss flatlines above expected floor |

#### 5. **Diagnostics Runner** (`sk_autod/runner.py`)

Orchestrates all detectors:

```python
class DiagnosticsRunner:
    def __init__(self, detectors: list[BaseDetector] | None = None):
        self.detectors = detectors or [
            OverfittingDetector(),
            ExplodingGradientDetector(),
            # ... etc
        ]
    
    def diagnose(self, train_loss, val_loss) -> DiagnosisReport:
        # 1. Preprocess
        preprocessor = Preprocessor()
        train, val, metadata = preprocessor.preprocess(train_loss, val_loss)
        
        # 2. Create report container
        report = DiagnosisReport(train_loss=train, val_loss=val, ...)
        
        # 3. Fire all detectors in parallel (or sequential, depending on perf)
        findings = []
        for detector in self.detectors:
            findings.extend(detector.detect(report))
        
        # 4. Deduplicate conflicting findings
        findings = self._deduplicate(findings)
        
        # 5. Sort by severity
        findings = sorted(findings, key=lambda f: severity_order[f.severity])
        
        report.findings = findings
        return report
```

**Deduplication logic:** If two detectors fire (e.g., "LR too high" + "oscillating"), merge them into the most specific finding.

#### 6. **Formatters** (`sk_autod/formatters/*.py`)

Three output channels:

**Text Formatter** (`text.py`):
```
CRITICAL: Classic overfitting
  Detected at epoch 4 (94% confidence)
  Val loss rose while train loss fell for 3 consecutive epochs.
  
  Fix: Add dropout (p=0.3–0.5), L2 regularisation, or reduce model capacity.

HIGH: LR too high
  Loss oscillated without a clear downward trend.
  
  Fix: Reduce LR by 5–10×, or add warmup schedule.
```

**JSON Formatter** (`json.py`):
```json
{
  "findings": [
    {
      "detector_name": "Classic overfitting",
      "severity": "CRITICAL",
      "confidence": 0.94,
      "affected_epoch": 4,
      "fix_recommendation": "Add dropout..."
    }
  ],
  "summary": "1 critical issue found."
}
```

**HTML Formatter** (`html.py`):
```html
<div class="report">
  <h2>CurveDoctor Diagnosis</h2>
  <div class="finding critical">
    <h3>Classic overfitting</h3>
    <p>Val loss rose while train loss fell...</p>
    <button>View chart</button>
  </div>
  <canvas id="curve-chart"><!-- Embedded loss curve --></canvas>
</div>
```

---

## Feature List (MVP v0.1.0)

### Core Features
- [ ] Preprocess curves (align, smooth, compute stats)
- [ ] 5 initial detectors (overfitting, LR issues, early stopping, underfitting, gradient explosion)
- [ ] Deduplication logic
- [ ] Text output + formatted CLI
- [ ] Basic Python API: `diagnose(train_loss, val_loss)`
- [ ] Unit tests for all detectors (with synthetic fixture curves)

### Future (v0.2.0+)
- [ ] Remaining 5 detectors
- [ ] JSON output
- [ ] HTML report with embedded loss curve chart
- [ ] PyTorch callback integration
- [ ] Keras callback integration
- [ ] Weights & Biases integration
- [ ] MLflow integration
- [ ] Jupyter magic command (`%curvedoctor`)
- [ ] VS Code extension

---

## 6-Phase Build Plan

### **Phase 1: Core Data Layer** (Week 1, 1–2 days)

**Deliverables:**
- `curvedoctor/models.py` — `Finding`, `DiagnosisReport` dataclasses
- `curvedoctor/preprocessor.py` — Input validation, alignment, smoothing, stats computation
- Basic project scaffold: `setup.py`, `pyproject.toml`, `.gitignore`

**Checklist:**
- [ ] Create project structure
- [ ] Define dataclasses with clear docstrings
- [ ] Implement Preprocessor with EMA smoothing
- [ ] Write validation logic (non-empty, numeric, min length)
- [ ] Unit tests for Preprocessor (5+ fixtures)

**Testing:**
```python
# Test with synthetic curves
train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
result = preprocessor.preprocess(train, val)
assert result['train_smooth'].shape == (7,)
assert result['train_delta'].shape == (6,)
```

---

### **Phase 2: First 3 Detectors** (Week 1–2, 2–3 days)

**Deliverables:**
- `curvedoctor/detectors/base.py` — BaseDetector abstract class
- `curvedoctor/detectors/overfitting.py` — Overfitting detector
- `curvedoctor/detectors/lr_high.py` — LR too high detector
- `curvedoctor/detectors/early_stop.py` — Early stopping detector
- Unit tests for each detector

**Checklist:**
- [ ] Implement BaseDetector interface
- [ ] Overfitting: detect val ↑ + train ↓ for 3+ epochs
- [ ] LR high: detect oscillations + no downtrend
- [ ] Early stop: find global min of val_loss, check if in last 10%
- [ ] Write 5+ fixture curves per detector (overfitting, noisy, well-trained, etc.)
- [ ] Validate confidence scoring

**Testing:**
```python
def test_overfitting_detector():
    detector = OverfittingDetector()
    # Well-trained curve: should NOT trigger
    train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
    val = [2.4, 2.0, 1.6, 1.4, 1.3, 1.2, 1.2]
    report = DiagnosisReport(train_loss=train, val_loss=val, ...)
    findings = detector.detect(report)
    assert len(findings) == 0
    
    # Overfitting curve: SHOULD trigger
    train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
    val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]  # diverges
    findings = detector.detect(report)
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL
```

---

### **Phase 3: Remaining 7 Detectors** (Week 2–3, 3–4 days)

**Deliverables:**
- All remaining detector files
- Comprehensive fixture library
- Threshold tuning (via synthetic data + manual validation)

**Detectors to build:**
1. Exploding gradient — single spike >300%
2. LR too low — slow slope (<-0.001/epoch)
3. Underfitting — both losses plateau high
4. Dying ReLU proxy — flatline before epoch 10
5. Noisy training — >40% sign flips
6. Data leakage — train > val consistently
7. Label noise floor — loss flatlines above threshold

**Checklist:**
- [ ] Implement all 7 detectors
- [ ] Tune thresholds on realistic synthetic data
- [ ] Write unit tests (fixtures) for each
- [ ] Cross-validate: detector shouldn't fire on "good" curves
- [ ] Document threshold rationale in docstrings

**Example threshold validation:**
```python
# Threshold: LR too low if slope > -0.001/epoch in first 50%
# Test on curves with varying learning rates
```

---

### **Phase 4: Runner + Deduplication** (Week 3, 1–2 days)

**Deliverables:**
- `curvedoctor/runner.py` — DiagnosticsRunner class
- Deduplication logic
- `curvedoctor/__init__.py` — Public API

**Checklist:**
- [ ] Implement DiagnosticsRunner
- [ ] Parallel/sequential detector execution
- [ ] Deduplication: merge conflicting findings (e.g., "LR too high" supersedes generic "oscillation")
- [ ] Sort findings by severity
- [ ] High-level API: `diagnose(train_loss, val_loss)` returns `DiagnosisReport`
- [ ] Unit tests for runner + deduplication

**Deduplication examples:**
```
Input: ["oscillations detected", "LR too high"]
Output: ["LR too high"]  # More specific

Input: ["overfitting", "val loss diverging"]
Output: ["overfitting"]  # Subsumes the other
```

---

### **Phase 5: Formatters** (Week 3, 1–2 days)

**Deliverables:**
- `curvedoctor/formatters/text.py` — Human-readable text
- `curvedoctor/formatters/json.py` — JSON output
- `report.summary()`, `report.to_dict()` methods

**Checklist:**
- [ ] Text formatter: pretty-printed findings with colors (optional: use `rich`)
- [ ] JSON formatter: clean, serializable output
- [ ] Report methods: `.summary()` → str, `.to_dict()` → dict
- [ ] Unit tests (snapshot tests for text format)
- [ ] CLI integration: `curvedoctor diagnose --train-loss [...] --val-loss [...]`

**Text output example:**
```
═══════════════════════════════════════
  CurveDoctor Diagnosis Report
═══════════════════════════════════════

🔴 CRITICAL: Classic overfitting
   Detected at epoch 4 (94% confidence)
   Val loss rose while train loss fell for 3 consecutive epochs.
   
   Fix: Add dropout (p=0.3–0.5), L2 regularisation, or reduce model capacity.

🟠 HIGH: LR too high
   Loss oscillated without a clear downward trend (rolling variance 2.3× baseline).
   
   Fix: Reduce LR by 5–10×, or add warmup schedule.

═══════════════════════════════════════
Summary: 2 issues found (1 critical, 1 high)
```

---

### **Phase 6: CLI + Packaging** (Week 4, 2–3 days)

**Deliverables:**
- `curvedoctor/cli.py` — Click/argparse CLI
- `pyproject.toml` — Package metadata, dependencies
- `README.md` — Installation + quickstart
- Published on PyPI

**Checklist:**
- [ ] Build CLI using Click or argparse
  ```bash
  curvedoctor diagnose --train-loss 2.3 1.9 1.4 --val-loss 2.4 2.0 1.8
  ```
- [ ] Accept input from:
  - Command-line arguments
  - CSV files
  - JSON files
  - stdin (pipe-friendly)
- [ ] Configure `pyproject.toml`:
  ```toml
  [project]
  name = "curvedoctor"
  version = "0.1.0"
  dependencies = ["numpy>=1.20", "click>=8.0"]
  
  [project.scripts]
  curvedoctor = "curvedoctor.cli:main"
  ```
- [ ] Build: `python -m build`
- [ ] Publish to PyPI: `python -m twine upload dist/*`
- [ ] Verify: `pip install curvedoctor` works

**CLI Examples:**
```bash
# From command line
curvedoctor diagnose \
  --train-loss 2.3 1.9 1.4 0.9 0.5 0.3 0.15 \
  --val-loss 2.4 2.0 1.8 1.9 2.3 2.8 3.4 \
  --output json

# From CSV file
curvedoctor diagnose --train-file train_losses.csv --val-file val_losses.csv

# From Python
from curvedoctor import diagnose
report = diagnose([2.3, 1.9, 1.4], [2.4, 2.0, 1.8])
print(report.summary())
```

---

## Project Structure

```
sk_autod/
├── __init__.py                # Public API exports
├── models.py                  # Finding, DiagnosisReport dataclasses
├── preprocessor.py            # Input validation, smoothing, stats
├── runner.py                  # DiagnosticsRunner orchestrator
├── detectors/
│   ├── __init__.py
│   ├── base.py                # BaseDetector abstract class
│   ├── overfitting.py
│   ├── gradient.py            # Exploding gradient
│   ├── lr_high.py
│   ├── lr_low.py
│   ├── underfitting.py
│   ├── dying_relu.py
│   ├── noisy.py
│   ├── leakage.py
│   ├── early_stop.py
│   └── label_noise.py
├── formatters/
│   ├── __init__.py
│   ├── text.py                # Human-readable output
│   └── json.py                # JSON serialization
├── cli.py                     # Click/argparse CLI
└── tests/
    ├── test_preprocessor.py
    ├── test_detectors.py
    └── fixtures.py            # Synthetic curve library

pyproject.toml                # Package config
README.md                     # Installation + quick start
ARCHITECTURE.md               # This document
```

---

## Development Workflow

### Environment Setup

```bash
# Clone + venv
git clone https://github.com/yourusername/curvedoctor.git
cd curvedoctor
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev deps
pip install -e ".[dev]"

# pyproject.toml should include:
[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov", "black", "ruff", "mypy"]
```

### Testing Strategy

**Fixture Curves:**
```python
# tests/fixtures.py
class CurveFixtures:
    @staticmethod
    def well_trained():
        """Smooth convergence, no pathology."""
        return [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15], [2.4, 2.0, 1.6, 1.4, 1.3, 1.2, 1.2]
    
    @staticmethod
    def overfitting():
        """Classic divergence pattern."""
        return [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15], [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
    
    @staticmethod
    def lr_too_high():
        """Oscillations without convergence."""
        return [2.3, 1.8, 1.6, 1.9, 1.5, 1.7, 1.4], [2.5, 2.0, 1.9, 2.1, 1.8, 2.0, 1.9]
    
    # ... etc for all 10 pathologies
```

**Unit Tests:**
```bash
# Run all tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=curvedoctor --cov-report=html

# Target: 80%+ coverage on detector logic
```

### Git Workflow

```bash
# Branch for each phase
git checkout -b phase-1-data-layer
git commit -m "Add Finding and DiagnosisReport dataclasses"
git push origin phase-1-data-layer

# Then PR + merge to main
```

---

## After Launch: Roadmap & Next Steps

### Week 1–2 (Post-launch): Community Feedback

- [ ] Publish on Reddit (r/MachineLearning, r/learnmachinelearning)
- [ ] Share on HackerNews (Show HN)
- [ ] Collect feedback: Which detectors are useful? Which thresholds need tuning?
- [ ] Set up GitHub Issues for feature requests

### v0.2.0 (Week 3–4): Full Feature Parity

- [ ] Implement remaining 5 detectors
- [ ] JSON + HTML output formatters
- [ ] Embedded loss curve visualization (matplotlib → base64 PNG)
- [ ] Publish v0.2.0 to PyPI

### v0.3.0 (Month 2): Framework Integration

- [ ] PyTorch callback: `CurveDoctorCallback()`
- [ ] Keras callback: `CurveDoctorCallback()`
  ```python
  model.fit(..., callbacks=[CurveDoctorCallback()])
  ```
- [ ] Notebook integration: `report.plot_with_annotations()`
- [ ] Weights & Biases logger:
  ```python
  wandb.log({"curvedoctor": report.to_dict()})
  ```

### v0.4.0 (Month 2–3): Intelligence Upgrades

- [ ] Adaptive thresholds based on dataset size + model type
- [ ] Multi-phase detection (e.g., "LR ok in phase 1, too high in phase 2")
- [ ] Confidence scoring refinement
- [ ] A/B test detector rules on community datasets

### v1.0.0 (Month 3–4): Production Ready

- [ ] 100% test coverage
- [ ] Full API documentation (Sphinx + ReadTheDocs)
- [ ] Performance benchmarks (can diagnose 1000 curves in <1ms)
- [ ] Stable API guarantee (no breaking changes)
- [ ] Production users + case studies

---

## Marketing & Distribution Strategy

### Phase 1: Awareness (Week 1–2)
- **Target:** Early adopters (researchers, Kaggle competitors)
- **Channels:**
  - Reddit (r/MachineLearning, r/datascience)
  - Twitter/X (tag @kaggle, @pytorch, @OpenAI)
  - HackerNews "Show HN"
  - ProductHunt (Day 1 launch)

### Phase 2: Integration (Week 3–4)
- **Partner with:**
  - Weights & Biases (feature on their blog)
  - Keras/TensorFlow community
  - PyTorch community
- **Sponsorship:** PyData conferences

### Phase 3: Monetization (Month 2+)
- **Free tier:** Core 5 detectors, CLI only
- **Pro tier:** All 10+ detectors, callbacks, integrations ($9/mo)
- **Enterprise:** Custom detectors, private SaaS instance

---

## Success Metrics

**v0.1.0 Launch:**
- [ ] 500+ PyPI downloads in first month
- [ ] 50+ GitHub stars
- [ ] 10+ community issues/PRs
- [ ] Detectors achieve >90% accuracy on synthetic test set

**v0.2.0:**
- [ ] 2,000+ PyPI downloads/month
- [ ] 200+ GitHub stars
- [ ] 5+ real-world integrations (Weights & Biases logs, etc.)

**v1.0.0:**
- [ ] 10,000+ PyPI downloads/month
- [ ] 1,000+ GitHub stars
- [ ] Featured in "Awesome ML" lists
- [ ] Adopted by 2+ research labs + companies

---

## Technology Stack

**Runtime:**
- Python 3.10+ (type hints, dataclasses)
- NumPy (array operations, statistics)
- Click (CLI framework, optional but recommended)

**Development:**
- pytest (testing)
- pytest-cov (coverage)
- black (formatting)
- ruff (linting)
- mypy (type checking)
- Sphinx (docs)

**Deployment:**
- PyPI (distribution)
- GitHub Actions (CI/CD)
- ReadTheDocs (documentation hosting)

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Detectors fire on noisy but "fine" curves | Use multiple confirmation rules; tune thresholds on 50+ realistic curves |
| False negatives (missing real pathologies) | Validate against manual expert eyeballing; collect benchmark dataset |
| Threshold brittleness (breaks on new domains) | Add adaptive thresholds; allow calibration per user |
| High false positive rate | Increase confidence threshold; combine multiple heuristics per finding |
| Slow on large curves | Lazy-evaluate detectors; cache preprocessed stats |
| Poor user messaging | A/B test fix recommendations; collect feedback on clarity |

---

## Key Implementation Details

### Confidence Scoring

Each finding should have a confidence score (0.0–1.0):

```python
# Example: Overfitting detector
def detect(self, report) -> list[Finding]:
    # Count epochs where val delta > 0 and train delta < 0
    overfitting_epochs = sum(1 for v, t in zip(val_deltas, train_deltas) 
                             if v > 0 and t < 0)
    
    if overfitting_epochs >= 3:
        # Confidence = (gap magnitude + duration) / max_possible
        avg_gap = np.mean([v - t for v, t in zip(val_deltas, train_deltas) 
                           if v > 0 and t < 0])
        confidence = min(1.0, (avg_gap / max_possible_gap) * (overfitting_epochs / 5))
        
        return [Finding(
            detector_name="Classic overfitting",
            severity=Severity.CRITICAL,
            confidence=confidence,
            ...
        )]
    return []
```

### Smooth Curves Properly

Use **exponential moving average (EMA)** to smooth without losing sharp changes:

```python
def ema_smooth(values, alpha=0.3):
    """Exponential moving average."""
    ema = [values[0]]
    for val in values[1:]:
        ema.append(alpha * val + (1 - alpha) * ema[-1])
    return np.array(ema)
```

**Why EMA?**
- Preserves spike detection (e.g., exploding gradients)
- Smooths noise without lag
- Simple, fast, interpretable

### Handle Edge Cases

```python
# Minimum data: 5 epochs
if len(train_loss) < 5:
    raise ValueError("Need at least 5 epochs of data")

# Missing values
if np.any(np.isnan(train_loss)):
    raise ValueError("NaN values in input")

# Infinite loss (exploded)
if np.any(np.isinf(train_loss)):
    finding = Finding(name="Exploding gradient", ...)

# Constant curve (no training)
if np.std(train_loss) < 1e-6:
    finding = Finding(name="No convergence", ...)
```

---

## Documentation Structure

### README.md
```markdown
# CurveDoctor 🩺

Auto-diagnose your training curves.

## Installation
pip install curvedoctor

## Quick Start
from curvedoctor import diagnose

report = diagnose(
    train_loss=[2.3, 1.9, 1.4, ...],
    val_loss=[2.4, 2.0, 1.8, ...]
)
print(report.summary())

## Features
- 10 detectors for common training pathologies
- ...
```

### API.md
- Full API reference
- Detector details (logic, thresholds, examples)
- Examples for each detector

### DEVELOPMENT.md
- Contributing guidelines
- Local setup
- Running tests
- Adding custom detectors

### CHANGELOG.md
- Version history
- Backward compatibility notes

---

## Final Checklist Before v0.1.0 Release

### Code Quality
- [ ] All detectors tested with 5+ fixtures each
- [ ] 80%+ test coverage
- [ ] Type hints on all functions
- [ ] Docstrings for all public APIs
- [ ] No hardcoded magic numbers (all thresholds documented)

### Documentation
- [ ] README with installation + examples
- [ ] API documentation (all classes, methods)
- [ ] Detector logic documented (threshold rationale)
- [ ] Contributing guidelines

### Packaging
- [ ] `pyproject.toml` complete + tested
- [ ] `setup.py` not needed (PEP 517+)
- [ ] Package builds without errors: `python -m build`
- [ ] Can install from wheel: `pip install dist/curvedoctor-0.1.0-py3-none-any.whl`

### Testing
- [ ] `pytest tests/ -v` passes 100%
- [ ] `pytest tests/ --cov=curvedoctor` shows 80%+ coverage
- [ ] No warnings from mypy, ruff, black

### Release
- [ ] Tagged version: `git tag v0.1.0`
- [ ] Published to PyPI: `twine upload dist/*`
- [ ] Installable via pip: `pip install curvedoctor==0.1.0`
- [ ] GitHub release with notes

---

## Getting Started Right Now

### Step 1: Initialize Project (30 min)

```bash
mkdir sk-autod && cd sk-autod
git init

# Create structure
mkdir -p sk_autod/detectors sk_autod/formatters tests
touch sk_autod/__init__.py
touch pyproject.toml README.md

# Git
git config user.name "Your Name"
git add .
git commit -m "Initial project structure"
```

### Step 2: Write pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sk-autod"
version = "0.1.0"
description = "Auto-diagnose your training curves"
authors = [{name = "Your Name", email = "your@email.com"}]
requires-python = ">=3.10"
dependencies = ["numpy>=1.20"]

[project.optional-dependencies]
dev = ["pytest>=7.0", "black", "ruff", "mypy"]

[project.scripts]
sk-autod = "sk_autod.cli:main"
```

### Step 3: Start Phase 1 (Models + Preprocessor)

Create `sk_autod/models.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    WARNING = "WARNING"

@dataclass
class Finding:
    detector_name: str
    severity: Severity
    signature: str
    logic: str
    fix_recommendation: str
    confidence: float
    affected_epoch: int | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class DiagnosisReport:
    findings: list[Finding]
    train_loss: np.ndarray
    val_loss: np.ndarray
    processed_at: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> str:
        """Return formatted text summary."""
        # Implement in Phase 5
        pass
    
    def to_dict(self) -> dict:
        """Return JSON-serializable dict."""
        # Implement in Phase 5
        pass
```

Create `sk_autod/preprocessor.py`:

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class PreprocessedData:
    train_loss: np.ndarray
    val_loss: np.ndarray
    train_smooth: np.ndarray
    val_smooth: np.ndarray
    train_delta: np.ndarray
    val_delta: np.ndarray
    metadata: dict

class Preprocessor:
    def __init__(self, ema_alpha=0.3):
        self.ema_alpha = ema_alpha
    
    def preprocess(self, train_loss, val_loss) -> PreprocessedData:
        """Validate, align, smooth, and compute stats."""
        train = np.array(train_loss, dtype=float)
        val = np.array(val_loss, dtype=float)
        
        # Validation
        if len(train) < 5:
            raise ValueError("Need at least 5 epochs")
        if np.any(np.isnan(train)) or np.any(np.isnan(val)):
            raise ValueError("NaN values in input")
        
        # Align
        max_len = max(len(train), len(val))
        # ... implement alignment if needed
        
        # Smooth
        train_smooth = self._ema(train)
        val_smooth = self._ema(val)
        
        # Deltas
        train_delta = np.diff(train_smooth)
        val_delta = np.diff(val_smooth)
        
        return PreprocessedData(
            train_loss=train,
            val_loss=val,
            train_smooth=train_smooth,
            val_smooth=val_smooth,
            train_delta=train_delta,
            val_delta=val_delta,
            metadata={}
        )
    
    def _ema(self, values):
        """Exponential moving average."""
        ema = [values[0]]
        for val in values[1:]:
            ema.append(self.ema_alpha * val + (1 - self.ema_alpha) * ema[-1])
        return np.array(ema)
```

### Step 4: Write Tests

Create `tests/test_preprocessor.py`:

```python
import pytest
from curvedoctor.preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    return Preprocessor()

def test_basic_preprocessing(preprocessor):
    train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
    val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
    
    result = preprocessor.preprocess(train, val)
    
    assert result.train_smooth.shape == (7,)
    assert result.val_smooth.shape == (7,)
    assert result.train_delta.shape == (6,)
    assert result.val_delta.shape == (6,)

def test_validation_short_curves(preprocessor):
    with pytest.raises(ValueError, match="at least 5"):
        preprocessor.preprocess([1, 2, 3], [1, 2, 3])
```

### Step 5: Install & Test

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## You're Ready!

This guide gives you everything needed to build CurveDoctor from scratch. The key to success:

1. **Start small:** Phase 1 (models) → Phase 2 (3 detectors) → expand
2. **Test early:** Write fixtures for each detector before implementation
3. **Collect feedback:** Launch v0.1.0 early, iterate based on user input
4. **Document thresholds:** Every magic number gets a rationale
5. **Celebrate milestones:** Each phase is shippable

---

**Good luck! 🚀**

For updates, questions, or contributions:
- GitHub: github.com/shamiquekhan/SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician
- Twitter: @shamiquekhan
- Email: shamique@example.com
