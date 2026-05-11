# Contributing to SK-AutoD

Thanks for your interest in contributing to SK-AutoD! 🎉

We welcome bug reports, feature requests, documentation improvements, and code contributions. This guide explains how to get started.

---

## Code of Conduct

By participating in this project, you agree to treat all contributors with respect and kindness. We're building a welcoming community for everyone.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- A GitHub account

### Local Development Setup

1. **Fork & clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician.git
cd SK-AutoD-ML-Library-for-Training-Curve-Auto-Diagnostician
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in editable mode with dev dependencies:**

```bash
pip install -e ".[dev]"
```

4. **Run quality checks:**

```bash
black sk_autod/ tests/
ruff check sk_autod/ tests/
mypy sk_autod/
pytest tests/ -v
```

---

## Types of Contributions

### 🐛 Bug Reports

Found a bug? Please open a GitHub issue with:

- **Description:** Clear summary of the bug
- **Steps to reproduce:** Exact code to trigger the bug
- **Expected vs actual behavior:** What should happen vs what does happen
- **Environment:** Python version, OS, CurveDoctor version
- **Traceback:** Full error message (if applicable)

**Example:**
```
Title: Overfitting detector fails on short curves (<10 epochs)

Description:
When I call diagnose() with 7 epochs of data, I get a ValueError.

Steps to reproduce:
```python
from curvedoctor import diagnose
train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
diagnose(train, val)  # Crashes here
```

Environment: Python 3.11, Windows 10, CurveDoctor 0.1.0
```

### 💡 Feature Requests

Have an idea? Open an issue with:

- **Description:** What feature would help?
- **Motivation:** Why do you need this?
- **Examples:** How would you use it?

**Example:**
```
Title: Support multi-metric diagnosis (loss + accuracy)

Motivation:
Currently CurveDoctor only supports loss curves. Many projects also track
accuracy, and diagnosing both together would catch issues like:
- Loss converging but accuracy stuck
- Accuracy oscillating while loss smooth

Example usage:
report = diagnose(train_loss, val_loss, train_acc, val_acc)
```

### 📝 Documentation

- **README improvements:** Clearer examples, better formatting
- **API documentation:** Docstring improvements, type hints
- **New guides:** Tutorials, integration examples
- **Blog posts:** Real-world usage stories (external repos welcome!)

### 🔧 Code Contributions

The best contributions solve real problems. Here are some areas:

#### Add a New Detector

1. **Create** `sk_autod/detectors/your_detector.py`
2. **Subclass** `BaseDetector`
3. **Implement** `detect()` method
4. **Add tests** to `tests/test_your_detector.py` with fixtures
5. **Update** `sk_autod/runner.py` to include it
6. **Document** threshold rationale in docstring

**Template:**

```python
from sk_autod.detectors.base import BaseDetector
from sk_autod.models import Finding, Severity, DiagnosisReport

class YourDetector(BaseDetector):
    """Detects [specific issue]."""
    
    name = "Your Issue Name"
    default_severity = Severity.HIGH
    
    def detect(self, report: DiagnosisReport) -> list[Finding]:
        """
        Detect your issue.
        
        Logic:
        - Check for specific pattern in loss curves
        - Return Finding if detected, empty list otherwise
        """
        findings = []
        
        # Your detection logic here
        if condition_met:
            findings.append(Finding(
                detector_name=self.name,
                severity=Severity.HIGH,
                signature="Short description of what was detected",
                logic="Detailed explanation of detection logic",
                fix_recommendation="What to do about it",
                confidence=0.85,
                affected_epoch=4,
            ))
        
        return findings
```

#### Improve Existing Detectors

- Lower false positives/negatives
- Tune thresholds on real datasets
- Add edge case handling
- Improve confidence scoring

#### Framework Integrations

- PyTorch callback: `CurveDoctorCallback`
- Keras callback: `KerasCurveDoctorCallback`
- Weights & Biases: Auto-log findings
- MLflow: Track diagnostic metadata

#### Other Areas

- Performance optimization
- Type hints (mypy compliance)
- Documentation generation
- CLI enhancements
- Test coverage

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming:
- `feature/` — new features
- `fix/` — bug fixes
- `docs/` — documentation
- `refactor/` — code improvements

### 2. Make Changes

- Write code following project style (see below)
- Write tests for new functionality
- Keep commits atomic and well-messaged

### 3. Run Quality Checks

```bash
# Format code
black curvedoctor/ tests/

# Lint
ruff check curvedoctor/ tests/

# Type check
mypy curvedoctor/

# Run tests
pytest tests/ -v --cov=curvedoctor

# Expected: 80%+ coverage
```

### 4. Commit & Push

```bash
git add .
git commit -m "Add your feature: clear description"
git push origin feature/your-feature-name
```

Commit message format:
```
Verb: Short description (50 chars max)

Longer explanation if needed (72 chars per line).
Explain WHY, not WHAT.

Fixes #123 (if applicable)
```

Examples:
```
Add: Dying ReLU detector with unit tests
Fix: Preprocessor crashes on NaN values
Docs: Clarify API usage in README
Refactor: Consolidate threshold logic into utils
```

### 5. Create a Pull Request

1. Go to GitHub
2. Click **New Pull Request**
3. Fill in template:

```markdown
## Description
Brief explanation of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Code refactor

## Testing
- [ ] Added unit tests
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] 80%+ coverage maintained

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes
```

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these tools:

#### Black (Code Formatting)

```bash
black curvedoctor/ tests/
```

#### Ruff (Linting)

```bash
ruff check curvedoctor/ tests/ --fix
```

Configuration (`.ruff.toml`):
```toml
[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "W"]
```

#### MyPy (Type Checking)

```bash
mypy curvedoctor/
```

All public functions should have type hints:

```python
def detect(self, report: DiagnosisReport) -> list[Finding]:
    """Detect overfitting pattern."""
    findings: list[Finding] = []
    return findings
```

### Docstring Style

Use Google-style docstrings:

```python
def preprocess(self, train_loss: list[float], val_loss: list[float]) -> PreprocessedData:
    """Validate, align, and smooth loss curves.
    
    Args:
        train_loss: Training loss values (≥5 epochs).
        val_loss: Validation loss values (same length as train_loss).
    
    Returns:
        PreprocessedData containing smoothed curves and statistics.
    
    Raises:
        ValueError: If curves too short or contain NaN/inf values.
    
    Examples:
        >>> preprocessor = Preprocessor()
        >>> result = preprocessor.preprocess([2.3, 1.9, 1.4], [2.4, 2.0, 1.8])
        >>> result.train_smooth.shape
        (3,)
    """
```

---

## Testing

### Test Structure

```
tests/
├── test_preprocessor.py       # Tests for Preprocessor
├── test_detectors.py          # Generic detector tests
├── test_overfitting.py        # Specific detector tests
├── test_runner.py             # DiagnosticsRunner tests
├── test_formatters.py         # Output formatters
└── fixtures.py                # Shared test curves
```

### Writing Tests

Use pytest fixtures for reusable test data:

```python
# tests/fixtures.py
import pytest

class CurveFixtures:
    @staticmethod
    def well_trained():
        """Curve with no pathologies."""
        train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
        val = [2.4, 2.0, 1.6, 1.4, 1.3, 1.2, 1.2]
        return train, val
    
    @staticmethod
    def overfitting():
        """Curve with classic overfitting pattern."""
        train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
        val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
        return train, val

# tests/test_overfitting.py
from curvedoctor.detectors.overfitting import OverfittingDetector
from tests.fixtures import CurveFixtures

def test_detects_overfitting():
    detector = OverfittingDetector()
    train, val = CurveFixtures.overfitting()
    findings = detector.detect(train, val)
    
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].confidence > 0.8

def test_no_false_positive_on_good_curve():
    detector = OverfittingDetector()
    train, val = CurveFixtures.well_trained()
    findings = detector.detect(train, val)
    
    assert len(findings) == 0
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_overfitting.py -v

# Run with coverage
pytest tests/ --cov=curvedoctor --cov-report=html

# Run and stop on first failure
pytest tests/ -x

# Run specific test
pytest tests/test_overfitting.py::test_detects_overfitting -v
```

### Coverage Target

- **Minimum:** 80% overall
- **Detectors:** 95% (critical logic)
- **Formatters:** 85%
- **Utils:** 70% (less critical)

---

## Documentation

### Docstring Requirements

Every public function/class needs a docstring:

```python
def diagnose(
    train_loss: list[float],
    val_loss: list[float],
    detectors: list[BaseDetector] | None = None,
) -> DiagnosisReport:
    """Analyze training curves for pathologies.
    
    Entry point for diagnosis. Automatically runs all detectors,
    deduplicates findings, and returns a ranked report.
    
    Args:
        train_loss: Training loss per epoch (≥5 epochs).
        val_loss: Validation loss per epoch (same length).
        detectors: Custom detector list (default: all 10 detectors).
    
    Returns:
        DiagnosisReport with findings sorted by severity.
    
    Raises:
        ValueError: If input validation fails.
    
    Examples:
        >>> from curvedoctor import diagnose
        >>> train = [2.3, 1.9, 1.4, 0.9, 0.5, 0.3, 0.15]
        >>> val = [2.4, 2.0, 1.8, 1.9, 2.3, 2.8, 3.4]
        >>> report = diagnose(train, val)
        >>> print(report.summary())
    """
```

### README Updates

- Add examples of new features
- Link to relevant documentation
- Update feature tables

### New Guides

Create guides in `docs/guides/`:
- Integration with PyTorch
- Integration with Keras
- Custom detector development
- Threshold tuning on your domain

---

## Performance Guidelines

### Optimization Rules

1. **Detect should be fast:** <1ms per detector (100 epoch curve)
2. **No external API calls:** Must work offline
3. **Minimal memory:** ~2KB per curve
4. **Vectorized math:** Use NumPy operations, not loops

### Profiling

```python
import time

start = time.perf_counter()
report = diagnose(train, val)
elapsed = time.perf_counter() - start

print(f"Diagnosis took {elapsed*1000:.1f}ms")
```

---

## Security

### Reporting Security Issues

⚠️ **Do not open public GitHub issues for security vulnerabilities.**

Email: shamique@example.com with:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

---

## Release Process

1. **Bump version** in `pyproject.toml`
2. **Update CHANGELOG.md** with changes
3. **Create git tag:** `git tag v0.2.0`
4. **Push:** `git push --tags`
5. **GitHub Actions** automatically:
   - Runs tests
   - Builds package
   - Publishes to PyPI

---

## Review Process

### What Reviewers Look For

- ✅ Does it solve a real problem?
- ✅ Are tests comprehensive?
- ✅ Is code style consistent?
- ✅ Are breaking changes justified?
- ✅ Is documentation updated?
- ✅ Does it maintain 80%+ coverage?

### Response Times

- Bug fixes: 2-3 days
- New features: 5-7 days
- Documentation: 1-2 days

---

## Recognition

Contributors are recognized in:
- **README:** Contributors section
- **CHANGELOG:** Version credits
- **GitHub:** Automatic contributor badge

---

## Questions?

- **Discussions:** GitHub Discussions tab
- **Twitter:** @shamiquekhan
- **Email:** shamique@example.com

---

## Thank You! 🙏

Every contribution makes CurveDoctor better. Whether it's code, documentation, bug reports, or ideas — we appreciate your help!

Let's build something amazing together. 🚀
