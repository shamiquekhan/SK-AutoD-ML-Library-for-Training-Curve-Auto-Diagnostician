# SK-AutoD Architecture

**Complete technical design and implementation guide for developers.**

This document explains how SK-AutoD works internally, the design decisions, and how to extend it.

---

## Table of Contents

1. [Data Flow](#data-flow)
2. [Core Components](#core-components)
3. [Detector System](#detector-system)
4. [Deduplication Logic](#deduplication-logic)
5. [Preprocessor Details](#preprocessor-details)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Performance Considerations](#performance-considerations)

---

## Data Flow

```
┌─────────────────────────┐
│  User Input Arrays      │ [train_loss, val_loss]
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Preprocessor                       │
│  - Validate (non-empty, numeric)    │
│  - Align (if length mismatch)       │
│  - Smooth (exponential moving avg)  │
│  - Compute deltas & rolling stats   │
└────────────┬────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│  DiagnosisReport (container)                                │
│  - train_loss, val_loss (original)                          │
│  - train_smooth, val_smooth (preprocessed)                  │
│  - train_delta, val_delta (epoch-to-epoch changes)          │
│  - rolling stats (mean, std, min, max over windows)         │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│  DiagnosticsRunner                                            │
│  - Create DetectorInstances                                  │
│  - Fire all detectors (parallel-ready)                       │
│  - Collect findings from each                                │
└────────────┬───────────────────────────────────────────────────┘
             │
    ┌────────┴────────┬─────────┬─────────┐
    │                 │         │         │
    ▼                 ▼         ▼         ▼
 [Det 1]         [Det 2]   [Det 3]  [Det N]
    │                 │         │         │
    └────────┬────────┴─────────┴─────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Deduplication                       │
│  - Merge conflicting findings        │
│  - Keep most specific diagnosis      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│  Severity Sort                       │
│  - CRITICAL → HIGH → MEDIUM → WARNING │
└────────────┬─────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────┐
│  DiagnosisReport (populated)                  │
│  - findings: list[Finding] (sorted)           │
└────────────┬─────────────────────────────────────┘
             │
    ┌────────┴──────────┬───────────┐
    │                   │           │
    ▼                   ▼           ▼
[Text Formatter]  [JSON Formatter] [HTML Formatter]
    │                   │           │
    └────────┬──────────┴───────────┘
             │
             ▼
      [User Output]
  (summary() / to_dict() / to_html())
```

---

## Core Components

### 1. Data Models (`sk_autod/models.py`)

#### `Severity` (Enum)

Ranking of issue importance:

```python
class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # Training is broken, stop immediately
    HIGH = "HIGH"           # Major issue, likely hurts performance
    MEDIUM = "MEDIUM"       # Noticeable issue, should address
    WARNING = "WARNING"     # Minor issue, good to know
```

**Ordering:** `CRITICAL > HIGH > MEDIUM > WARNING`

#### `Finding` (Dataclass)

A single diagnostic issue:

```python
@dataclass
class Finding:
    detector_name: str              # "Classic overfitting"
    severity: Severity              # CRITICAL, HIGH, etc.
    signature: str                  # "Val loss rises while train loss falls"
    logic: str                      # Detailed explanation of detection logic
    fix_recommendation: str         # "Add dropout (p=0.3–0.5), ..."
    confidence: float               # 0.0–1.0 (how sure are we?)
    affected_epoch: int | None      # Which epoch triggered this (optional)
    metadata: dict                  # Detector-specific data (e.g., gap_size, duration)
```

**Key design:**
- Immutable after creation
- All fields are serializable (for JSON)
- Metadata allows extensibility without changing Finding signature

#### `DiagnosisReport` (Dataclass)

Container for all results:

```python
@dataclass
class DiagnosisReport:
    findings: list[Finding]         # Sorted by severity, deduplicated
    train_loss: np.ndarray          # Original input (unchanged)
    val_loss: np.ndarray            # Original input (unchanged)
    metadata: dict                  # Preprocessed stats, timings, etc.
    processed_at: datetime          # When diagnosis was run
    
    def summary(self) -> str:
        """Return formatted text report."""
    
    def to_dict(self) -> dict:
        """Return JSON-serializable dictionary."""
    
    def to_html(self) -> str:
        """Return standalone HTML report."""
```

**Immutability:** Once created, don't modify. Create new report for new diagnosis.

---

### 2. Preprocessor (`sk_autod/preprocessor.py`)

Normalizes and enriches raw input curves.

#### Input Validation

```python
def validate(train_loss, val_loss) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert to numpy arrays."""
    train = np.array(train_loss, dtype=np.float32)
    val = np.array(val_loss, dtype=np.float32)
    
    # Check: non-empty
    if len(train) == 0:
        raise ValueError("Empty train_loss")
    
    # Check: minimum 5 epochs for meaningful detection
    if len(train) < 5:
        raise ValueError("Need at least 5 epochs of data")
    
    # Check: no NaN or inf
    if np.any(np.isnan(train)) or np.any(np.isinf(train)):
        raise ValueError("NaN or inf values in train_loss")
    if np.any(np.isnan(val)) or np.any(np.isinf(val)):
        raise ValueError("NaN or inf values in val_loss")
    
    return train, val
```

#### Alignment

If arrays have different lengths (rare but possible):

```python
def align(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad shorter array with last value."""
    if len(train) == len(val):
        return train, val
    
    max_len = max(len(train), len(val))
    
    if len(train) < max_len:
        train = np.pad(train, (0, max_len - len(train)), 
                       mode='edge')  # Repeat last value
    
    if len(val) < max_len:
        val = np.pad(val, (0, max_len - len(val)), 
                     mode='edge')
    
    return train, val
```

**Why edge padding?** Maintains semantic meaning (final loss value extends to final epoch).

#### Smoothing (EMA)

Exponential moving average removes noise without lag:

```python
def ema_smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply exponential moving average."""
    ema = np.zeros_like(values)
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    return ema
```

**Why EMA over moving average?**
- Preserves sharp changes (gradient explosions)
- No lag artifacts
- Lightweight computation
- Parameters are interpretable

**Default alpha=0.3:** Balances noise reduction (70% historical weight) vs. responsiveness.

#### Delta Computation

Epoch-to-epoch changes:

```python
def compute_deltas(values: np.ndarray) -> np.ndarray:
    """Compute difference between consecutive epochs."""
    return np.diff(values)  # [v1-v0, v2-v1, ...]
```

**Why?** Detectors often care about trends, not absolute values.

#### Rolling Statistics

Pre-compute aggregates over windows for detector efficiency:

```python
def compute_rolling_stats(values: np.ndarray, window: int = 5) -> dict:
    """Compute mean, std, min, max over rolling windows."""
    return {
        'rolling_mean': np.convolve(values, np.ones(window)/window, mode='valid'),
        'rolling_std': np.array([np.std(values[i:i+window]) 
                                 for i in range(len(values)-window+1)]),
        'rolling_min': np.array([np.min(values[i:i+window]) 
                                 for i in range(len(values)-window+1)]),
        'rolling_max': np.array([np.max(values[i:i+window]) 
                                 for i in range(len(values)-window+1)]),
    }
```

---

### 3. Base Detector (`sk_autod/detectors/base.py`)

Abstract interface for all detectors:

```python
from abc import ABC, abstractmethod
from sk_autod.models import Finding, Severity, DiagnosisReport

class BaseDetector(ABC):
    """Template for all diagnostic detectors."""
    
    # Subclasses must override these
    name: str                          # e.g., "Classic overfitting"
    default_severity: Severity         # Default severity level
    
    @abstractmethod
    def detect(self, report: DiagnosisReport) -> list[Finding]:
        """
        Analyze preprocessed curves and return findings.
        
        Args:
            report: DiagnosisReport with preprocessed data
        
        Returns:
            list[Finding] (0 to N findings, most often 0 or 1)
        
        Contract:
            - Must be deterministic (same input → same output)
            - Must not modify report
            - Must not call external APIs
            - Should complete in <1ms
        """
        pass
```

**Design principles:**
1. **Stateless:** No internal state (all needed data in report)
2. **Pure function:** Same input always gives same output
3. **Isolated:** No side effects, no external dependencies
4. **Fast:** <1ms per detector on typical 100-epoch curve

---

### 4. Detector Implementations

Example detector structure:

```python
# sk_autod/detectors/overfitting.py
import numpy as np
from sk_autod.detectors.base import BaseDetector
from sk_autod.models import Finding, Severity, DiagnosisReport

class OverfittingDetector(BaseDetector):
    """Detect classic overfitting pattern (val↑ while train↓)."""
    
    name = "Classic overfitting"
    default_severity = Severity.CRITICAL
    
    def detect(self, report: DiagnosisReport) -> list[Finding]:
        findings = []
        
        # Get preprocessed data
        train_delta = report.metadata['train_delta']
        val_delta = report.metadata['val_delta']
        
        # Count epochs where val rises but train falls
        overfitting_epochs = []
        for i, (td, vd) in enumerate(zip(train_delta, val_delta)):
            if td < 0 and vd > 0:  # train ↓, val ↑
                overfitting_epochs.append(i + 1)  # Convert to epoch #
        
        # Trigger if 3+ consecutive epochs of divergence
        if self._is_consecutive(overfitting_epochs, min_length=3):
            affected_epoch = overfitting_epochs[0]
            
            # Calculate confidence based on magnitude and duration
            val_delta_subset = val_delta[affected_epoch-1:affected_epoch+3]
            train_delta_subset = train_delta[affected_epoch-1:affected_epoch+3]
            gap = np.mean(val_delta_subset - train_delta_subset)
            
            confidence = min(1.0, (abs(gap) / 0.5) * (len(overfitting_epochs) / 5))
            
            findings.append(Finding(
                detector_name=self.name,
                severity=Severity.CRITICAL,
                signature="Val loss rises while train loss falls",
                logic="Detected 3+ epochs where val_delta > 0 and train_delta < 0",
                fix_recommendation="Add dropout (p=0.3–0.5), L2 regularisation, or reduce model capacity",
                confidence=confidence,
                affected_epoch=affected_epoch,
                metadata={
                    'duration': len(overfitting_epochs),
                    'avg_gap': float(gap),
                },
            ))
        
        return findings
    
    @staticmethod
    def _is_consecutive(indices: list[int], min_length: int = 3) -> bool:
        """Check if indices contain min_length consecutive numbers."""
        if len(indices) < min_length:
            return False
        for i in range(len(indices) - min_length + 1):
            if all(indices[j+1] == indices[j] + 1 
                   for j in range(i, i + min_length - 1)):
                return True
        return False
```

**Key patterns:**
- Use preprocessed data from `report.metadata`
- Multiple confirmation rules (avoid false positives)
- Confidence scoring based on magnitude + duration
- Helpful, actionable fix recommendations
- Metadata for debugging/logging

---

### 5. DiagnosticsRunner (`sk_autod/runner.py`)

Orchestrates detector execution:

```python
class DiagnosticsRunner:
    """Runs all detectors and aggregates findings."""
    
    def __init__(self, detectors: list[BaseDetector] | None = None):
        """Initialize with detector list (default: all 10 detectors)."""
        self.detectors = detectors or self._get_default_detectors()
    
    def diagnose(self, train_loss, val_loss) -> DiagnosisReport:
        """Full diagnosis pipeline."""
        
        # 1. Preprocess
        preprocessor = Preprocessor()
        prep_data = preprocessor.preprocess(train_loss, val_loss)
        
        # 2. Create report container
        report = DiagnosisReport(
            train_loss=prep_data.train_loss,
            val_loss=prep_data.val_loss,
            metadata=prep_data.metadata,  # Contains smoothed + deltas + stats
            findings=[],
        )
        
        # 3. Fire all detectors
        all_findings = []
        for detector in self.detectors:
            findings = detector.detect(report)
            all_findings.extend(findings)
        
        # 4. Deduplicate (resolve conflicts)
        findings = self._deduplicate(all_findings)
        
        # 5. Sort by severity
        findings = sorted(
            findings,
            key=lambda f: self._severity_rank(f.severity),
        )
        
        report.findings = findings
        return report
    
    @staticmethod
    def _deduplicate(findings: list[Finding]) -> list[Finding]:
        """Merge conflicting or redundant findings."""
        # Implementation: see Deduplication Logic section
        pass
    
    @staticmethod
    def _severity_rank(sev: Severity) -> int:
        """Return numeric rank (lower = higher priority)."""
        rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.WARNING: 3,
        }
        return rank[sev]
    
    @staticmethod
    def _get_default_detectors() -> list[BaseDetector]:
        """Return all 10 standard detectors."""
        from sk_autod.detectors import (
            OverfittingDetector, ExplodingGradientDetector,
            LRTooHighDetector, LRTooLowDetector, UnderfittingDetector,
            DyingReluDetector, NoisyTrainingDetector, DataLeakageDetector,
            MissedEarlyStopDetector, LabelNoiseFloorDetector,
        )
        return [
            OverfittingDetector(),
            ExplodingGradientDetector(),
            LRTooHighDetector(),
            LRTooLowDetector(),
            UnderfittingDetector(),
            DyingReluDetector(),
            NoisyTrainingDetector(),
            DataLeakageDetector(),
            MissedEarlyStopDetector(),
            LabelNoiseFloorDetector(),
        ]
```

---

## Detector System

### Detector List & Logic

| # | Name | Severity | Detection Logic | Threshold |
|---|------|----------|-----------------|-----------|
| 1 | Classic overfitting | CRITICAL | Val↑ 3+ epochs while train↓ | 3 consec epochs |
| 2 | Exploding gradient | CRITICAL | Loss spike >300% in 1 epoch | spike_ratio > 3.0 |
| 3 | LR too high | HIGH | High variance + no downtrend | var > 2×baseline |
| 4 | LR too low | MEDIUM | Slow slope (<-0.001/epoch) | slope > -0.001 |
| 5 | Underfitting | HIGH | Both losses plateau high | final_loss > 0.5 |
| 6 | Dying ReLU | HIGH | Flatline early (before ep10) | std < 1e-4, ep < 10 |
| 7 | Noisy training | MEDIUM | >40% sign flips in deltas | flips > 40% |
| 8 | Data leakage | HIGH | train_loss > val_loss (mean) | gap > 0.1 |
| 9 | Missed early stop | WARNING | Val min not in last 10% | position < 90% |
| 10 | Label noise | MEDIUM | Loss floor too high | loss > expected + σ |

### Confidence Scoring

Each finding should estimate confidence (0.0–1.0):

```python
# Pattern: combine magnitude + duration
def compute_confidence(magnitude: float, duration: int) -> float:
    """Confidence = (magnitude / expected) × (duration / ideal)."""
    expected_magnitude = 0.5
    ideal_duration = 5
    
    conf = (magnitude / expected_magnitude) * (duration / ideal_duration)
    return min(1.0, conf)  # Cap at 1.0
```

**Factors that increase confidence:**
- Large magnitude of signal
- Sustained pattern (multiple epochs)
- Multiple confirmation rules (e.g., high variance AND no downtrend)

**Factors that decrease confidence:**
- Noisy data (one-off spike)
- Edge case (very short training)
- Ambiguous pattern (could be multiple issues)

---

## Deduplication Logic

Some detectors can fire on the same underlying issue. Deduplication merges findings:

```python
def _deduplicate(findings: list[Finding]) -> list[Finding]:
    """Merge conflicting/redundant findings into most specific diagnosis."""
    
    if not findings:
        return []
    
    # Group findings by root cause
    cause_groups = {
        'overfitting': ['Classic overfitting'],
        'lr_issue': ['LR too high', 'LR too low', 'Oscillating'],
        'convergence': ['Underfitting', 'Dying ReLU', 'Slow convergence'],
        'quality': ['Noisy training', 'Label noise floor'],
        'leakage': ['Data leakage'],
        'earlyStop': ['Missed early stopping'],
    }
    
    # If multiple findings in same group, keep highest severity
    deduplicated = []
    seen_causes = set()
    
    for finding in sorted(findings, key=lambda f: f.severity.value):
        for cause, detectors in cause_groups.items():
            if finding.detector_name in detectors:
                if cause not in seen_causes:
                    deduplicated.append(finding)
                    seen_causes.add(cause)
                break
    
    return deduplicated
```

**Example:**
```
Input:  ["LR too high", "Oscillating"]
Output: ["LR too high"]  # More specific

Input:  ["Overfitting", "Underfitting"]
Output: ["Overfitting", "Underfitting"]  # Different causes
```

---

## Preprocessor Details

### Full Preprocessing Pipeline

```python
class Preprocessor:
    def __init__(self, ema_alpha: float = 0.3, rolling_window: int = 5):
        self.ema_alpha = ema_alpha
        self.rolling_window = rolling_window
    
    def preprocess(self, train_loss, val_loss) -> dict:
        """Complete preprocessing pipeline."""
        
        # 1. Validate
        train = np.array(train_loss, dtype=np.float32)
        val = np.array(val_loss, dtype=np.float32)
        self._validate(train, val)
        
        # 2. Align (pad if needed)
        train, val = self._align(train, val)
        
        # 3. Smooth (EMA)
        train_smooth = self._ema(train)
        val_smooth = self._ema(val)
        
        # 4. Compute deltas
        train_delta = np.diff(train_smooth)
        val_delta = np.diff(val_smooth)
        
        # 5. Rolling statistics
        stats = {
            'train': self._rolling_stats(train_smooth),
            'val': self._rolling_stats(val_smooth),
            'train_delta': self._rolling_stats(train_delta),
            'val_delta': self._rolling_stats(val_delta),
        }
        
        return {
            'train_loss': train,
            'val_loss': val,
            'train_smooth': train_smooth,
            'val_smooth': val_smooth,
            'train_delta': train_delta,
            'val_delta': val_delta,
            'stats': stats,
            'num_epochs': len(train),
        }
    
    def _validate(self, train, val):
        """Check inputs are valid."""
        assert len(train) >= 5, "Need ≥5 epochs"
        assert len(val) >= 5, "Need ≥5 epochs"
        assert not np.any(np.isnan(train)), "NaN in train"
        assert not np.any(np.isinf(train)), "Inf in train"
        # ... etc for val
    
    def _align(self, train, val):
        """Pad shorter array to match longer."""
        # ... see above
    
    def _ema(self, values):
        """Exponential moving average."""
        # ... see above
    
    def _rolling_stats(self, values, window=None):
        """Compute rolling mean, std, min, max."""
        window = window or self.rolling_window
        return {
            'mean': np.convolve(values, np.ones(window)/window, mode='valid'),
            'std': np.array([np.std(values[i:i+window]) 
                            for i in range(len(values)-window+1)]),
            'min': np.array([np.min(values[i:i+window]) 
                            for i in range(len(values)-window+1)]),
            'max': np.array([np.max(values[i:i+window]) 
                            for i in range(len(values)-window+1)]),
        }
```

---

## Implementation Guidelines

### When Adding a New Detector

1. **Subclass BaseDetector**
   ```python
   class YourDetector(BaseDetector):
       name = "Your Issue"
       default_severity = Severity.HIGH
   ```

2. **Implement detect() with clear logic**
   - Use preprocessed data: `report.metadata`
   - Write multiple confirmation rules
   - Compute confidence (0.0–1.0)
   - Return list of Finding(s)

3. **Write comprehensive tests**
   - Test positive case (should trigger)
   - Test negative case (shouldn't trigger)
   - Test edge cases (very short training, noisy data)
   - Use 5+ fixture curves

4. **Document thresholds**
   ```python
   # Example: LR too low if slope > -0.001 per epoch
   # Rationale: Real learning should show >0.1% loss decrease per epoch
   # Validated on: 50+ real curves, Kaggle competition data
   ```

5. **Register in runner**
   ```python
   # sk_autod/runner.py
   def _get_default_detectors(self):
       return [
           # ... existing detectors ...
           YourDetector(),
       ]
   ```

### Memory-Efficient Design

For large-scale applications:

```python
# ❌ Don't: Store all intermediate arrays
report.all_rolling_means = [...]  # Large memory!

# ✅ Do: Store only what detectors need
report.metadata['rolling_stats'] = {
    'mean': np.ndarray,
    'std': np.ndarray,
}

# ✅ Do: Compute on-the-fly if needed
rolling_mean = np.convolve(values, kernel, mode='valid')
```

### Thread Safety

Current design is thread-safe:

```python
# ✅ Safe: Multiple threads can run this
runner = DiagnosticsRunner()
report1 = runner.diagnose(train1, val1)  # Thread 1
report2 = runner.diagnose(train2, val2)  # Thread 2
```

**Why:** Detectors are stateless (all data comes from `report` parameter).

---

## Performance Considerations

### Target Benchmarks

| Operation | Target | Notes |
|-----------|--------|-------|
| Preprocess 100-epoch curve | <0.5ms | NumPy ops only |
| Run 1 detector | <0.1ms | Simple heuristics |
| Run 10 detectors (parallel) | <1ms | Expected linear scaling |
| Full diagnosis (preprocess + all detectors + format) | <2ms | Target latency |

### Optimization Checklist

- [ ] Use NumPy vectorization (no Python loops on arrays)
- [ ] Avoid creating temporary arrays (use in-place ops where possible)
- [ ] Precompute stats (rolling mean, std, etc.)
- [ ] Lazy evaluation for expensive operations
- [ ] Cache results if diagnosis run multiple times

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

report = diagnose(train, val)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 slowest functions
```

---

## Testing Strategy

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_preprocessor.py
├── test_detectors/
│   ├── test_base.py
│   ├── test_overfitting.py
│   ├── test_gradient.py
│   └── ... (one per detector)
├── test_runner.py
├── test_formatters.py
└── fixtures.py              # Curve library
```

### Fixture Curves

Create realistic test curves:

```python
# tests/fixtures.py
class Curves:
    @staticmethod
    def well_trained(length=100, noise=0.01):
        """Smooth convergence."""
        epochs = np.arange(length)
        loss = 2.3 * np.exp(-0.02 * epochs) + noise * np.random.randn(length)
        return loss
    
    @staticmethod
    def overfitting(length=100):
        """Train falls, val rises."""
        epochs = np.arange(length)
        train = 2.3 * np.exp(-0.03 * epochs) + 0.01
        val = 2.4 * np.exp(-0.02 * epochs)
        val[int(length * 0.4):] += 0.1 * (epochs[int(length * 0.4):] - int(length * 0.4))
        return train, val
    
    # ... more fixtures
```

### Test Detectors

```python
def test_overfitting_detector_positive():
    """Should detect classic overfitting pattern."""
    detector = OverfittingDetector()
    train, val = Curves.overfitting()
    
    findings = detector.detect(train, val)
    
    assert len(findings) == 1
    assert findings[0].severity == Severity.CRITICAL
    assert findings[0].confidence > 0.8

def test_overfitting_detector_negative():
    """Should NOT detect overfitting in well-trained curve."""
    detector = OverfittingDetector()
    train, val = Curves.well_trained()
    
    findings = detector.detect(train, val)
    
    assert len(findings) == 0
```

---

## API Contracts

### User-Facing API

```python
# Public function: diagnose()
def diagnose(
    train_loss: list[float] | np.ndarray,
    val_loss: list[float] | np.ndarray,
    detectors: list[BaseDetector] | None = None,
) -> DiagnosisReport:
    """
    Diagnose training curves for pathologies.
    
    Args:
        train_loss: Training loss per epoch (≥5 values)
        val_loss: Validation loss per epoch (same length)
        detectors: Custom detectors (default: all 10)
    
    Returns:
        DiagnosisReport with ranked findings
    
    Raises:
        ValueError: If input invalid
    """
    runner = DiagnosticsRunner(detectors)
    return runner.diagnose(train_loss, val_loss)
```

### Internal Contracts

**Detector contract:**
```python
# Detector.detect(report) must:
# - Not modify report
# - Not access external APIs
# - Return list[Finding] (may be empty)
# - Complete in <1ms
# - Be deterministic
```

**Preprocessor contract:**
```python
# Preprocessor.preprocess(train, val) must:
# - Validate inputs
# - Return dict with keys:
#   - train_loss, val_loss (original)
#   - train_smooth, val_smooth
#   - train_delta, val_delta
#   - stats (rolling statistics)
# - Raise ValueError if invalid
```

---

## Common Extension Points

### Custom Detectors

```python
from sk_autod import BaseDetector, Finding, Severity

class MyDetector(BaseDetector):
    name = "My Custom Issue"
    default_severity = Severity.MEDIUM
    
    def detect(self, report):
        # Your logic
        return []

# Use it
from sk_autod import diagnose
report = diagnose(train, val, detectors=[MyDetector()])
```

### Custom Formatters (v0.2+)

```python
class MyFormatter:
    def format(self, report: DiagnosisReport) -> str:
        # Custom output format
        return "Custom output"
```

### Framework Callbacks (v0.3+)

```python
class AutoDCallback:
    def on_epoch_end(self, epoch, train_loss, val_loss):
        report = diagnose(train_loss, val_loss)
        # Act on findings
```

---

## Stability & Versioning

### API Stability

**v0.1.0:** Experimental, APIs may change  
**v1.0.0:** Stable, semantic versioning (MAJOR.MINOR.PATCH)

Breaking changes only in major versions.

### Threshold Changes

When tuning detector thresholds:
- Update threshold in detector code
- Add comment with rationale
- Re-run all tests (should still pass)
- Update CHANGELOG

---

## Debugging Tips

### Add Logging

```python
import logging

logger = logging.getLogger('sk_autod')
logger.debug(f"Preprocessing {len(train)} epochs")
logger.debug(f"Detector {detector.name} found {len(findings)} issues")
```

### Inspect Preprocessed Data

```python
report = diagnose(train, val)
print(report.metadata)
# → Check smooth curves, deltas, stats
```

### Trace Detector Execution

```python
detector = MyDetector()
findings = detector.detect(report)
print(f"Findings: {len(findings)}")
for f in findings:
    print(f"  - {f.detector_name} ({f.confidence:.1%})")
    print(f"    Metadata: {f.metadata}")
```

---

## References

- **NumPy docs:** https://numpy.org/doc/
- **Dataclasses:** https://docs.python.org/3/library/dataclasses.html
- **ABC (Abstract Base Classes):** https://docs.python.org/3/library/abc.html

---

**Questions?** Open an issue or discuss in Discussions tab.
