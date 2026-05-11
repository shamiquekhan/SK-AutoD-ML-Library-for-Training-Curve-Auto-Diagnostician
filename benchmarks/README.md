# Benchmarks

Performance benchmarks for SK-AutoD.

## Running Benchmarks

```bash
python benchmarks/benchmark_performance.py
```

## Expected Results

On a typical machine (Python 3.11, 2024 hardware):

| Curve Length | Time per Diagnosis |
|---|---|
| 10 epochs | ~0.1 ms |
| 100 epochs | ~0.2 ms |
| 1000 epochs | ~1.0 ms |

| Batch Size | Total Time |
|---|---|
| 100 curves (100 epochs each) | ~20 ms |
| 1000 curves (100 epochs each) | ~200 ms |

## Memory Usage

- ~2 KB per 1000-epoch diagnosis
- Pure Python implementation (no numpy required)

## Benchmark Script

See `benchmark_performance.py` for the full benchmark suite.
