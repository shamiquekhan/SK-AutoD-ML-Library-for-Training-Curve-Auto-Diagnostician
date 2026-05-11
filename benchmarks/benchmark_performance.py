#!/usr/bin/env python
"""Performance benchmarks for SK-AutoD.

Run with: python benchmarks/benchmark_performance.py
"""

import time
from statistics import mean, stdev

from sk_autod import diagnose


def generate_curve(length: int, pattern: str = "converging") -> tuple[list[float], list[float]]:
    """Generate synthetic training curves for benchmarking."""
    if pattern == "converging":
        train = [2.5 * (0.95**i) + 0.01 for i in range(length)]
        val = [2.6 * (0.94**i) + 0.02 for i in range(length)]
    elif pattern == "overfitting":
        train = [2.5 * (0.9**i) for i in range(length)]
        val = [2.6 * (0.9**i) for i in range(length)]
        # Make val rise after 40% of epochs
        pivot = int(length * 0.4)
        for i in range(pivot, length):
            val[i] = val[pivot - 1] + 0.05 * (i - pivot)
    elif pattern == "noisy":
        import random
        train = [2.5 * (0.95**i) + random.uniform(-0.1, 0.1) for i in range(length)]
        val = [2.6 * (0.94**i) + random.uniform(-0.1, 0.1) for i in range(length)]
    else:
        train = [2.5 - i * 0.02 for i in range(length)]
        val = [2.6 - i * 0.015 for i in range(length)]
    return train, val


def benchmark_single_diagnosis(length: int, iterations: int = 100) -> dict:
    """Benchmark single curve diagnosis."""
    train, val = generate_curve(length, "overfitting")
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        diagnose(train, val)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        "curve_length": length,
        "iterations": iterations,
        "mean_ms": mean(times),
        "std_ms": stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_batch_diagnosis(count: int, length: int) -> dict:
    """Benchmark diagnosing multiple curves."""
    curves = [generate_curve(length, "converging") for _ in range(count)]
    
    start = time.perf_counter()
    for train, val in curves:
        diagnose(train, val)
    elapsed = time.perf_counter() - start
    
    return {
        "curve_count": count,
        "curve_length": length,
        "total_ms": elapsed * 1000,
        "per_curve_ms": (elapsed * 1000) / count,
    }


def main() -> None:
    print("=" * 60)
    print("SK-AutoD Performance Benchmarks")
    print("=" * 60)
    
    # Single curve benchmarks
    print("\n📊 Single Curve Diagnosis")
    print("-" * 40)
    
    for length in [10, 50, 100, 500, 1000]:
        result = benchmark_single_diagnosis(length, iterations=50)
        print(f"  {length:4d} epochs: {result['mean_ms']:.3f} ms ± {result['std_ms']:.3f} ms")
    
    # Batch benchmarks
    print("\n📊 Batch Diagnosis (100 epochs each)")
    print("-" * 40)
    
    for count in [10, 100, 500, 1000]:
        result = benchmark_batch_diagnosis(count, 100)
        print(f"  {count:4d} curves: {result['total_ms']:.1f} ms total ({result['per_curve_ms']:.3f} ms/curve)")
    
    # Memory estimate
    print("\n📊 Memory Usage Estimate")
    print("-" * 40)
    train, val = generate_curve(1000, "overfitting")
    report = diagnose(train, val)
    import sys
    mem_bytes = sys.getsizeof(report.findings) + sum(
        sys.getsizeof(f.__dict__) for f in report.findings
    )
    print(f"  ~{mem_bytes / 1024:.1f} KB per 1000-epoch diagnosis")
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
