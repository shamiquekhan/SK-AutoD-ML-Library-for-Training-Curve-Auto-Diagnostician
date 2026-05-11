# Architecture

This page points to the full technical design in the repository root.

- [Root architecture guide](../ARCHITECTURE.md)

## High-level flow

1. Validate and align the input curves
2. Smooth and compute curve statistics
3. Run detectors
4. Deduplicate overlapping findings
5. Sort by severity
6. Format the report as text, JSON, or HTML
