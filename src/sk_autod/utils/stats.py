from __future__ import annotations

from math import sqrt


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    center = mean(values)
    return sqrt(sum((value - center) ** 2 for value in values) / len(values))


def diff(values: list[float]) -> list[float]:
    return [current - previous for previous, current in zip(values, values[1:])]


def ema(values: list[float], alpha: float = 0.3) -> list[float]:
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return smoothed


def moving_average(values: list[float], window: int = 3) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)

    averaged: list[float] = []
    for index in range(len(values) - window + 1):
        segment = values[index : index + window]
        averaged.append(mean(segment))
    return averaged
