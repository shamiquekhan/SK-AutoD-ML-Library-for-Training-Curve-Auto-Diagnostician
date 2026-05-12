from __future__ import annotations

from abc import ABC, abstractmethod

from sk_autod.core.preprocessing import PreprocessedCurves
from sk_autod.core.models import Finding


class BaseDetector(ABC):
    name = "BaseDetector"

    @abstractmethod
    def detect(self, curves: PreprocessedCurves) -> list[Finding]:
        raise NotImplementedError
