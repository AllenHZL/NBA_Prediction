from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    threshold: float

