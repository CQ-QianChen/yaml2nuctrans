from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BoundaryCondition:
    type: Literal["Dirichlet", "Neumann", "Python"]
    mesh: str
    value: float | str


@dataclass(frozen=True)
class BoundaryConditions:
    conditions: dict[str, BoundaryCondition]