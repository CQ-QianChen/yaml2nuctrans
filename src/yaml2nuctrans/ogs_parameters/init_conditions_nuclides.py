from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InitialCondition:
    type: Literal["Constant", "Function"]
    value: float | str


@dataclass(frozen=True)
class InitialConditions:
    conditions: dict[str, InitialCondition]