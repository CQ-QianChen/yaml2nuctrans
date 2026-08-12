from dataclasses import dataclass

@dataclass(frozen=True)
class PressureSetup:
    pressure_initial: float
    pressure_top: float
    pressure_bottom: float