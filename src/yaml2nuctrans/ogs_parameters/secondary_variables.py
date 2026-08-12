from dataclasses import dataclass


@dataclass(frozen=True)
class SecondaryVars:
    vars: dict[str, str]