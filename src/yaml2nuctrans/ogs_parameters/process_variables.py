from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessVars:
    vars: dict[str, str | list[str]]