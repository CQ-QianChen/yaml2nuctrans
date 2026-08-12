from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessCoupling:
    decay: bool
    diffusion: bool
    advection: bool
    sorption: bool
    # thermal: bool