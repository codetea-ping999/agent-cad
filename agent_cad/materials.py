"""Mechanical properties for strength checks.

Typical handbook values for the heat-treated condition (ASM / Aluminum Association
data for the aluminium alloys, annealed-and-aged bar for Ti-6Al-4V). They are not
guaranteed minimums: a production release should replace them with the supplier's
certified values or the specification minimums for the forged product (JIS H 4140).
Fatigue strength is the rotating-beam value at 5e8 cycles (Ti: 1e7), R = -1.
cte_per_K is the mean linear expansion coefficient 20 C -> forging temperature
(used for the die-cavity shrink allowance).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    E_MPa: float
    nu: float
    yield_MPa: float
    tensile_MPa: float
    fatigue_MPa: float
    density_g_cm3: float
    cte_per_K: float


MATERIALS = {
    "A6061-T6": Material(68_900, 0.33, 276, 310, 96.5, 2.70, 25.2e-6),
    "A7075-T6": Material(71_700, 0.33, 503, 572, 159, 2.81, 25.2e-6),
    "A2014-T6": Material(72_400, 0.33, 414, 483, 124, 2.80, 24.7e-6),
    "Ti-6Al-4V": Material(113_800, 0.342, 880, 950, 510, 4.43, 9.7e-6),
}
