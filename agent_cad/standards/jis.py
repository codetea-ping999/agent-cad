"""JIS / ISO tolerance tables used by drawings and checks.

- JIS B 0405 (= ISO 2768-1) general tolerances for linear/angular dimensions
- JIS B 0401 (= ISO 286) limits and fits, sizes 3..120 mm, common grades
- JIS B 0031 (= ISO 1302) surface texture defaults used in this project
"""
import math

# ---------------------------------------------------------------- JIS B 0405
# class -> list of (upper size bound mm, +/- tolerance mm); 0.5 mm < size
_B0405_LINEAR = {
    "f": [(3, 0.05), (6, 0.05), (30, 0.1), (120, 0.15), (400, 0.2), (1000, 0.3), (2000, 0.5)],
    "m": [(3, 0.1), (6, 0.1), (30, 0.2), (120, 0.3), (400, 0.5), (1000, 0.8), (2000, 1.2)],
    "c": [(3, 0.2), (6, 0.3), (30, 0.5), (120, 0.8), (400, 1.2), (1000, 2.0), (2000, 3.0)],
    "v": [(3, None), (6, 0.5), (30, 1.0), (120, 1.5), (400, 2.5), (1000, 4.0), (2000, 6.0)],
}
_B0405_RADIUS = {  # radii and chamfer heights
    "f": [(3, 0.2), (6, 0.5), (math.inf, 1.0)], "m": [(3, 0.2), (6, 0.5), (math.inf, 1.0)],
    "c": [(3, 0.4), (6, 1.0), (math.inf, 2.0)], "v": [(3, 0.4), (6, 1.0), (math.inf, 2.0)],
}
# angular: length of the shorter side -> tolerance in degrees
_B0405_ANGLE = {
    "f": [(10, 1.0), (50, 0.5), (120, 1 / 3), (400, 1 / 6), (math.inf, 1 / 12)],
    "m": [(10, 1.0), (50, 0.5), (120, 1 / 3), (400, 1 / 6), (math.inf, 1 / 12)],
    "c": [(10, 1.5), (50, 1.0), (120, 0.5), (400, 0.25), (math.inf, 1 / 6)],
    "v": [(10, 3.0), (50, 2.0), (120, 1.0), (400, 0.5), (math.inf, 1 / 3)],
}


def _lookup(table, x):
    for ub, t in table:
        if x <= ub:
            return t
    raise ValueError(f"size {x} out of table range")


def general_linear(size, cls="m"):
    if size < 0.5:
        raise ValueError("JIS B 0405 linear tolerances start at 0.5 mm")
    return _lookup(_B0405_LINEAR[cls], size)


def general_radius(size, cls="m"):
    return _lookup(_B0405_RADIUS[cls], size)


def general_angle_deg(shorter_side, cls="m"):
    return _lookup(_B0405_ANGLE[cls], shorter_side)


def b0405_table(cls="m"):
    """Rows for the general-tolerance table printed on drawings."""
    lo = [0.5, 3, 6, 30, 120, 400, 1000]
    return [(a, ub, t) for a, (ub, t) in zip(lo, _B0405_LINEAR[cls]) if ub <= 400]


# ---------------------------------------------------------------- JIS B 0401 / ISO 286
_STEPS = [3, 6, 10, 18, 30, 50, 80, 120]  # upper bounds of nominal size steps
# standard tolerance grades, micrometres, per size step (3,6,10,18,30,50,80,120]
_IT = {
    5: [4, 5, 6, 8, 9, 11, 13, 15],
    6: [6, 8, 9, 11, 13, 16, 19, 22],
    7: [10, 12, 15, 18, 21, 25, 30, 35],
    8: [14, 18, 22, 27, 33, 39, 46, 54],
    9: [25, 30, 36, 43, 52, 62, 74, 87],
    10: [40, 48, 58, 70, 84, 100, 120, 140],
    11: [60, 75, 90, 110, 130, 160, 190, 220],
}
# shaft fundamental deviations (upper deviation es for a..h, lower ei for k..), micrometres
_SHAFT_ES = {
    "d": [-20, -30, -40, -50, -65, -80, -100, -120],
    "e": [-14, -20, -25, -32, -40, -50, -60, -72],
    "f": [-6, -10, -13, -16, -20, -25, -30, -36],
    "g": [-2, -4, -5, -6, -7, -9, -10, -12],
    "h": [0] * 8,
}
_SHAFT_EI = {
    "k": [0, 1, 1, 1, 2, 2, 2, 3],   # grades 4..7
    "m": [2, 4, 6, 7, 8, 9, 11, 13],
    "n": [4, 8, 10, 12, 15, 17, 20, 23],
    "p": [6, 12, 15, 18, 22, 26, 32, 37],
}


def _step(size):
    if not 0 < size <= _STEPS[-1]:
        raise ValueError("ISO 286 table here covers 0..120 mm")
    for i, ub in enumerate(_STEPS):
        if size <= ub:
            return i


def it_grade(size, grade):
    return _IT[grade][_step(size)] / 1000.0


def fit_deviations(size, code):
    """'F8', 'H7', 'h6', 'js7'... -> (upper, lower) deviation in mm."""
    letter, grade = code.rstrip("0123456789"), int(code[len(code.rstrip("0123456789")):])
    i, it = _step(size), _IT[grade][_step(size)]
    low = letter.lower()
    if low == "js":
        return it / 2000.0, -it / 2000.0
    if letter.islower():  # shaft
        if low in _SHAFT_ES:
            es = _SHAFT_ES[low][i]; return es / 1000.0, (es - it) / 1000.0
        ei = _SHAFT_EI[low][i]; return (ei + it) / 1000.0, ei / 1000.0
    # hole: A..H mirror of the shaft (EI = -es), valid for the letters provided
    if low in _SHAFT_ES:
        EI = -_SHAFT_ES[low][i]; return (EI + it) / 1000.0, EI / 1000.0
    raise ValueError(f"hole deviation {code} not tabulated")


def fmt_dev(x):
    if abs(x) < 1e-9:
        return "0"
    return f"{x:+.3f}".rstrip("0").rstrip(".") if abs(x * 1000 - round(x * 1000)) > 1e-6 else f"{x:+.3f}"


# ---------------------------------------------------------------- surface texture (JIS B 0031)
RA_SERIES = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5, 25]
