"""Verification loop for the stem: every parameter set is rebuilt and checked.

    report = verify(StemParams(length=110), fea=True)

Each rule records its value, its limit and its basis, so a failure says what to
change. Geometry and walls are measured on the B-rep; strength comes from a linear
FEA of the body (steerer bore fixed, force on the bar bore).

Load cases: the magnitudes below are PROVISIONAL design loads in the spirit of the
ISO 4210-5 stem tests (static forward bending, lateral bending, fatigue). The
standard's exact forces, lever arms and cycle counts must be taken from the purchased
text and entered here before a design is released; until then the report marks them
as provisional.
"""
from dataclasses import dataclass, asdict
import numpy as np
from ..parts import stem
from ..materials import MATERIALS
from .wall import min_wall

MIN_WALL_MM = 1.5            # forged + machined aluminium: thinnest wall we accept
THREAD_ENGAGEMENT_D = 1.5    # internal thread in aluminium: engagement >= 1.5 x d
SF_STATIC = 1.2              # on yield, static load cases
SF_FATIGUE = 1.5             # on fatigue strength, fatigue load case
BOLT_D = 5.0                 # M5 throughout


@dataclass(frozen=True)
class LoadCase:
    name: str
    force_N: tuple           # global axes: X forward (stem direction at 0 deg), Y lateral, Z up
    kind: str                # "static" or "fatigue"
    provisional: bool = True


LOAD_CASES = (
    LoadCase("forward_bending_static", (0.0, 0.0, -1600.0), "static"),
    LoadCase("lateral_bending_static", (0.0, 1000.0, 0.0), "static"),
    LoadCase("forward_bending_fatigue", (0.0, 0.0, -450.0), "fatigue"),
)


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def body_fea(p, body, cases=LOAD_CASES, h=2.0):
    """One mesh, one solve per load case."""
    from ..fea.static import mesh_shape, solve
    d, n, S, B = stem.frame(p)
    Bv = np.array([B.X, B.Y, B.Z])
    mesh = mesh_shape(body, h)
    rs, rb = p.steerer_d / 2, p.bar_d / 2

    def fixed(c):      # steerer bore (the steerer tube holds the stem)
        return (np.abs(np.hypot(c[0], c[1]) - rs) < 0.3) & (c[2] > -0.1) & (c[2] < p.stack_h + 0.1)

    def bar(c):        # bar bore (the handlebar pushes on it)
        rel = c - Bv[:, None]
        return np.abs(np.hypot(rel[0], rel[2]) - rb) < 0.3
    mat = MATERIALS[p.material]
    return {lc.name: solve(mesh, mat, fixed, [(bar, lc.force_N)]) for lc in cases}


def verify(p: stem.StemParams, fea=True, h=2.0, cases=LOAD_CASES):
    cases = tuple(cases)                        # iterated twice: the solves, then the rules
    names = [lc.name for lc in cases]
    if len(set(names)) != len(names):           # results are keyed by name
        raise ValueError(f"duplicate load case names: {names}")
    b, f = stem.build(p)
    g = stem.checks(p, b, f)
    mat = MATERIALS[p.material]
    rules = [
        _rule("geometry.valid_solids", g["body_valid"] and g["faceplate_valid"], True, True, "B-rep check, one solid each"),
        *[_rule(f"geometry.interference.{k}", v == 0, v, 0, "mm3 overlap with mating part")
          for k, v in g["interference_mm3"].items()],
    ]
    walls = {"body": min_wall(b), "faceplate": min_wall(f)}
    for k, w in walls.items():
        t = w["min_mm"] if w else 0.0
        rules.append(_rule(f"wall.{k}", t >= MIN_WALL_MM - 1e-3, t, MIN_WALL_MM,
                           "measured on the B-rep (ray along inward normal)", at_mm=w and w["at_mm"]))
    for k, v in g["thread_engagement_mm"].items():
        rules.append(_rule(f"thread.{k}", v >= THREAD_ENGAGEMENT_D * BOLT_D, v, THREAD_ENGAGEMENT_D * BOLT_D,
                           f"M{BOLT_D:g} in aluminium: engagement >= {THREAD_ENGAGEMENT_D} d"))
    rules.append(_rule("thread.tap_drill_M5", g["tap_drill_fits_M5"], p.tap_drill_d, list(stem.M5_DRILL),
                       "tap drill within the M5 minor-diameter band"))
    m = g["mass_g"]["total_excl_bolts"]
    rules.append(_rule("mass.total_g", m <= p.mass_target_g, m, p.mass_target_g, "parameter mass_target_g"))
    fea_out = {}
    if fea:
        res = body_fea(p, b, cases, h)
        for lc in cases:
            r = res[lc.name]
            allow = mat.yield_MPa / SF_STATIC if lc.kind == "static" else mat.fatigue_MPa / SF_FATIGUE
            basis = (f"yield {mat.yield_MPa} MPa / {SF_STATIC}" if lc.kind == "static"
                     else f"fatigue {mat.fatigue_MPa} MPa / {SF_FATIGUE}")
            rules.append(_rule(f"strength.{lc.name}", r.vm_p999_MPa <= allow, r.vm_p999_MPa, round(allow, 1),
                               basis + " (von Mises, 99.9th percentile)", provisional_load=lc.provisional,
                               load_N=lc.force_N, at_mm=r.vm_at_mm, max_disp_mm=r.max_disp_mm))
            fea_out[lc.name] = asdict(r)
    return {"params": stem.asdict(p), "material": asdict(mat), "rules": rules, "walls": walls,
            "fea": fea_out, "mass_g": g["mass_g"],
            "pass": all(r["pass"] for r in rules),
            "failed": [r["check"] for r in rules if not r["pass"]]}
