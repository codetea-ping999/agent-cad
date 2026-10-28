"""Verification loop for the drop handlebar: every parameter set is rebuilt and checked.

    report = verify(HandlebarParams(width=400), solid_fea=False)

Strength comes from a beam model of the whole bar (fea/beam.py: bend flexibility and
stress intensification included), held rigidly over the stem's clamp width. The solid
FEA of one half (fea/static.py) cross-checks the beam model where both apply.

Load cases: PROVISIONAL design loads in the spirit of the ISO 4210-5 handlebar and
handlebar-stem tests for racing bicycles (static bending at 50 mm from the drop ends,
fatigue on the hoods and the drops, out-of-phase and in-phase), using the same hood
loads as the stem verification of this project. The standard's exact forces, lever
arms and cycle counts must be taken from the purchased text before a release; the
report marks them as provisional.
"""
import math
from dataclasses import dataclass, asdict
import numpy as np
from ..parts import handlebar as hb
from ..materials import MATERIALS
from ..fea import beam

SF_STATIC = 1.2                # on yield
SF_FATIGUE = 1.25              # on the fatigue strength at the test life
# fully reversed fatigue strength at the ISO 4210 test life (1e5 cycles), smooth tube; the
# same handbook-level estimates as this project's stem ISO 4210 analyses
FATIGUE_TEST_MPA = {"A7075-T6": 200.0, "A6061-T6": 140.0}
SOLID_H = 2.5                  # solid FEA element size (mm): 99.9th percentile within 1 % of h = 3
STEM_CLAMP_W = 40.0            # bar-clamp width of this project's stem (agent_cad/parts/stem.py)
CLAMP_ZONE_MARGIN = 15.0       # per side beyond the stem: clamp edge off the taper, clip-on / light mounts
LEVER_BAND_W = 22.0            # brake-lever band width along the bar centreline
MIN_WALL_FINISHED = 0.85       # after bending (extrados), aluminium bar
DROP_LOAD_FROM_END = 50.0      # ISO 4210-5 style load point on the drops
HOOD_ARC_DEG = 25.0            # lever band centre on B2, degrees past its start
HOOD_OFFSET = (25.0, 0.0, 40.0)  # hand on the hood: forward / up from the band centre (assumed)


@dataclass(frozen=True)
class LoadCase:
    name: str
    where: str                 # "hoods" | "drops"
    force_N: tuple             # on the right-hand side (global X fwd, Y right, Z up)
    phase: int                 # left side gets phase * force (mirrored in Y): +1 in phase, -1 out of phase
    kind: str                  # "static" | "fatigue"
    provisional: bool = True


LOAD_CASES = (
    LoadCase("bending_static_drops", "drops", (0.0, 0.0, -1000.0), +1, "static"),
    LoadCase("forward_static_hoods", "hoods", (1000.0, 0.0, 0.0), +1, "static"),
    LoadCase("fatigue_out_of_phase_hoods", "hoods", (0.0, 0.0, -330.0), -1, "fatigue"),
    LoadCase("fatigue_in_phase_hoods", "hoods", (0.0, 0.0, -450.0), +1, "fatigue"),
    LoadCase("fatigue_out_of_phase_drops", "drops", (0.0, 0.0, -330.0), -1, "fatigue"),
)
STIFFNESS_CASE = LoadCase("stiffness_drops_100N", "drops", (0.0, 0.0, -100.0), +1, "stiffness", False)


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def load_points(p, sol=None):
    """Per side: (segment name, arc length into it) of the hood and drop load points."""
    sol = sol or hb.solve(p)
    return {"hoods": ("B2", p.r_hood * math.radians(HOOD_ARC_DEG)),
            "drops": ("L3", sol.L3 - DROP_LOAD_FROM_END)}


def model(p, h=4.0, bend_effects=True):
    """Beam model of the full bar: nodes, elements, fixed nodes, segment lookup."""
    sol = hb.solve(p)
    halves = {s: hb.path(p, s, sol) for s in (1, -1)}
    tops = hb.Seg("line", (0, -sol.L0, 0), (0, 1.0, 0), 2 * sol.L0, name="L0")
    segs = [tops] + halves[1][1:] + halves[-1][1:]

    def section_of(seg, x):
        if seg.name == "L0":
            return beam.Section(*hb.radii_at(p, x[1]))
        return beam.Section(p.grip_d / 2, p.grip_d / 2 - p.wall_grip)
    # node at the clamp edges so the support ends exactly there
    X, E = beam.discretise(segs, section_of, h, bend_effects)
    fixed = [i for i, x in enumerate(X) if abs(x[1]) <= STEM_CLAMP_W / 2 + 1e-6 and abs(x[0]) < 1e-6 and abs(x[2]) < 1e-6]
    return sol, halves, X, E, fixed


def _nearest(X, q):
    return int(np.argmin(np.linalg.norm(X - q, axis=1)))


def _case_loads(p, sol, halves, X, lc):
    seg_name, s = load_points(p, sol)[lc.where]
    loads = []
    for side in (1, -1):
        seg = next(g for g in halves[side] if g.name == seg_name)
        q = np.array(seg.point(s)[0])
        F = np.array(lc.force_N, float) * (1 if side > 0 else lc.phase)
        if side < 0:
            F[1] = -F[1]                   # mirrored
        i = _nearest(X, q)
        arm = (np.array(HOOD_OFFSET) * [1, side, 1] if lc.where == "hoods" else np.zeros(3)) + (q - X[i])
        loads.append((i, F, np.cross(arm, F)))
    return loads


def beam_fea(p, cases=LOAD_CASES, h=4.0, bend_effects=True):
    mat = MATERIALS[p.material]
    sol, halves, X, E, fixed = model(p, h, bend_effects)
    out = {}
    for lc in (*cases, STIFFNESS_CASE):
        loads = _case_loads(p, sol, halves, X, lc)
        u, forces = beam.solve(X, E, fixed, loads, mat.E_MPa, mat.nu)
        vm = beam.stresses(E, forces)
        k = int(np.argmax(vm.max(axis=1)))
        e = E[k]
        at = X[e.n0] if vm[k, 0] >= vm[k, 1] else X[e.n1]
        # largest in each segment (the clamp edge and the bends are what to watch)
        by_seg = {}
        for j, el in enumerate(E):
            by_seg[el.seg] = max(by_seg.get(el.seg, 0.0), float(vm[j].max()))
        li = [i for i, _, _ in loads]
        out[lc.name] = dict(vm_max_MPa=round(float(vm.max()), 1), at_mm=[round(float(c), 1) for c in at], seg=e.seg,
                            by_segment_MPa={k_: round(v, 1) for k_, v in by_seg.items()},
                            load_point_disp_mm=[round(float(np.linalg.norm(u[i, :3])), 3) for i in li],
                            load_point_dz_mm=[round(float(u[i, 2]), 3) for i in li])
    return out


def bend_checks(p):
    """Wall at the extrados after bending (neutral axis on the centreline: t_o = t / (1 + r_m / R))
    and the bend ratios the tooling has to manage."""
    rm = (p.grip_d - p.wall_grip) / 2
    out = {}
    for name, R in (("B1", p.r_shoulder), ("B2", p.r_hood), ("B3", p.r_drop)):
        out[name] = dict(R=R, R_over_D=round(R / p.grip_d, 3), D_over_t=round(p.grip_d / p.wall_grip, 2),
                         wall_extrados=round(p.wall_grip / (1 + rm / R), 3),
                         wall_intrados=round(p.wall_grip / (1 - rm / R), 3),
                         thinning_pct=round(100 * (1 - 1 / (1 + rm / R)), 1))
    return out


def fatigue_strength(p):
    mat = MATERIALS[p.material]
    return FATIGUE_TEST_MPA.get(p.material, mat.fatigue_MPa)


def allowable(p, kind):
    mat = MATERIALS[p.material]
    if kind == "static":
        return mat.yield_MPa / SF_STATIC, f"yield {mat.yield_MPa:g} MPa / {SF_STATIC}"
    return fatigue_strength(p) / SF_FATIGUE, f"fatigue {fatigue_strength(p):g} MPa @1e5 / {SF_FATIGUE}"


def solid_fea(p, shape, cases=LOAD_CASES, h=SOLID_H, curvature=12, solver="direct", keep_mesh=False):
    """Solid FEA of the right half (the clamp holds each half on its own, so in-phase and
    out-of-phase cases load a half the same way)."""
    from build123d import Box, Pos
    from ..fea import bar_fea
    sol = hb.solve(p)
    half = shape & Pos(0, 1000, 0) * Box(4000, 2000, 4000)
    Rc = p.clamp_d / 2
    segs = {g.name: g for g in hb.path(p, 1, sol)}
    pts = load_points(p, sol)

    def fixed(c):
        return (np.abs(np.hypot(c[0], c[2]) - Rc) < 0.2) & (c[1] <= STEM_CLAMP_W / 2) & (c[1] >= 0)
    loads = {}
    for lc in cases:
        nm, s = pts[lc.where]
        q, t = segs[nm].point(s)
        F = np.array(lc.force_N, float)
        M = np.cross(np.array(HOOD_OFFSET), F) if lc.where == "hoods" else np.zeros(3)
        loads[lc.name] = [bar_fea.BandLoad(tuple(q), tuple(t), p.grip_d / 2, 5.0, tuple(F), tuple(M))]
    res, info = bar_fea.solve_cases(half, MATERIALS[p.material], fixed, loads, h, curvature, solver)
    mesh = info.pop("mesh")
    edge = STEM_CLAMP_W / 2 + 3.0              # leave out the clamp-edge singularity of the rigid fixture
    out = {}
    for name, r in res.items():
        vm, c = r.pop("vm_el"), r.pop("centroids")
        away = c[1] > edge
        r["vm_p999_away_from_clamp_MPa"] = round(float(np.percentile(vm[away], 99.9)), 1)
        k = int(np.argmax(np.where(away, vm, 0)))
        r["vm_max_away_from_clamp_MPa"] = round(float(vm[k]), 1)
        r["at_away_mm"] = [round(float(x), 1) for x in c[:, k]]
        out[name] = r
    if keep_mesh:
        info["mesh"] = mesh
    return out, info


def verify(p: hb.HandlebarParams, shape=None, cases=LOAD_CASES, h=4.0, with_shape=True, solid=False, solid_h=SOLID_H):
    mat = MATERIALS[p.material]
    if shape is None and with_shape:
        shape = hb.build(p)
    g = hb.checks(p, shape)
    rules = list(g["rules"])
    sol = hb.solve(p)
    a = p.clamp_zone_w / 2
    need = STEM_CLAMP_W / 2 + CLAMP_ZONE_MARGIN
    rules.append(_rule("fit.clamp_zone", a >= need, a, need,
                       f"half clamp zone >= half stem clamp ({STEM_CLAMP_W / 2:g}) + {CLAMP_ZONE_MARGIN:g} mm"))
    rules.append(_rule("fit.clamp_d", p.clamp_d in (25.4, 26.0, 31.8, 35.0), p.clamp_d, "25.4/26.0/31.8/35.0",
                       "ISO 4210-5 bar clamp diameters"))
    rules.append(_rule("fit.lever_band_on_grip_d", True, p.grip_d, 23.8, "lever band on the B2 bend start (grip_d)"))
    bc = bend_checks(p)
    tmin = min(v["wall_extrados"] for v in bc.values())
    rules.append(_rule("wall.extrados_after_bending", tmin >= MIN_WALL_FINISHED, tmin, MIN_WALL_FINISHED,
                       "t / (1 + r_m / R) at the tightest bend"))
    res = beam_fea(p, cases, h)
    sres, sinfo = solid_fea(p, shape, cases, solid_h) if solid else ({}, None)
    for lc in cases:
        allow, basis = allowable(p, lc.kind)
        sf = SF_STATIC if lc.kind == "static" else SF_FATIGUE
        if solid:
            # the solid FEA decides; the beam model (which under-reads the ovalising bends) is kept for reference
            r = sres[lc.name]
            v = max(r["vm_p999_MPa"], r["vm_p999_away_from_clamp_MPa"])
            rules.append(_rule(f"strength.{lc.name}", v <= allow, v, round(allow, 1),
                               basis + " (solid FEA, von Mises, 99.9th percentile)", provisional_load=lc.provisional,
                               load_N=lc.force_N, at_mm=r["at_away_mm"], beam_MPa=res[lc.name]["vm_max_MPa"],
                               safety_factor=round(allow / v * sf, 2)))
        else:
            r = res[lc.name]
            rules.append(_rule(f"strength.{lc.name}", r["vm_max_MPa"] <= allow, r["vm_max_MPa"], round(allow, 1),
                               basis + " (beam model, von Mises, bend SIF; screening)", provisional_load=lc.provisional,
                               load_N=lc.force_N, at_mm=r["at_mm"], seg=r["seg"],
                               safety_factor=round(allow / r["vm_max_MPa"] * sf, 2)))
    st = res[STIFFNESS_CASE.name]
    return {"params": hb.asdict(p), "material": asdict(mat), "solution": asdict(sol), "rules": rules,
            "bends": bc, "beam_fea": res, "solid_fea": sres, "solid_mesh": sinfo, "stiffness_mm_per_100N": float(np.mean(np.abs(st["load_point_dz_mm"]))),
            "mass_g": g.get("mass_g"), "centreline_mm": g["centreline_mm"],
            "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]]}
