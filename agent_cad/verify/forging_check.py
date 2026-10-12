"""Checks on a forging and its dies, measured on the B-rep.

    report = verify_forging(StemParams())

- allowance: every finished surface lies inside the forging, at least the machining
  allowance below its surface (along the surface normal, by ray casting)
- draft: every wall of the forging tapers away from the parting plane (Y = 0)
  by at least the draft angle
- undercut: from every surface point a ray away from the parting plane, along the
  die stroke, leaves the forging without re-entering it (the die can withdraw)
- dies: one valid solid each, no overlap with each other or with the hot forging
- process: forging mass, material yield, press load estimate
"""
import math
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Dir, gp_Lin
from OCP.Precision import Precision
from build123d import scale
from ..parts import stem
from ..process import forging as fg

TOL_ALLOW = 0.05      # mm: polygonised sections and tessellation
TOL_DRAFT = 0.05      # deg
TOL_UNDERCUT = 0.01   # mm: the STEP check's shortest allowed edge


def surface_samples(shape, per_face=4):
    """[(point, outward normal, face index)] on a per_face x per_face grid of each face."""
    out = []
    for fi, f in enumerate(shape.faces()):
        s = BRepAdaptor_Surface(f.wrapped)
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        props = BRepLProp_SLProps(s, 1, 1e-7)
        cls = BRepClass_FaceClassifier()
        rev = f.wrapped.Orientation() == TopAbs_REVERSED
        for a in (np.arange(per_face) + 0.5) / per_face:
            for b in (np.arange(per_face) + 0.5) / per_face:
                u, v = u0 + a * (u1 - u0), v0 + b * (v1 - v0)
                cls.Perform(f.wrapped, gp_Pnt2d(u, v), 1e-7)
                if cls.State() != TopAbs_IN:
                    continue
                props.SetParameters(u, v)
                if not props.IsNormalDefined():
                    continue
                n = props.Normal()
                nv = np.array([n.X(), n.Y(), n.Z()]) * (-1 if rev else 1)
                q = props.Value()
                out.append((np.array([q.X(), q.Y(), q.Z()]), nv, fi))
    return out


def allowance(finished, forging, per_face=3):
    """Machining stock: from sample points on the finished surfaces, the distance along
    the outward normal to the forging surface (a ray cast). Returns (smallest, where)."""
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(forging.wrapped, 1e-6)
    best = (math.inf, None)
    for shp in finished:
        for q, n, _ in surface_samples(shp, per_face):
            inter.Perform(gp_Lin(gp_Pnt(*q), gp_Dir(*n)), 0.0, Precision.Infinite_s())
            if not inter.IsDone() or inter.NbPnt() == 0:
                return 0.0, tuple(round(float(c), 2) for c in q)     # the point is outside the forging
            w = min(inter.WParameter(k) for k in range(1, inter.NbPnt() + 1))
            if w < best[0]:
                best = (w, tuple(round(float(c), 2) for c in q))
    return best


def draft_and_undercut(forging, per_face=4, parting_y=0.0, draft_deg=5.0):
    """(min draft deg, where), [undercut points]. Draft at a point = angle between
    the wall and the die stroke, positive when the wall opens away from the parting
    plane. Faces lying in the parting plane are skipped.

    Undercut: a crossing counts beyond TOL_UNDERCUT / tan(draft) along the ray. Next
    to a wall drafted at that angle, a sideways deviation within the model tolerance
    (healed sub-0.01 mm edges) is crossed over that length; any real overhang is
    met farther up."""
    reach = TOL_UNDERCUT / math.tan(math.radians(max(draft_deg, 0.5)))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(forging.wrapped, 1e-6)
    worst, under = (math.inf, None), []
    for q, n, _ in surface_samples(forging, per_face):
        side = np.sign(q[1] - parting_y)
        if side == 0 or abs(q[1] - parting_y) < 1e-4:
            continue
        dr = math.degrees(math.asin(max(-1.0, min(1.0, n[1] * side))))
        if dr < worst[0]:
            worst = (dr, tuple(round(float(c), 2) for c in q))
        if n[1] * side < 1e-6 and abs(n[1]) < 1e-6:
            continue        # a vertical wall: reported by the draft rule, the ray would graze it
        start = gp_Pnt(*(q + np.array([0, side * 1e-3, 0]) + n * 1e-4))
        inter.Perform(gp_Lin(start, gp_Dir(0, side, 0)), 0.0, Precision.Infinite_s())
        if inter.IsDone() and any(inter.WParameter(k) > reach for k in range(1, inter.NbPnt() + 1)):
            under.append(tuple(round(float(c), 2) for c in q))
    return worst, under


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def verify_forging(p: stem.StemParams, fp: fg.ForgingParams = fg.ForgingParams(), parts=None, ds=None):
    from ..materials import MATERIALS
    b, f = parts or stem.build(p)
    ds = ds or fg.die_set(p, fp)
    blank, forging, up, lo = ds["blank"], ds["forging"], ds["upper"], ds["lower"]
    rho = stem.density(p)
    rules = []
    unc = (b - blank).volume + (f - blank).volume
    rules.append(_rule("forging.contains_part", unc < 1e-3, round(unc, 4), 0.0, "mm3 of finished part outside the forging"))
    amin, at = allowance([b, f], blank)
    rules.append(_rule("forging.allowance", amin >= fp.allowance_mm - TOL_ALLOW, round(amin, 3), fp.allowance_mm,
                       "machining allowance, distance from finished surface to forging surface", at_mm=at))
    (dmin, dat), under = draft_and_undercut(blank, per_face=2, draft_deg=fp.draft_deg)
    rules.append(_rule("forging.draft", dmin >= fp.draft_deg - TOL_DRAFT, round(dmin, 2), fp.draft_deg,
                       "wall angle to the die stroke (Y), away from the parting plane", at_mm=dat))
    rules.append(_rule("forging.undercut", not under, len(under), 0, "points from which the die cannot withdraw",
                       at_mm=under[:5]))
    k = fg.shrink_factor(p, fp)
    hot = scale(blank, k)
    for name, die in (("upper", up), ("lower", lo)):
        ok = die.is_valid and len(die.solids()) == 1
        rules.append(_rule(f"die.{name}.valid", ok, ok, True, "one valid solid"))
        ov = (die & hot).volume
        rules.append(_rule(f"die.{name}.clears_forging", ov < 1e-2, round(ov, 4), 0.0, "mm3 overlap with the hot forging"))
    ov = (up & lo).volume
    rules.append(_rule("die.halves_clear", ov < 1e-2, round(ov, 4), 0.0, "mm3 overlap between upper and lower die"))
    outline, area, t, land_w, tg = fg.flash_dims(blank, fp)
    m_fin = (b.volume + f.volume) / 1000 * rho
    m_blank = blank.volume / 1000 * rho
    m_forge = forging.volume / 1000 * rho
    flow = fg.FLOW_STRESS_MPA[p.material]
    area_flash = outline.buffer(land_w).area
    load_kN = fp.load_factor * flow * area_flash / 1000
    die_steel, die_rho = fg.DIE_STEEL
    return {
        "rules": rules, "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]],
        "process": {
            "parting_plane": "Y = 0 (stem plane), die stroke along Y",
            "forge_temp_C": fg.FORGE_TEMP_C[p.material], "die_temp_C": fp.die_temp_C,
            "shrink_factor": round(k, 5),
            "flash_t_mm": t, "flash_land_w_mm": land_w, "gutter_t_mm": tg, "gutter_w_mm": fp.gutter_w_mm,
            "projected_area_mm2": round(area, 0), "projected_area_with_land_mm2": round(area_flash, 0),
            "press_load_kN": round(load_kN, 0), "press_load_t": round(load_kN / 9.81, 0),
            "press_load_basis": f"{fp.load_factor} x flow stress {flow} MPa x area incl. flash land (estimate)",
            "mass_finished_g": round(m_fin, 1), "mass_forging_trimmed_g": round(m_blank, 1),
            "mass_forging_with_flash_g": round(m_forge, 1),
            "material_yield_pct": round(100 * m_fin / m_forge, 1),
            "die_steel": die_steel,
            "die_mass_kg": round((up.volume + lo.volume) / 1e6 * die_rho, 1),
            "die_block_mm": [round(v, 1) for v in (up.bounding_box().size.X, up.bounding_box().size.Y,
                                                   up.bounding_box().size.Z)],
        },
    }
