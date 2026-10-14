"""Closed-die hot forging: forging (blank) model, flash, and the die pair.

The die closes along Y and the parting plane is the stem plane (Y = 0), so every
side wall must taper away from the parting plane (draft) and nothing may overhang
in Y (undercut). Machined features the dies cannot form (bores across Y, threads,
the clamp slot) are left solid and machined later; the bar bore lies along Y and is
forged as two punched recesses with a web, pierced at trimming.

Envelope construction (`drafted_envelope`): the half above the parting plane is cut
into Y levels; for each band [y_k-1, y_k] the section at y_k-1 is grown by the
machining allowance plus the draft run over the band height and extruded from the
parting plane to y_k + allowance with the draft applied to its side faces. Each
prism contains its band with at least the allowance, every prism tapers away from
the parting plane, and so does their union: the result is forgeable by
construction and is then checked independently (`verify_forging`).

Process numbers (flash, shrinkage, press load) are the usual empirical starting
values for aluminium closed-die forging; they are to be tuned against a forging
simulation or die trials before the dies are cut.
"""
import math
from dataclasses import dataclass, asdict
import numpy as np
import shapely
from shapely.geometry import Polygon as SPolygon
from build123d import (Vector, Plane, Face, Wire, Solid, Part, Pos, Box, Cone, Align,
                       extrude, section, mirror, scale)
from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from ..parts import stem
from ..materials import MATERIALS

Y = Vector(0, 1, 0)
MERGE_MM = 0.15     # envelope prisms whose bases differ by less than this are merged

# forging temperature (stock), deg C: middle of the usual range for each alloy
FORGE_TEMP_C = {"A6061-T6": 450.0, "A7075-T6": 410.0, "A2014-T6": 440.0, "Ti-6Al-4V": 940.0}
# flow stress at forging temperature and a moderate strain rate, MPa (estimate for press sizing)
FLOW_STRESS_MPA = {"A6061-T6": 40.0, "A7075-T6": 65.0, "A2014-T6": 55.0, "Ti-6Al-4V": 150.0}
DIE_STEEL = ("SKD61", 7.80)   # JIS G 4404 hot-work tool steel, density g/cm3


@dataclass(frozen=True)
class ForgingParams:
    allowance_mm: float = 1.5      # machining allowance on every machined surface
    draft_deg: float = 5.0         # outer walls
    inner_draft_deg: float = 7.0   # recesses (bar-bore punches): the part shrinks onto them
    step_mm: float = 1.0           # Y spacing of the envelope levels
    web_mm: float = 6.0            # bar-bore web at the parting plane, pierced at trimming
    flash_t_mm: float = 0.0        # flash land thickness; 0 -> 0.015 sqrt(projected area)
    land_ratio: float = 3.0        # flash land width / thickness
    gutter_w_mm: float = 12.0      # flash gutter width beyond the land
    gutter_t_mm: float = 0.0       # gutter (total) thickness; 0 -> 4 x flash thickness
    die_wall_mm: float = 40.0      # die block wall beyond the gutter
    die_base_mm: float = 40.0      # die block thickness behind the deepest cavity point
    die_temp_C: float = 200.0      # preheated die
    load_factor: float = 10.0      # press load multiplier for a complex closed-die part with flash


# --------------------------------------------------------------- 2D helpers
def section2d(shape, y):
    """Section of `shape` by the plane Y = y as a shapely geometry in (x, z)."""
    pl = Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    tris = []
    for f in section(shape, pl).faces():
        vs, ts = f.tessellate(0.01, 0.1)
        P = [(v.X, v.Z) for v in vs]
        tris += [SPolygon([P[i] for i in t]) for t in ts]
    return shapely.union_all(tris).buffer(0) if tris else SPolygon()


def _polys(g):
    return [x for x in getattr(g, "geoms", [g]) if not x.is_empty]


def face_at(poly, y=0.0, tol=0.0):
    """Planar face at Y = y from a shapely polygon (holes kept), normal +Y."""
    poly = poly.simplify(tol) if tol else poly

    def wire(ring):
        return Wire.make_polygon([Vector(x, y, z) for x, z in list(ring.coords)[:-1]], close=True)
    f = Face(wire(poly.exterior), [wire(r) for r in poly.interiors])
    return f if f.normal_at().Y > 0 else -f


def drafted_prism(poly, h, deg, y0=0.0):
    """Prism on `poly` at Y = y0, height h along +Y, side faces drafted inward by `deg`
    (the base stays, the top shrinks by h tan(deg))."""
    return draft_sides(extrude(face_at(poly, y0), h, dir=(0, 1, 0)), deg, y0)


def draft_sides(pr, deg, y0=0.0):
    """Draft every side face of a prism standing on Y = y0 (planes and cylinders alike)
    inward by `deg` going up; the base stays."""
    da = BRepOffsetAPI_DraftAngle(pr.solids()[0].wrapped)
    pln = gp_Pln(gp_Pnt(0, y0, 0), gp_Dir(0, 1, 0))
    for fc in pr.faces():
        if abs(fc.normal_at().Y) < 1e-6:
            da.Add(fc.wrapped, gp_Dir(0, 1, 0), math.radians(deg), pln)
            if not da.AddDone():
                raise RuntimeError("draft could not be applied to a prism side")
    da.Build()
    if not da.IsDone():
        raise RuntimeError("draft failed")
    return Solid(da.Shape())


def _levels(ymax, step, breaks):
    lv = {0.0, ymax} | {min(ymax, i * step) for i in range(int(ymax / step) + 2)}
    lv |= {b for b in breaks if 0 < b < ymax}
    return sorted(round(v, 6) for v in lv)


def _clean(g, tol=0.05):
    """Fewer vertices (each one becomes a B-rep face), no slivers, no near-duplicate
    vertices. Grown by twice the tolerance first, so it still contains `g` after the
    simplification and after snapping (each may move the boundary in by `tol`)."""
    g = shapely.set_precision(g.buffer(2 * tol, quad_segs=4).simplify(tol), 1e-4)
    return shapely.union_all([x for x in _polys(g.buffer(0)) if x.area > 0.5])


def tidy(shape, min_edge=0.01):
    """Merge coplanar faces left by the boolean union, then collapse edges shorter
    than `min_edge` (they come from near-coincident prism walls and are rejected by
    downstream CAD). The STEP exchange check confirms the result."""
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.ShapeFix import ShapeFix_Wireframe, ShapeFix_Shape
    u = ShapeUpgrade_UnifySameDomain(shape.wrapped, True, True, True)
    u.SetLinearTolerance(1e-4); u.Build()
    s = u.Shape()
    fw = ShapeFix_Wireframe(s)
    fw.SetPrecision(min_edge); fw.ModeDropSmallEdges = True
    fw.FixSmallEdges(); fw.FixWireGaps()
    fx = ShapeFix_Shape(fw.Shape()); fx.Perform()
    return Solid(fx.Shape()) if fx.Shape().ShapeType().name == "TopAbs_SOLID" else type(shape)(fx.Shape())


def drafted_envelope(shape, allowance, draft_deg, step=1.0, breaks=(), symmetric=False):
    """Forgeable envelope of `shape` for a die closing along Y with the parting plane
    at Y = 0: contains `shape` grown by `allowance`, every side wall drafted.

    Band k = [y_k-1, y_k] of the shape, grown by the allowance, lies inside the drafted
    prism P_k on B_k = section(y_k-1) + allowance + h_k tan(draft), h_k = y_k +
    allowance. All P_k stand on the parting plane and taper away from it, so their
    union does too. A prism whose base lies within a taller one's (to MERGE_MM) is
    folded into it."""
    tan = math.tan(math.radians(draft_deg))

    def upper(s):
        ymax = s.bounding_box().max.Y
        lv = _levels(ymax, step, breaks)
        prisms = []
        for y0, y1 in zip(lv, lv[1:]):
            h = y1 + allowance
            for poly in _polys(_clean(section2d(s, y0 + 1e-3).buffer(allowance + h * tan, quad_segs=8))):
                prisms.append((SPolygon(poly.exterior), h))
        # tallest first; a base within MERGE_MM of a taller kept base is folded into it
        # (that prism grows by at most MERGE_MM): walls that would run a few microns
        # apart become one wall instead of sliver faces where they cross
        keep = []
        for b, h in sorted(prisms, key=lambda t: -t[1]):
            for i, (b2, h2) in enumerate(keep):
                if b.within(b2.buffer(MERGE_MM)):
                    if not b.within(b2):
                        keep[i] = (SPolygon(_clean(shapely.union_all([b2, b])).exterior), h2)
                    break
            else:
                keep.append((b, h))
        half = None
        for b, h in keep:
            pr = drafted_prism(b, h, draft_deg)
            half = pr if half is None else half + pr
        return half
    up = upper(shape)
    lo = mirror(up if symmetric else upper(mirror(shape, Plane.XZ)), Plane.XZ)
    return up + lo


# --------------------------------------------------------------- the stem forging
def _bar_recess(p, fp, ytop):
    """Two tapered punches forming the bar bore from both die halves, web at Y = 0.
    The radius at the die face never exceeds the finished bore minus the allowance."""
    d, n, S, B = stem.frame(p)
    tan = math.tan(math.radians(fp.inner_draft_deg))
    h = ytop + 1.0 - fp.web_mm / 2                          # punch runs out past the die face
    # the radius widens towards the die face and reaches bore - allowance where the
    # finished bore ends (Y = bar_clamp_w / 2); beyond that there is no finished surface
    r_web = p.bar_d / 2 - fp.allowance_mm - (p.bar_clamp_w / 2 - fp.web_mm / 2) * tan
    r_top = r_web + h * tan
    if r_web <= 1.0:
        return None
    out = None
    for s_ in (1, -1):
        pl = Plane(origin=(B.X, s_ * fp.web_mm / 2, B.Z), x_dir=(1, 0, 0), z_dir=(0, s_, 0))
        c = pl * Cone(r_web, r_top, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
        out = c if out is None else out + c
    return out


def forging_blank(p: stem.StemParams, fp: ForgingParams = ForgingParams()):
    """The forged part before trimming the flash: machining envelope of the stem
    (body and faceplate forged as one piece, split by machining)."""
    outer = stem.outer(p)
    breaks = (p.ear_half_w, p.body_w / 2, (p.body_w_end or p.body_w) / 2,
              p.fp_bolt_w - p.boss_margin, p.bar_clamp_w / 2)
    env = drafted_envelope(outer, fp.allowance_mm, fp.draft_deg, fp.step_mm, breaks, symmetric=True)
    rec = _bar_recess(p, fp, env.bounding_box().max.Y)
    return tidy(env - rec if rec is not None else env)


def flash_dims(blank, fp):
    outline = section2d(blank, 1e-3)
    area = outline.area
    t = fp.flash_t_mm or max(1.0, round(0.015 * math.sqrt(area), 1))
    b = fp.land_ratio * t
    tg = fp.gutter_t_mm or 4 * t
    return outline, area, t, b, tg


def forging_with_flash(blank, fp=ForgingParams()):
    """Forging as it leaves the die: the blank plus a full flash land of thickness t
    and width b round the parting outline (the gutter fills only partly, so it is not
    modelled). The plate is one solid under the whole outline: its edge stays b away
    from the forging walls, so the union has no near-coincident faces."""
    outline, _, t, b, _ = flash_dims(blank, fp)
    out = blank
    for r in _polys(_clean(outline.buffer(b, quad_segs=8))):
        out += extrude(face_at(SPolygon(r.exterior), -t / 2), t, dir=(0, 1, 0))
    return tidy(out)


def shrink_factor(p, fp=ForgingParams()):
    """Die cavity = cold forging x this: the part contracts from forging to room
    temperature, the die has grown from room to die temperature."""
    m = MATERIALS[p.material]
    return (1 + m.cte_per_K * (FORGE_TEMP_C[p.material] - 20)) / (1 + 11.5e-6 * (fp.die_temp_C - 20))


def dies(p, blank, fp=ForgingParams()):
    """(upper die, lower die) blocks with the cavity, flash land and gutter.
    The upper die sits at Y >= 0; the lower is its mirror (the part is symmetric)."""
    k = shrink_factor(p, fp)
    cavity = scale(blank, k)
    outline, _, t, b, tg = flash_dims(cavity, fp)
    land = _clean(outline.buffer(b, quad_segs=8))                    # the section's tessellation
    gutter_out = _clean(outline.buffer(b + fp.gutter_w_mm, quad_segs=8))  # leaves sub-0.01 mm segments
    gutter = gutter_out.difference(land)          # the ring beyond the land: the land keeps its t/2
    x0, z0, x1, z1 = gutter_out.bounds
    w = fp.die_wall_mm
    ytop = cavity.bounding_box().max.Y
    hb = ytop + fp.die_base_mm
    block = Pos((x0 + x1) / 2, hb / 2, (z0 + z1) / 2) * Box(x1 - x0 + 2 * w, hb, z1 - z0 + 2 * w)
    upper = block - cavity
    upper -= extrude(face_at(land, 0.0), t / 2, dir=(0, 1, 0))
    for g in _polys(gutter):
        upper -= extrude(face_at(g, 0.0), tg / 2, dir=(0, 1, 0))
    return upper, mirror(upper, Plane.XZ)


def die_set(p, fp=ForgingParams()):
    blank = forging_blank(p, fp)
    up, lo = dies(p, blank, fp)
    return dict(blank=blank, forging=forging_with_flash(blank, fp), upper=up, lower=lo)
