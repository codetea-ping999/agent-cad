"""Injection moulding: feed system, the mould halves, the slide, ejection and cooling
for the moulded housing (single cavity, cold runner).

Layout. The parting plane is the housing rim (Y = 0). The fixed half (Y >= 0)
holds the cavity (the outside of the box), the sprue, runner and edge gate; the
moving half (Y <= 0, its core standing up into the box) forms the inside, the
ribs and bosses and holds the cold-slug well that pulls the sprue. The part
shrinks onto the core and leaves with the moving half; ejector pins push on the
floor between the ribs, ejector sleeves on the boss tips, one pin each under the
runner and the cold-slug well.

The connector window is an undercut for both halves. A slide on the moving half
forms it together with the outer face around it; an angle pin in the fixed half
pulls the slide out along -Z while the mould opens, and a locking heel in the
fixed half (steeper than the pin) holds it against the injection pressure.

    component     forms / does                          leaves the part along
    fixed half    outside, sprue, runner, gate          +Y (part moves -Y with the core)
    moving half   inside, ribs, bosses, cold-slug well  -Y (part ejected +Y)
    slide         window and the outer face around it   -Z
    core pins     screw holes in the bosses             -Y
    sleeves       push on the boss tips                 -Y
    ejector pins  push on the floor and the well        -Y

The mould is cut at room temperature: every moulding surface is the shot scaled
by (1 + shrinkage). Process numbers (pressures, temperatures, diffusivity, shear
limits, shrinkage) are typical datasheet / handbook starting values for each
resin, to be confirmed by a filling-packing-warpage simulation and mould trials.
"""
import math
from dataclasses import dataclass, asdict
from shapely.geometry import box, Point
import shapely
from shapely.geometry import Polygon as SPolygon
from build123d import Box, Cylinder, Pos, Rot, Align, Compound, Vector, Face, Wire, extrude, scale
from ..parts import housing
from .forging import drafted_prism

MOULD_STEEL = ("SCM440 / NAK80", 7.85)       # plates (SCM440H) / cavity and core (pre-hardened NAK80)
MACHINES_T = (30, 50, 80, 100, 130, 180, 220, 280, 350, 450, 550, 650, 850, 1300)   # usual sizes, tf


@dataclass(frozen=True)
class Resin:
    shrink_pct: float          # linear moulding shrinkage (mean of flow / cross-flow)
    T_melt_C: float
    T_mould_C: float
    T_eject_C: float           # part centre temperature at which it can be ejected
    diffusivity_mm2_s: float   # thermal diffusivity in the melt range
    max_shear_1_s: float       # maximum wall shear rate the grade tolerates
    flow_ratio: float          # usual limit of flow length / wall thickness
    cavity_MPa: float          # mean cavity pressure for clamp sizing
    min_draft_deg: float
    min_wall_mm: float
    iso_mark: str              # ISO 11469 / JIS K 6899 marking


RESINS = {
    "PA66-GF30": Resin(0.5, 290, 85, 180, 0.12, 60_000, 150, 40, 0.5, 1.0, ">PA66-GF30<"),
    "PA6-GF30": Resin(0.5, 275, 80, 170, 0.12, 60_000, 150, 40, 0.5, 1.0, ">PA6-GF30<"),
    "PBT-GF30": Resin(0.6, 260, 70, 160, 0.12, 50_000, 150, 40, 0.5, 1.0, ">PBT-GF30<"),
    "PP": Resin(1.6, 230, 40, 90, 0.09, 100_000, 250, 25, 0.5, 0.8, ">PP<"),
}


@dataclass(frozen=True)
class MouldParams:
    fill_time_s: float = 0.8
    clamp_margin: float = 1.2
    gate_t_ratio: float = 0.6         # gate thickness / wall
    gate_land_mm: float = 1.0
    gate_w_min_mm: float = 4.0
    shear_design_frac: float = 0.5    # gate sized for this fraction of the resin's shear limit
    runner_w_mm: float = 6.0          # trapezoid, at the parting plane
    runner_d_mm: float = 5.0
    runner_draft_deg: float = 5.0
    runner_len_mm: float = 30.0       # gate land end to sprue axis (along +X)
    sprue_d_mm: float = 4.0           # small end, at the nozzle (nozzle orifice + 0.5)
    sprue_taper_deg: float = 1.5      # per side
    well_depth_mm: float = 6.0        # cold-slug well below the sprue
    well_draft_deg: float = 2.0
    pin_d_mm: float = 4.0             # ejector pins (JIS B 5103)
    pin_clear_mm: float = 1.0         # pin edge to any wall foot
    pin_tail_mm: float = 30.0         # pin length behind the mould (ejector plate)
    sleeve_inset_mm: float = 0.5      # sleeve outside below the boss tip outside
    cool_d_mm: float = 8.0            # drilled channels in the plates
    bubbler_d_mm: float = 6.0         # bubblers up into the core
    cool_cover_mm: float = 17.0       # plate channel axis to the nearest moulding surface plane
    cool_pitch_mm: float = 20.0
    slide_margin_mm: float = 6.0      # slide face beyond the window, each side along X
    slide_top_mm: float = 6.0         # slide top above the window
    slide_depth_mm: float = 40.0      # slide body behind the part's outer face (Z)
    angle_pin_deg: float = 18.0
    lock_deg: float = 20.0            # locking heel; >= angle pin + 2
    angle_pin_d_mm: float = 12.0
    angle_pin_clear_mm: float = 0.5   # radial, in the slide
    angle_pin_tip_mm: float = 4.0     # lowest point of the pin above the parting plane
    angle_pin_anchor_mm: float = 30.0  # pin length held in the fixed half above the slide
    slide_safety_mm: float = 2.0      # slide travel beyond the undercut
    plate_wall_mm: float = 40.0       # mould block beyond the shot and slide
    fixed_over_mm: float = 45.0       # fixed half above the part top
    moving_t_mm: float = 50.0         # moving half below the parting plane


def resin(p):
    return RESINS[p.material]


def shrink_factor(p):
    return 1 + resin(p).shrink_pct / 100


# ------------------------------------------------------------------ feed system
def gate_size(p, mp, volume_mm3):
    """Edge gate (thickness, width) for the flow rate at the set fill time: width from the
    rectangular-slit wall shear rate 6Q/(w h^2) at the design fraction of the resin's limit."""
    h = mp.gate_t_ratio * p.wall
    q = volume_mm3 / mp.fill_time_s
    w = max(mp.gate_w_min_mm, 6 * q / (mp.shear_design_frac * resin(p).max_shear_1_s * h * h))
    return h, w


def sprue_x(p, mp):
    return p.length / 2 + mp.gate_land_mm + mp.runner_len_mm


def fixed_height(p, mp):
    return p.height * shrink_factor(p) + mp.fixed_over_mm


def cone_y(x, z, y0, r0, h, deg):
    return housing.cone_y(x, z, y0, r0, h, deg)


def gating(p, mp, part_volume):
    """dict(gate, runner, sprue, well) solids at part scale (the shot before shrink scaling)."""
    h, w = gate_size(p, mp, part_volume)
    x0 = p.length / 2
    gate = drafted_prism(box(x0 - 0.5, -w / 2, x0 + mp.gate_land_mm + 0.5, w / 2), h, mp.runner_draft_deg)
    xs = sprue_x(p, mp)
    rw = mp.runner_w_mm
    runner = drafted_prism(box(x0 + mp.gate_land_mm, -rw / 2, xs, rw / 2).union(Point(xs, 0).buffer(rw / 2, quad_segs=8)),
                           mp.runner_d_mm, mp.runner_draft_deg)
    Hs = fixed_height(p, mp) / shrink_factor(p)            # sprue runs through the whole fixed half
    r_top = mp.sprue_d_mm / 2
    r0 = r_top + Hs * math.tan(math.radians(mp.sprue_taper_deg))
    sprue = cone_y(xs, 0.0, 0.0, r0, Hs, mp.sprue_taper_deg)
    rw0 = r0 - mp.well_depth_mm * math.tan(math.radians(mp.well_draft_deg))
    well = cone_y(xs, 0.0, -mp.well_depth_mm, rw0, mp.well_depth_mm, -mp.well_draft_deg)
    return dict(gate=gate, runner=runner, sprue=sprue, well=well, gate_t=h, gate_w=w, sprue_r0=r0)


def shot(p, mp, part):
    g = gating(p, mp, part.volume)
    s = part + g["gate"] + g["runner"] + g["sprue"] + g["well"]
    return s.clean(), g


# ------------------------------------------------------------------ slide
def slide_frame(p, mp):
    """Slide block in mould coordinates: x range, top Y, z of the back face at Y = 0 and of
    the part's outer face at the rim."""
    k = shrink_factor(p)
    m = mp.slide_margin_mm
    x0, x1 = (p.window_x - p.window_w / 2 - m) * k, (p.window_x + p.window_w / 2 + m) * k
    ys = (p.window_y + p.window_h / 2 + mp.slide_top_mm) * k
    zf = -p.width / 2 * k
    return x0, x1, ys, zf - mp.slide_depth_mm, zf


def _prism_x(pts_yz, x0, x1):
    """Prism along X over a polygon given as (y, z) points."""
    w = Wire.make_polygon([Vector(x0, y, z) for y, z in pts_yz], close=True)
    return extrude(Face(w), x1 - x0, dir=(1, 0, 0))


def slide_block(p, mp):
    """The slide's envelope: front at 1 mm inside the part outline (trimmed by the part
    envelope later), back face leaning in by the locking angle so the fixed half's heel
    lifts off it while the mould opens."""
    x0, x1, ys, zb, zf = slide_frame(p, mp)
    tl = math.tan(math.radians(mp.lock_deg))
    return _prism_x([(0, zb), (0, zf + 1), (ys, zf + 1), (ys, zb + ys * tl)], x0, x1)


def angle_pin_axis(p, mp):
    """(point on the axis, unit direction): leaning toward the part (+Z) going up."""
    x0, x1, ys, zb, zf = slide_frame(p, mp)
    a = math.radians(mp.angle_pin_deg)
    u = Vector(0, math.cos(a), math.sin(a))
    c = Vector((x0 + x1) / 2, ys / 2, (zb + zf) / 2)
    return c, u


def angle_pin(p, mp, r_extra=0.0, y_lo=None, y_hi=None):
    c, u = angle_pin_axis(p, mp)
    x0, x1, ys, zb, zf = slide_frame(p, mp)
    r = mp.angle_pin_d_mm / 2 + r_extra
    a = math.radians(mp.angle_pin_deg)
    y_lo = mp.angle_pin_tip_mm + r * math.sin(a) if y_lo is None else y_lo   # the end face's lowest point
    y_hi = ys + mp.angle_pin_anchor_mm if y_hi is None else y_hi
    s0, s1 = (y_lo - c.Y) / u.Y, (y_hi - c.Y) / u.Y
    start = c + u * s0
    return Pos(start.X, start.Y, start.Z) * (Rot(mp.angle_pin_deg - 90, 0, 0) *
                                             Cylinder(r, s1 - s0, align=(Align.CENTER, Align.CENTER, Align.MIN)))


def slide_travel(p, mp):
    """Slide stroke while the angle pin is engaged: the pin's vertical run inside the
    slide times tan(angle), less the radial clearance taken up first."""
    x0, x1, ys, zb, zf = slide_frame(p, mp)
    a = math.radians(mp.angle_pin_deg)
    return (ys - mp.angle_pin_tip_mm) * math.tan(a) - mp.angle_pin_clear_mm / math.cos(a)


# ------------------------------------------------------------------ ejection
def flat_faces(solid, y, sign, tol=1e-3):
    """Planar faces of `solid` facing `sign` * Y at height y, as one shapely region (x, z)."""
    tris = []
    for f in solid.faces():
        if f.geom_type.name != "PLANE":
            continue
        if f.normal_at().Y * sign > 1 - 1e-9 and abs(f.center().Y - y) < tol:
            vs, ts = f.tessellate(0.01, 0.1)
            P = [(v.X, v.Z) for v in vs]
            tris += [SPolygon([P[i] for i in t]) for t in ts]
    return shapely.union_all(tris).buffer(0) if tris else SPolygon()


def bubbler_sites(p):
    return [(sx * 0.2 * p.length, sz * 0.2 * p.width) for sx in (-1, 1) for sz in (-1, 1)]


def pin_sites(p, mp, shot_solid):
    """Ejector pins (x, z, y_face) at part scale, pushing +Y: on the floor's underside
    between ribs, bosses and walls (every pin circle pin_clear inside the flat face), and
    under the cold-slug well."""
    r = mp.pin_d_mm / 2 + mp.pin_clear_mm
    fy = housing.floor_y(p)
    cands = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            cands += [(sx * 0.38 * p.length, sz * 0.12 * p.width, fy), (sx * 0.08 * p.length, sz * 0.3 * p.width, fy)]
    xs = sprue_x(p, mp)
    cands.append((xs, 0.0, -mp.well_depth_mm))                               # sprue puller
    out = []
    for x, z, y in cands:
        region = flat_faces(shot_solid, y, -1)
        if not region.is_empty and region.buffer(-1e-3).contains(Point(x, z).buffer(r, quad_segs=16)):
            out.append((x, z, y))
    return out


def _down_cyl(x, y_top, z, r, y_bottom):
    """Cylinder on the axis (x, z) from y_bottom up to y_top."""
    return Pos(x, y_bottom, z) * Cylinder(r, y_top - y_bottom, rotation=(-90, 0, 0),
                                          align=(Align.CENTER, Align.CENTER, Align.MIN))


# ------------------------------------------------------------------ mould
def mould(p, mp, part, shot_solid):
    k = shrink_factor(p)
    cav = scale(shot_solid, k)
    env = scale(housing.envelope(p), k)
    part_k = scale(part, k)
    pocket_k = scale(housing.pocket(p), k)
    win = ((scale(housing.window_cutter(p), k) & env) - part_k - pocket_k).clean()

    blk = slide_block(p, mp)
    retracted = Pos(0, 0, -slide_travel(p, mp)) * blk
    bb = (Compound([cav, blk, retracted])).bounding_box()
    w = mp.plate_wall_mm
    x0, x1, z0, z1 = bb.min.X - w, bb.max.X + w, bb.min.Z - w, bb.max.Z + w
    cx, cz, lx, lz = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    hf, hm = fixed_height(p, mp), mp.moving_t_mm
    bottom = -hm - mp.pin_tail_mm

    # slide: its block trimmed to the part's outer face, plus the window insert
    slide = (blk - env) + win
    slide -= angle_pin(p, mp, r_extra=mp.angle_pin_clear_mm, y_lo=-10.0, y_hi=slide_frame(p, mp)[2] + 10.0)
    pin_a = angle_pin(p, mp)

    fixed = Pos(cx, hf / 2, cz) * Box(lx, hf, lz) - env - cav - blk - pin_a
    moving = (Pos(cx, -hm / 2, cz) * Box(lx, hm, lz) + env) - cav - win

    # boss tips: ejector sleeves around the core pins that form the screw holes
    sleeves, core_pins = [], []
    for x, z in housing.bosses(p):
        rs = (p.boss_d / 2 - mp.sleeve_inset_mm) * k
        rh = p.hole_d / 2 * k
        yt = p.boss_recess * k
        outer = _down_cyl(x * k, yt, z * k, rs, bottom)
        # the core pin: a cylinder of the hole's tip diameter up to the hole bottom, less the
        # moulding (the hole narrows going up, the plastic around it is already cavity)
        full = _down_cyl(x * k, (p.boss_recess + p.hole_depth) * k, z * k, rh, bottom - 20.0)
        core_pins.append((full - cav).clean())
        sleeves.append(outer - _down_cyl(x * k, yt + 1, z * k, rh, bottom - 1))
        moving -= outer
        moving -= full

    sites = pin_sites(p, mp, shot_solid)
    pins = [_down_cyl(x * k, y * k, z * k, mp.pin_d_mm / 2, bottom) for x, z, y in sites]
    for pn in pins:
        moving -= pn

    # cooling. Moving half: channels along Z under the core at x = bubbler x, each feeding
    # bubblers up into the core; fixed half: channels along X above the part, clear of the
    # sprue and the angle pin
    ym = -mp.cool_cover_mm
    rb = mp.bubbler_d_mm / 2
    yb = housing.floor_y(p) * k - 1.5 * mp.bubbler_d_mm - 0.5
    bxs = sorted({x for x, _ in bubbler_sites(p)})
    ch_m = [Pos(x * k, ym, cz) * Cylinder(mp.cool_d_mm / 2, lz + 2) for x in bxs]
    bub = [_down_cyl(x * k, yb, z * k, rb, ym) for x, z in bubbler_sites(p)]
    yf = p.height * k + mp.cool_cover_mm
    c, u = angle_pin_axis(p, mp)
    z_pin = c.Z + (yf - c.Y) / u.Y * u.Z
    rc = mp.cool_d_mm / 2
    g = gating(p, mp, part.volume)
    # the angle pin leans by its angle: the true gap to a horizontal channel is the offset x cos(angle)
    avoid = [(0.0, g["sprue_r0"] * k + rc + 1.5 * mp.cool_d_mm),
             (z_pin, (mp.angle_pin_d_mm / 2 + rc + 4.0) / math.cos(math.radians(mp.angle_pin_deg)) + 0.5)]
    zs, z = [], z0 + w / 2
    while z <= z1 - w / 2:
        if all(abs(z - a) >= gap for a, gap in avoid) and (not zs or z - zs[-1] >= mp.cool_pitch_mm):
            zs.append(z); z += mp.cool_pitch_mm
        else:
            z += 0.5
    ch_f = [Pos(cx, yf, zz) * Cylinder(rc, lx + 2, rotation=(0, 90, 0)) for zz in zs]
    for c_ in ch_m + bub:
        moving -= c_
    for c_ in ch_f:
        fixed -= c_
    # the machine nozzle seats on the sprue's small end at the back of the fixed half
    nozzle = _down_cyl(sprue_x(p, mp) * k, hf + 20.0, 0.0, mp.sprue_d_mm, hf)
    return dict(nozzle=nozzle, fixed=fixed.clean(), moving=moving.clean(), slide=slide.clean(), angle_pin=pin_a,
                core_pins=core_pins, sleeves=sleeves, pins=pins, pin_sites=sites,
                channels_fixed=ch_f, channels_moving=ch_m, bubblers=bub, cavity=cav, window_insert=win,
                envelope=env, slide_block=blk, block_mm=(lx, hf + hm, lz), plate_y=(-hm, hf))


def mould_set(p=housing.HousingParams(), mp=MouldParams()):
    part = housing.build(p)
    s, g = shot(p, mp, part)
    d = mould(p, mp, part, s)
    d.update(part=part, shot=s, gating=g)
    return d
