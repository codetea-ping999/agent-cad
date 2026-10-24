"""Manufacturing route and tooling of the drop handlebar: drawn tube -> rotary swaging ->
rotary-draw bending -> heat treatment.

Route (each step is checked in verify/tube_check.py):
 1. Cut a seamless drawn tube in the O temper (A7075-O, JIS H 4080), clamp_d x wall_clamp,
    to the blank length: the volume of the finished bar plus the bending allowance on both
    drop ends, at the start section (swaging keeps the volume and lengthens the tube).
 2. Rotary-swage both ends to grip_d: the outside taper free (sinking, the wall
    thickens), then over a stepped mandrel inserted from the end, which sets the butt and
    the grip bore. Passes from the total reduction of area.
 3. Rotary-draw bend in the O temper, each half from the centre outward (B1, B2, B3), so
    the bend still to come is straight and the mandrel enters from the open drop end. One
    bend die serves every bend of the same radius; the clamp holds the leading side (toward
    the centre). Behind B1 the hood straight is too short to clamp, so B2 uses a contoured
    clamp die and insert that follow the B1 arc.
 4. Cut the bending allowance off the drop ends, solution treat 470 C, quench, age to T6
    (120 C x 24 h) in a sizing fixture, then check on the inspection gauge.
 5. Deburr, shot blast, anodise, laser-mark (centre and angle scale) the clamp zone.

The machine frame of a bend (the bend die axis is Z', the tangent point of the tube on the
die is (0, -R, 0), the tube runs along +X' from the pressure die toward the clamp) is
fixed to the arc centre at the end of the bend; the leading part swings about Z' as the
die turns. The tools are made in that frame for the set-up of the bend they belong to.
"""
import math
from dataclasses import dataclass, asdict
import numpy as np
from build123d import (Box, Cylinder, Sphere, Torus, Pos, Rot, Location, Plane, Align, Axis, Vector, Compound,
                       Polyline, make_face, revolve, Part, Line, ThreePointArc, Wire)
from ..parts import handlebar as hb
from ..materials import MATERIALS

TOOL_STEEL = ("SKD11 (JIS G 4404)", "焼入れ焼戻し 58〜60 HRC、溝部 硬質クロムめっき")
WIPER_MAT = ("アルミニウム青銅 (JIS H 5120 CAC703)", "鋳放し後機械加工")
MANDREL_MAT = ("SKD11 (JIS G 4404)", "焼入れ焼戻し 58〜60 HRC、硬質クロムめっき")
SWAGE_MAT = ("SKD11 (JIS G 4404)", "焼入れ焼戻し 60〜62 HRC")
STEEL_DENSITY = 7.8
BENDING_TEMPER = {"A7075-T6": "A7075-O", "A6061-T6": "A6061-O"}
MAX_RA_PER_PASS = 0.30          # rotary swaging, reduction of area per pass (rule of thumb)
ANNEAL_ABOVE_RA = 0.50          # intermediate anneal when the total reduction exceeds this


@dataclass(frozen=True)
class TubeProcessParams:
    end_allow: float = 40.0        # straight added to each drop end, cut off after bending
    trim_allow: float = 2.0        # cut and face, per end
    clamp_len_D: float = 2.0       # clamp grip length / tube diameter (serrated insert)
    pd_extra_D: float = 1.0        # pressure die beyond the bend arc / D
    wiper_len_D: float = 2.5
    groove_clr: float = 0.1        # radial, tube groove in every tool
    groove_depth: float = 0.52     # bend die groove depth / D: the die holds the inner half, the
                                   # pressure and clamp dies the outer half
    die_rim: float = 15.0          # bend die above and below the groove (axial)
    pd_gap: float = 1.0            # pressure die face to the bend die (behind the tube centre plane)
    die_hub_d: float = 50.0        # bend die bore on the machine spindle
    tool_wall: float = 14.0        # clamp / pressure / wiper die beyond the groove
    mandrel_clr: float = 0.15      # radial, mandrel to bore
    ball_pitch_d: float = 0.8      # ball pitch / ball diameter
    swage_entry: float = 12.0      # swaging die entry bell length
    swage_land: float = 40.0       # sizing land length
    swage_block: tuple = (90.0, 30.0, 70.0)   # die half outside: across, height, (length from profile)
    table_gap: float = 5.0         # machine table below the bend die
    carriage_half: float = 100.0   # carriage / collet envelope around the tube axis


# ------------------------------------------------------------------ blank and swaging

def _area(ro, ri):
    return math.pi * (ro * ro - ri * ri)


def blank(p, tp=TubeProcessParams()):
    """Start tube and cut length from the volume of the finished bar (per half: tops
    profile integrated, then the grip section to the end plus the bending allowance)."""
    sol = hb.solve(p)
    Rc = p.clamp_d / 2
    A0 = _area(Rc, Rc - p.wall_clamp)
    ys = np.linspace(0, sol.L0, 4001)
    A = np.array([_area(*hb.radii_at(p, y)) for y in ys])
    V_tops = np.trapezoid(A, ys)
    rest = sum(s.length for s in hb.path(p, 1, sol)[1:])
    Ag = _area(p.grip_d / 2, p.grip_d / 2 - p.wall_grip)
    V_half = V_tops + Ag * (rest + tp.end_allow)
    L_half = V_half / A0
    L = 2 * (L_half + tp.trim_allow)
    m = (2 * V_half + 2 * tp.trim_allow * A0) * hb.density(p) / 1000
    return dict(od=p.clamp_d, wall=p.wall_clamp, temper=BENDING_TEMPER.get(p.material, p.material),
                cut_length=round(L, 1), mass_g=round(m, 1), area_mm2=round(A0, 2),
                bent_length_half=round(sol.L0 + rest + tp.end_allow, 2))


def swaging(p, tp=TubeProcessParams()):
    Rc, Rg = p.clamp_d / 2, p.grip_d / 2
    A0 = _area(Rc, Rc - p.wall_clamp)
    a_min = min(_area(Rg, Rg - w) for w in (p.wall_butt, p.wall_grip))
    ra = 1 - a_min / A0
    passes = max(1, math.ceil(math.log(1 - ra) / math.log(1 - MAX_RA_PER_PASS)))
    a, b, n, c, d = hb.zones(p)
    return dict(od_from=p.clamp_d, od_to=p.grip_d, reduction_of_area=round(ra, 3), passes=passes,
                anneal_between_passes=ra > ANNEAL_ABOVE_RA, elongation=round(A0 / a_min, 3),
                mandrel_from_y=b, taper_wall_end=round(hb.sunk_wall(p, p.grip_d), 3),
                mandrel=dict(nose_d=[round(2 * hb.radii_at(p, b + 1e-6)[1] - 2 * tp.mandrel_clr, 3),
                                     round(2 * (Rg - p.wall_butt) - 2 * tp.mandrel_clr, 3)],
                             butt_d=round(2 * (Rg - p.wall_butt), 3), grip_d=round(2 * (Rg - p.wall_grip), 3)))


# ------------------------------------------------------------------ bending

def springback(p, R, angle_deg):
    """Elastic-plastic beam springback of the bend in the bending temper:
    R_i / R_f = 4 x^3 - 3 x + 1, x = Sy R_i / (E c), c = D / 2; arc length is kept, so the
    die has to over-bend by angle (R_f / R_i - 1)."""
    m = MATERIALS[BENDING_TEMPER.get(p.material, p.material)]
    x = m.yield_MPa * R / (m.E_MPa * p.grip_d / 2)
    k = 4 * x ** 3 - 3 * x + 1
    Rf = R / k
    return dict(R_die=R, R_after=round(Rf, 3), overbend_deg=round(angle_deg * (Rf / R - 1), 2), x=round(x, 5))


def mandrel_rule(D, t, R):
    """Mandrel and wiper from the wall factor W = D/t and the bend ratio R/D (tube-bending
    rule of thumb, as in the usual mandrel selection charts)."""
    W, dob = D / t, R / D
    if dob >= 3 and W <= 20:
        return dict(W=round(W, 2), DOB=round(dob, 3), mandrel="plug", balls=0, wiper=False)
    if dob >= 2:
        return dict(W=round(W, 2), DOB=round(dob, 3), mandrel="ball", balls=1, wiper=W > 30)
    if dob >= 1.5:
        balls = 0 if W <= 10 else 2 if W <= 20 else 3 if W <= 40 else 4
        return dict(W=round(W, 2), DOB=round(dob, 3), mandrel="ball" if balls else "plug", balls=balls, wiper=W > 10)
    return dict(W=round(W, 2), DOB=round(dob, 3), mandrel="ball", balls=3 if W <= 20 else 4, wiper=True)


def bend_torque_Nm(p, R):
    m = MATERIALS[BENDING_TEMPER.get(p.material, p.material)]
    D, d = p.grip_d, p.grip_d - 2 * p.wall_grip
    Zp = (D ** 3 - d ** 3) / 6
    return 1.3 * m.tensile_MPa * Zp / 1000     # 1.3: friction in the tools and the mandrel


def _dihedral(t, b0, b1):
    """Rotation (deg) about the tube axis t from bend-plane normal b0 to b1."""
    return math.degrees(math.atan2(np.dot(np.cross(b0, b1), t), np.dot(b0, b1)))


def lra_table(p, side=1, with_springback=True):
    """CNC bending data from the centre outward: feed L to the next bend's tangent,
    rotation R of the tube about its axis from the previous bend plane, bend angle A
    (plus the over-bend for springback), radius. And the XYZ vertex table (intersection of
    the tangent lines) for the drawing."""
    segs = hb.path(p, side)
    rows, feed, prev_b = [], 0.0, None
    for s in segs:
        if s.kind == "line":
            feed += s.length
            continue
        b = np.cross(s.t0, s.n0)
        rot = 0.0 if prev_b is None else _dihedral(np.array(s.t0), prev_b, b)
        sb = springback(p, s.R, s.angle_deg)
        rows.append(dict(bend=s.name, L=round(feed, 3), R_rot=round(rot, 2), A=round(s.angle_deg, 3),
                         A_machine=round(s.angle_deg + (sb["overbend_deg"] if with_springback else 0), 2),
                         radius=s.R, arc=round(s.length, 3)))
        prev_b, feed = b, 0.0
    rows.append(dict(bend="end", L=round(feed, 3)))
    # vertices: tangent-line intersections, from the centre to the end
    pts = [np.zeros(3)]
    for s in segs:
        if s.kind == "arc":
            th = math.radians(s.angle_deg)
            if th < math.pi - 1e-6:
                pts.append(np.array(s.p0) + np.array(s.t0) * s.R * math.tan(th / 2))
            else:                                    # 180 deg: no single vertex
                pts.append(np.array(s.p0) + np.array(s.n0) * s.R)
    pts.append(np.array(segs[-1].end()[0]))
    return rows, [tuple(round(float(c), 3) for c in q) for q in pts]


# ------------------------------------------------------------------ the tube during bending

def bend_order(p):
    return [(side, name) for side in (1, -1) for name in ("B1", "B2", "B3")]


def machine_location(seg):
    """Location of the machine frame (bar coordinates) at the end of bend `seg`."""
    P, t, n = seg.end()
    c = np.array(seg.p0) + seg.R * np.array(seg.n0)
    X = -np.asarray(t); Y = np.asarray(n); Z = np.cross(X, Y)
    return Location(Plane(origin=tuple(c), x_dir=tuple(X), z_dir=tuple(Z)))


def to_machine(seg, pts):
    loc = machine_location(seg)
    pl = Plane(loc)
    M = np.array([[pl.x_dir.X, pl.x_dir.Y, pl.x_dir.Z], [pl.y_dir.X, pl.y_dir.Y, pl.y_dir.Z],
                  [pl.z_dir.X, pl.z_dir.Y, pl.z_dir.Z]])
    o = np.array([pl.origin.X, pl.origin.Y, pl.origin.Z])
    return (np.asarray(pts) - o) @ M.T


def in_process(p, tp, side, name):
    """Centreline segments of the tube at the end of bend (side, name): bends done so far
    in their final shape, the rest of the tube straight. Returns (leading, arc, trailing)
    segment lists, leading ordered from the arc toward the far end of the tube."""
    sol = hb.solve(p)
    order = bend_order(p)
    k = order.index((side, name))
    done = set(order[:k])
    halves = {}
    for s_ in (1, -1):
        segs = hb.path(p, s_, sol)
        out, i = [], 0
        while i < len(segs):
            g = segs[i]
            if g.kind == "arc" and (s_, g.name) not in done and (s_, g.name) != (side, name):
                # unbent from here: one straight to the end (plus the allowance)
                P = np.array(g.p0); t = np.array(g.t0)
                L = sum(x.length for x in segs[i:]) + tp.end_allow
                out.append(hb.Seg("line", tuple(P), tuple(t), L, name="straight"))
                break
            if i == len(segs) - 1 and g.kind == "line":
                g = hb.Seg("line", g.p0, g.t0, g.length + tp.end_allow, name=g.name)
            out.append(g)
            i += 1
        halves[s_] = out
    this = halves[side]
    j = next(i for i, g in enumerate(this) if g.name == name)
    arc = this[j]
    trailing = this[j + 1:]
    # leading: back along this half to the centre, then out along the other half
    leading = list(reversed(this[:j])) + [None] + halves[-side]
    return leading, arc, trailing, halves


def centreline_points(segs, step=3.0):
    """Points along segments in order. In a leading list (in_process) the segments before
    the None marker run back toward the centre, so they are sampled end to start."""
    pts, back = [], None in segs
    for g in segs:
        if g is None:
            back = False
            continue
        n = max(2, int(math.ceil(g.length / step)) + 1)
        us = np.linspace(0, g.length, n)
        pts += [g.point(u)[0] for u in (us[::-1] if back else us)]
    return np.array(pts)


def die_height(p, tp):
    return p.grip_d + 2 * tp.die_rim


def die_radius(p, tp, R):
    return R - p.grip_d / 2 + tp.groove_depth * p.grip_d


def pressure_die_len(p, tp, arc):
    return arc.length + tp.pd_extra_D * p.grip_d


def _clamp_len(p, tp, R=None):
    """Clamp (and insert pocket) length: the grip length, and at least as long as the
    tangent stays inside the bend die's rim, so the leading tube clears the die."""
    Lc = tp.clamp_len_D * p.grip_d
    if R is not None:
        rd = die_radius(p, tp, R)
        Lc = max(Lc, math.sqrt((rd + p.grip_d / 2) ** 2 - R ** 2) + 2.0)
    return Lc


def straight_grip(arc, leading, tol=0.5):
    """Length of the leading centreline from the arc start that stays within tol of the
    straight tangent line (what a straight clamp groove can hold)."""
    P0 = np.array(arc.p0); u = -np.array(arc.t0)
    pts = centreline_points(leading, 0.5)
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    v = pts - P0
    off = np.linalg.norm(v - np.outer(v @ u, u), axis=1)
    bad = (off > tol) | (v @ u < -1e-6)
    return float(sl[np.argmax(bad)]) if bad.any() else float(sl[-1])


def sweep_clearance(p, tp, side, name, up=1, steps=19):
    """Smallest clearance of the swinging leading part (tube surface) to the machine table,
    the carriage, the pressure die and the trailing tube while the die turns through the
    bend. up = +1: machine with its table toward -Z'; -1: the mirrored machine."""
    leading, arc, trailing, _ = in_process(p, tp, side, name)
    R, D = arc.R, p.grip_d
    # leading part beyond the clamp (the clamped length rides with the die, in its pocket)
    pts = centreline_points(leading)
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    pts = pts[sl > _clamp_len(p, tp, R)]
    q = to_machine(arc, pts)
    a = math.radians(arc.angle_deg)
    Hd = die_height(p, tp)
    Lpd = pressure_die_len(p, tp, arc)
    Lt = sum(g.length for g in trailing) if trailing else 0.0
    rd = die_radius(p, tp, R)
    worst = dict(table=1e9, carriage=1e9, pressure_die=1e9, trailing_tube=1e9)
    relief = 0.0
    for phi in np.linspace(0, a, steps):
        c, s = math.cos(phi - a), math.sin(phi - a)
        x = c * q[:, 0] - s * q[:, 1]; y = s * q[:, 0] + c * q[:, 1]; z = q[:, 2]
        # where the leading part is in the groove (on the die), it is not an obstacle
        r = np.hypot(x, y)
        on_die = (np.abs(r - R) < D / 2 + 1.0) & (np.abs(z) < D / 2 + 1.0)
        free = ~on_die
        worst["table"] = min(worst["table"], float((up * z + Hd / 2 + tp.table_gap - D / 2).min()))
        # carriage: box behind the pressure die around the trailing axis
        inside = (x < -Lpd - 20) & (np.abs(y + R) < tp.carriage_half) & (np.abs(z) < tp.carriage_half)
        dc = np.where(inside, -1.0, 1e9)
        worst["carriage"] = min(worst["carriage"], float(dc.min()))
        # pressure die: outer side of the trailing tube, x' in [-Lpd, 0]
        pd = (x > -Lpd) & (x < 0) & (y < -R + D / 2) & (y > -R - D / 2 - tp.tool_wall) & (np.abs(z) < Hd / 2)
        worst["pressure_die"] = min(worst["pressure_die"], float(np.where(pd & free, -1.0, 1e9).min()))
        # trailing tube: the straight still to be bent, x' in [-Lt, 0] on y' = -R
        xs = np.clip(x, -Lt, 0.0)
        dt = np.sqrt((x - xs) ** 2 + (y + R) ** 2 + z ** 2) - D
        far = (x < -D) | (np.hypot(x, y + R) > 2 * D)       # leave the tangent point itself out
        worst["trailing_tube"] = min(worst["trailing_tube"], float(np.where(far, dt, 1e9).min()))
        if phi == 0:
            # bend die body: the leading part turns with the die, so check it once in the die's
            # start frame. The tube's point nearest the die axis must stay outside the rim
            # unless it is in the clamp pocket; otherwise the die gets a relief that deep
            # (bend_die() cuts it).
            Lp = _clamp_len(p, tp, R)
            near = 1 - (D / 2) / np.maximum(r, 1e-9)
            xn, yn = x * near, y * near
            in_pocket = (xn > -1e-6) & (xn < Lp) & (np.abs(yn + R) < D / 2 + tp.tool_wall)
            body = (np.abs(z) < Hd / 2 + D / 2) & free & ~in_pocket
            if body.any():
                relief = max(relief, float((rd - (r[body] - D / 2)).max()))
    out = {k: round(v, 2) if v < 1e8 else None for k, v in worst.items()}
    out["die_relief"] = round(max(relief, 0.0), 2)
    return out


def bending_plan(p, tp=TubeProcessParams()):
    """Every bend in machine order with its checks' inputs."""
    out = []
    for side, name in bend_order(p):
        leading, arc, trailing, _ = in_process(p, tp, side, name)
        Lc = tp.clamp_len_D * p.grip_d
        grip = straight_grip(arc, leading)
        contoured = grip < Lc - 1e-6
        # grip available when the clamp pair follows the leading part in its split plane
        _, follow = clamp_path(p, tp, arc, leading, Lc)
        trail = sum(g.length for g in trailing)
        sweeps = {u: sweep_clearance(p, tp, side, name, up=u) for u in (1, -1)}
        ok = {u: all(v is None or v >= 0 for k, v in s.items() if k != "die_relief") for u, s in sweeps.items()}
        hand = 1 if ok[1] else -1 if ok[-1] else 0
        out.append(dict(side="R" if side > 0 else "L", bend=name, R=arc.R, angle=arc.angle_deg,
                        clamp_needed=round(Lc, 2), clamp_len=round(_clamp_len(p, tp, arc.R), 2),
                        straight_grip=round(grip, 2), contoured_clamp=bool(contoured),
                        contoured_grip=round(min(follow, Lc), 2),
                        trailing_straight=round(trail, 2), pressure_die=round(pressure_die_len(p, tp, arc), 2),
                        sweep=sweeps, machine_up=hand, springback=springback(p, arc.R, arc.angle_deg),
                        mandrel=mandrel_rule(p.grip_d, p.wall_grip, arc.R),
                        torque_Nm=round(bend_torque_Nm(p, arc.R), 1)))
    return out


# ------------------------------------------------------------------ tool geometry (machine frame)

def tube_solid(p, segs, ro=None, ri=None):
    """Hollow (or, ri = 0, solid) tube along centreline segments of the grip section."""
    ro = p.grip_d / 2 if ro is None else ro
    ri = p.grip_d / 2 - p.wall_grip if ri is None else ri
    part = None
    for g in segs:
        if g is None or g.length < 1e-6:
            continue
        sol = hb._seg_solid(g, ro, ri) if ri > 0 else _solid_seg(g, ro)
        part = sol if part is None else part + sol
    return part


def _solid_seg(g, r):
    from build123d import Circle, extrude
    pl = hb._plane(g.p0, g.t0)
    face = pl * Circle(r)
    if g.kind == "line":
        return extrude(face, g.length, dir=Vector(*g.t0))
    c = np.array(g.p0) + g.R * np.array(g.n0)
    return revolve(face, Axis(tuple(c), tuple(np.cross(g.t0, g.n0))), g.angle_deg)


def in_process_shape(p, tp, side, name):
    """The tube at the end of bend (side, name), hollow, in bar coordinates."""
    leading, arc, trailing, halves = in_process(p, tp, side, name)
    sol = hb.solve(p)
    part = hb.tops_solid(p, sol.L0)
    for s_ in (1, -1):
        for g in halves[s_]:
            if g.name == "L0":
                continue
            part = part + hb._seg_solid(g, p.grip_d / 2, p.grip_d / 2 - p.wall_grip)
    return part


def _machine(shape, seg):
    return shape.moved(machine_location(seg).inverse())


DIE_RELIEF_CLR = 1.0            # relief in the bend die around a leading part that comes inside its rim


def bend_die(p, tp, R):
    """Disc with the tube groove (torus) at radius R on the machine spindle (axis Z'),
    in its start position: tangent point (0, -R, 0), the clamp-insert pocket along +X'
    from there (the insert and the clamp die close on the tube in it), and a relief for
    every set-up whose leading part comes inside the rim (cut by the leading tube, in the
    die's frame, grown by DIE_RELIEF_CLR)."""
    D, c = p.grip_d, tp.groove_clr
    Hd = die_height(p, tp)
    rd = die_radius(p, tp, R)
    Lp = _clamp_len(p, tp, R)
    pocket = Pos(Lp / 2, -R, 0) * Box(Lp, 2 * (D / 2 + tp.tool_wall), Hd + 2)
    die = Cylinder(rd, Hd) - Torus(R, D / 2 + c) - Cylinder(tp.die_hub_d / 2, Hd + 2) - pocket
    for side, name in bend_order(p):
        if side < 0:
            continue                                     # the left half mirrors the right (other hand)
        leading, arc, _, _ = in_process(p, tp, side, name)
        if arc.R != R or sweep_clearance(p, tp, side, name, steps=2)["die_relief"] <= 0:
            continue
        t = _machine(tube_solid(p, _leading_solid_segs(leading), p.grip_d / 2 + DIE_RELIEF_CLR, 0.0), arc)
        die = die - Rot(0, 0, -arc.angle_deg) * t
    return die.clean()


def _outer_tube(p, tp, segs, extra=0.0):
    return tube_solid(p, segs, p.grip_d / 2 + tp.groove_clr + extra, 0.0)


def _arc_start_frame(arc):
    """Arc start (where the clamp begins) in machine coordinates at the end of the bend:
    point, direction along the leading part, direction away from the die axis."""
    th = -math.pi / 2 + math.radians(arc.angle_deg)
    P = np.array([arc.R * math.cos(th), arc.R * math.sin(th), 0.0])
    return P, np.array([-math.sin(th), math.cos(th), 0.0]), np.array([math.cos(th), math.sin(th), 0.0])


def clamp_path(p, tp, arc, leading, L):
    """Leading centreline over the first L from the arc start, machine coordinates, and
    how far it stays in the clamp split plane (the plane through the tube axis normal to
    the die radius): a contoured clamp pair can follow it that far."""
    pts = centreline_points(leading, 0.5)
    sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    q = to_machine(arc, pts)
    P, u, rad = _arc_start_frame(arc)
    off = np.abs((q - P) @ rad)
    out = off > 0.5
    in_plane = float(sl[np.argmax(out)]) if out.any() else float(sl[-1])
    return q[sl <= L + 1e-9], in_plane


def clamp_pair(p, tp, side, name):
    """Clamp die (outer side) and insert (die side) for bend (side, name) in machine
    coordinates: a block around the leading centreline over the clamp length, split on the
    plane through the tube axis normal to the die radius, minus the leading tube. A straight
    leading part gives a straight pair; one that curves in the split plane (B2: the B1 arc
    rises out of the bending plane) gives a contoured pair whose groove follows it."""
    leading, arc, _, _ = in_process(p, tp, side, name)
    D, Lc, Hd, w = p.grip_d, _clamp_len(p, tp, arc.R), die_height(p, tp), tp.tool_wall
    P, u, rad = _arc_start_frame(arc)
    q, _ = clamp_path(p, tp, arc, leading, Lc)
    zs = q[:, 2]
    z0, z1 = min(zs.min() - D / 2 - w, -Hd / 2), max(zs.max() + D / 2 + w, Hd / 2)
    L = max(float(((q - P) @ u).max()), 1.0)
    half = D / 2 + w
    ang = math.degrees(math.atan2(u[1], u[0]))
    zc, H = (z0 + z1) / 2, z1 - z0
    tube = _machine(_outer_tube(p, tp, _leading_solid_segs(leading)), arc)
    clamp = Pos(*(P + u * L / 2 + rad * half / 2 + np.array([0, 0, zc]))) * Rot(0, 0, ang) * Box(L, half, H)
    insert = Pos(*(P + u * L / 2 - rad * half / 2 + np.array([0, 0, zc]))) * Rot(0, 0, ang) * Box(L, half, H)
    return (clamp - tube).clean(), (insert - tube).clean(), dict(P=P, u=u, rad=rad, Lc=Lc, L=L, z=(z0, z1))


def _leading_solid_segs(leading, n=3):
    """The first few leading segments next to the arc (enough to cut the clamp groove)."""
    back = []
    for g in leading:
        if g is None:
            break
        back.append(g)
    return back[:n] if back else [g for g in leading if g is not None][:n]


def pressure_die(p, tp, arc):
    D, Hd, w = p.grip_d, die_height(p, tp), tp.tool_wall
    L = pressure_die_len(p, tp, arc)
    R = arc.R
    h = D / 2 + w - tp.pd_gap
    blk = Pos(-L / 2, -R - tp.pd_gap - h / 2, 0) * Box(L, h, Hd)
    groove = Pos(-L / 2, -R, 0) * Rot(0, 90, 0) * Cylinder(D / 2 + tp.groove_clr, L + 2)
    return (blk - groove).clean()


def wiper_die(p, tp, arc):
    """Wiper on the die side of the trailing tube; its tip runs into the die groove up to
    the tangent point and is cut by the die (feather edge)."""
    D, Hd = p.grip_d, die_height(p, tp)
    R = arc.R
    L = tp.wiper_len_D * D
    tip = 0.15 * D
    wb = D / 2 + 10.0
    blk = Pos(-L / 2 + tip, -R + wb / 2, 0) * Box(L + 0.0, wb, Hd)
    groove = Pos(-L / 2, -R, 0) * Rot(0, 90, 0) * Cylinder(D / 2 + tp.groove_clr, L + 2 * tip + 2)
    die_env = Cylinder(die_radius(p, tp, R) + 0.05, Hd + 2) - Torus(R, D / 2 + tp.groove_clr)
    w = blk - groove - die_env - Torus(R, D / 2 + tp.groove_clr)
    # keep the piece behind the tangent point (the solid on x' < tip)
    return max(w.solids(), key=lambda s: s.volume)


def mandrel(p, tp, arc, mr):
    """Ball mandrel at the end of the bend: shank up to the tangent point, balls along the
    bent arc. Returns (shank, [balls])."""
    d = p.grip_d - 2 * p.wall_grip - 2 * tp.mandrel_clr
    R = arc.R
    L_shank = 160.0
    shank = Pos(-L_shank / 2, -R, 0) * Rot(0, 90, 0) * Cylinder(d / 2, L_shank)
    balls = []
    pitch = tp.ball_pitch_d * d
    for i in range(mr["balls"]):
        s = (i + 1) * pitch                              # along the arc from the tangent point
        th = -math.pi / 2 + s / R
        balls.append(Pos(R * math.cos(th), R * math.sin(th), 0) * Sphere(d / 2 - 0.05))
    return shank, balls


def swage_die(p, tp):
    """One of the two rotary-swaging die halves (split on Z = 0): block with the half
    bore of the swaged outside: entry bell, taper, sizing land. Axis along Y, the clamp
    zone toward -Y."""
    Rc, Rg = p.clamp_d / 2, p.grip_d / 2
    L = tp.swage_entry + p.taper_len + tp.swage_land
    W, H = tp.swage_block[0], tp.swage_block[1]
    # profile (r, y): entry bell at the clamp side, the taper, the land
    y0 = -tp.swage_entry - p.taper_len / 2
    pts = [(0, 0, y0), (Rc + 1.5, 0, y0), (Rc + 0.05, 0, y0 + tp.swage_entry),
           (Rg + 0.0, 0, y0 + tp.swage_entry + p.taper_len), (Rg, 0, y0 + L), (0, 0, y0 + L)]
    bore = Rot(-90, 0, 0) * revolve(make_face(Polyline(*pts, close=True)), Axis.Z, 360)
    blk = Pos(0, y0 + L / 2, H / 2) * Box(W, L, H)
    return (blk - bore).clean()


def swage_mandrel(p, tp):
    """Stepped swaging mandrel, axis along Y, nose at the end of the outside taper (y = b),
    toward +Y the butt, the step and the grip bore, then the holder shank."""
    a, b, n, c, d = hb.zones(p)
    Rg = p.grip_d / 2
    clr = tp.mandrel_clr
    sol = hb.solve(p)
    t0 = hb.sunk_wall(p, p.grip_d)
    r_tip = Rg - t0 - clr
    pts = [(0, 0, b - 2), (r_tip - 0.6, 0, b - 2), (r_tip, 0, b), (Rg - p.wall_butt - clr, 0, n),
           (Rg - p.wall_butt - clr, 0, c), (Rg - p.wall_grip - clr, 0, d),
           (Rg - p.wall_grip - clr, 0, sol.L0 + 40.0), (0, 0, sol.L0 + 40.0)]
    return Rot(-90, 0, 0) * revolve(make_face(Polyline(*pts, close=True)), Axis.Z, 360)


def tool_set(p, tp=TubeProcessParams(), setup=(1, "B2")):
    """Bending tools and the in-process tube for one set-up, in machine coordinates, plus
    the straight clamp pair (from B1) and the swaging tools."""
    side, name = setup
    leading, arc, trailing, _ = in_process(p, tp, side, name)
    plan = {(1 if b["side"] == "R" else -1, b["bend"]): b for b in bending_plan(p, tp)}
    mr = plan[setup]["mandrel"]
    die = Rot(0, 0, arc.angle_deg) * bend_die(p, tp, arc.R)      # turned through the bend
    clamp, insert, cinfo = clamp_pair(p, tp, side, name)
    s_clamp, s_insert, _ = clamp_pair(p, tp, 1, "B1")
    pd = pressure_die(p, tp, arc)
    wp = wiper_die(p, tp, arc)
    shank, balls = mandrel(p, tp, arc, mr)
    tube = _machine(in_process_shape(p, tp, side, name), arc)
    return dict(bend_die=die, clamp=clamp, insert=insert, clamp_straight=s_clamp, insert_straight=s_insert,
                pressure_die=pd, wiper=wp, mandrel_shank=shank, mandrel_balls=balls, tube=tube, arc=arc,
                clamp_info=cinfo, swage_die=swage_die(p, tp), swage_mandrel=swage_mandrel(p, tp), plan=plan)


# ------------------------------------------------------------------ tool parts for the drawings

def mandrel_part(p, tp, n_balls):
    """The ball mandrel as one straight part (as it is made and inspected): shank, then the
    balls at the ball pitch, each joined to the one before by a link; the last ball's nose
    is cut flat at the link diameter. Made as one revolved profile (the ball-and-socket
    chain drawn as one solid; link detail by the mandrel maker). Axis along X, shank end
    at X = 0."""
    d = p.grip_d - 2 * p.wall_grip - 2 * tp.mandrel_clr
    L_shank = 160.0
    rs, rb, rl = d / 2, d / 2 - 0.05, 0.35 * d
    h = math.sqrt(rb ** 2 - rl ** 2)              # half the axial length of a ball between its links
    pitch = tp.ball_pitch_d * d
    edges = [Line((0, 0), (0, rs)), Line((0, rs), (L_shank, rs)), Line((L_shank, rs), (L_shank, rl))]
    x = L_shank
    for i in range(n_balls):
        xc = L_shank + (i + 1) * pitch
        edges.append(Line((x, rl), (xc - h, rl)))
        edges.append(ThreePointArc((xc - h, rl), (xc, rb), (xc + h, rl)))
        x = xc + h
    edges += [Line((x, rl), (x, 0)), Line((x, 0), (0, 0))]
    return revolve(make_face(Wire(edges)), Axis.X, 360)


def _to_drawing(shape, rot_deg=0.0, axis=(0, 0, 1)):
    """Turn a tool about the given axis, then place it with its bounding box centred in X
    and Z and its top face at Y = 0 (the face toward the top of the sheet)."""
    s = shape if not rot_deg else shape.rotate(Axis((0, 0, 0), axis), rot_deg)
    bb = s.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -bb.max.Y, -(bb.min.Z + bb.max.Z) / 2) * s


def tool_parts(p, tp=TubeProcessParams(), tools=None, setup=(1, "B2")):
    """Each tool part in its own drawing frame, with the block size the design asks for
    (X, Y, Z: what the drawing's overall dimensions are checked against), its material and
    its drilled holes [(use, diameter, axis, count)]."""
    tools = tools or tool_set(p, tp, setup)
    arc = tools["arc"]
    R, D, Hd, w = arc.R, p.grip_d, die_height(p, tp), tp.tool_wall
    rd = die_radius(p, tp, R)
    out = {}
    out["bend_die"] = dict(shape=_to_drawing(bend_die(p, tp, R)), block=(2 * rd, 2 * rd, Hd), material=TOOL_STEEL,
                           holes=[("主軸穴", tp.die_hub_d, "Z", 1)], density=STEEL_DENSITY)
    for key, src in (("clamp", "clamp"), ("insert", "insert"), ("clamp_straight", "clamp_straight"),
                     ("insert_straight", "insert_straight")):
        info = tools["clamp_info"] if key in ("clamp", "insert") else clamp_pair(p, tp, 1, "B1")[2]
        ang = math.degrees(math.atan2(info["u"][1], info["u"][0]))
        z0, z1 = info["z"]
        out[key] = dict(shape=_to_drawing(tools[src], -ang), block=(info["L"], D / 2 + w, z1 - z0), material=TOOL_STEEL,
                        holes=[], density=STEEL_DENSITY)
    L = pressure_die_len(p, tp, arc)
    out["pressure_die"] = dict(shape=_to_drawing(tools["pressure_die"]), block=(L, D / 2 + w - tp.pd_gap, Hd),
                               material=TOOL_STEEL, holes=[], density=STEEL_DENSITY)
    wb = tools["wiper"].bounding_box()
    out["wiper"] = dict(shape=_to_drawing(tools["wiper"]), block=(wb.size.X, wb.size.Y, Hd), material=WIPER_MAT,
                        holes=[], density=7.6)
    mr = tools["plan"][setup]["mandrel"]
    mp = mandrel_part(p, tp, mr["balls"])
    mb = mp.bounding_box()
    out["mandrel"] = dict(shape=_to_drawing(mp), block=(mb.size.X, mb.size.Y, mb.size.Z), material=MANDREL_MAT,
                          holes=[], density=STEEL_DENSITY)
    # swaging tools: axis along Y in their own frame; drawn with the axis along X
    W, H = tp.swage_block[0], tp.swage_block[1]
    Ls = tp.swage_entry + p.taper_len + tp.swage_land
    out["swage_die"] = dict(shape=_to_drawing(tools["swage_die"], 90), block=(Ls, W, H), material=SWAGE_MAT, holes=[],
                            density=STEEL_DENSITY)
    sm = _to_drawing(tools["swage_mandrel"], 90)
    smb = sm.bounding_box()
    out["swage_mandrel"] = dict(shape=sm, block=(smb.size.X, smb.size.Y, smb.size.Z), material=MANDREL_MAT, holes=[],
                                density=STEEL_DENSITY)
    return out
