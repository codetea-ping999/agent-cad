"""Parametric road drop handlebar: a butted, swaged and bent aluminium tube.

The shape is defined the way the tube is made (agent_cad/process/tube_bending.py). A
seamless drawn tube of the clamp diameter is rotary-swaged down to the grip diameter
at both ends, then bent on a rotary-draw bender. Through the outside taper the tube is
sunk without a mandrel, so its wall thickens as the diameter falls (wall ~ D^-sink_exp);
beyond it a mandrel inserted from the open end sets the bore: a short nose cone, the
butt, a step up to the grip bore. The bore only grows from the nose toward the end, so
the mandrel can be pulled out. Each half is a bend table from the centre outward:

    L0 straight (tops, from the centre)  - includes clamp zone, swaged taper and butt
    B1 shoulder bend, R1, 90 deg + sweep  (turns the tops forward)
    L1 straight (hoods)
    B2 hood bend, R2, a2 deg              (turns down; its plane tilted outward by the flare)
    L2 straight (front of the drop)
    B3 drop bend, R3, 180 - a2 deg        (turns back)
    L3 straight (drops, to the end)

Axes as the stem: X forward, Y to the rider's right, Z up; the bar axis is Y through
the origin (the stem's bar-clamp centre). width, reach and drop are what the rider
buys; L0, L1 and L2 are solved so the centreline meets them:

    width  centre-to-centre of the tube at the drop ends
    reach  clamp axis to the most forward point of the centreline
    drop   tops centreline to the drop-end centreline (vertical)

    p = HandlebarParams(width=400, material="A6061-T6")
    bar = build(p)
"""
import math
from dataclasses import dataclass, asdict, replace
import numpy as np
from build123d import (Plane, Circle, Axis, Rot, Line, ThreePointArc, Spline, Wire, make_face, revolve, extrude,
                       Vector)
from ..materials import MATERIALS

__all__ = ["HandlebarParams", "build", "solve", "path", "density", "asdict", "GEOMETRY_REV", "mass_g"]

# Revision of the geometry build() makes; part of the document fingerprints (tests pin the
# default model's volume to it). Bump it whenever build() returns a different shape for the
# same parameters.
GEOMETRY_REV = 1


@dataclass(frozen=True)
class HandlebarParams:
    # fit (what the rider buys)
    width: float = 420.0          # c-c at the drop ends
    reach: float = 80.0
    drop: float = 125.0
    flare_deg: float = 0.0        # drops tilted outward in front view (B2/B3 bend plane)
    sweep_deg: float = 0.0        # tops swept back: B1 = 90 + sweep
    # interface
    clamp_d: float = 31.8         # stem bar clamp (ISO 4210-5: 31.8 / 26.0 / 35.0)
    clamp_zone_w: float = 110.0   # full-diameter zone, symmetric; stem + clip-on / light mounts
    grip_d: float = 23.8          # brake-lever band and bar-end diameter
    # swaged taper and internal butting
    taper_len: float = 40.0       # outside: clamp_d -> grip_d (cone), per side
    taper_blend_r: float = 60.0   # tangent arcs at both ends of the cone (profile radius)
    wall_clamp: float = 1.9       # in the clamp zone: the wall of the drawn tube
    sink_exp: float = 0.45        # free sinking in the taper: wall = wall_clamp (clamp_d / D)^sink_exp
    nose_len: float = 8.0         # swaging mandrel nose cone: sunk wall -> wall_butt
    wall_butt: float = 1.4        # mandrel butt, over butt_len
    butt_len: float = 30.0
    butt_taper_len: float = 15.0  # mandrel step butt -> grip bore
    wall_grip: float = 1.15       # through the bends and the drops
    # bends (centreline radii) and the drops
    r_shoulder: float = 40.0      # B1
    r_hood: float = 40.0          # B2
    r_drop: float = 40.0          # B3
    hood_bend_deg: float = 90.0   # a2; B3 = 180 - a2 so the drop ends run horizontally back
    drop_len: float = 85.0        # L3
    material: str = "A7075-T6"
    density_g_cm3: float = 0.0    # 0 -> from the material table
    mass_target_g: float = 285.0


def density(p):
    return p.density_g_cm3 or MATERIALS[p.material].density_g_cm3


# ------------------------------------------------------------------ the bend table and the centreline

@dataclass(frozen=True)
class Seg:
    kind: str                     # "line" | "arc"
    p0: tuple                     # start point
    t0: tuple                     # unit tangent at the start
    length: float                 # arc length along the centreline
    n0: tuple = (0, 0, 0)         # arc: unit normal toward the centre of curvature
    R: float = 0.0
    angle_deg: float = 0.0
    name: str = ""

    def point(self, s):
        p0, t0 = np.array(self.p0), np.array(self.t0)
        if self.kind == "line":
            return p0 + s * t0, t0
        n0 = np.array(self.n0)
        th = s / self.R
        c = p0 + self.R * n0
        return c - self.R * n0 * math.cos(th) + self.R * t0 * math.sin(th), t0 * math.cos(th) + n0 * math.sin(th)

    def end(self):
        p, t = self.point(self.length)
        if self.kind == "line":
            return p, t, np.array(self.n0)
        th = math.radians(self.angle_deg)
        n0, t0 = np.array(self.n0), np.array(self.t0)
        return p, t, n0 * math.cos(th) - t0 * math.sin(th)


def bend_angles(p):
    return 90.0 + p.sweep_deg, p.hood_bend_deg, 180.0 - p.hood_bend_deg


def _half(p, L, side=1):
    """Segments of one half (side +1 right, -1 left) from the centre outward, for the
    straight lengths L = (L0, L1, L2, L3)."""
    a1, a2, a3 = bend_angles(p)
    phi = math.radians(p.flare_deg)
    segs = []
    P, t = np.zeros(3), np.array([0.0, side, 0.0])
    n_b1 = np.array([1.0, 0.0, 0.0])            # B1 turns the tops forward (+X)

    def line(L_, name):
        nonlocal P
        if L_ > 1e-9:
            segs.append(Seg("line", tuple(P), tuple(t), L_, name=name))
            P = P + L_ * t

    def arc(n, R, a, name):
        nonlocal P, t
        s = Seg("arc", tuple(P), tuple(t), R * math.radians(a), tuple(n), R, a, name)
        segs.append(s)
        P, t, _ = s.end()
        return s

    line(L[0], "L0")
    s1 = arc(n_b1, p.r_shoulder, a1, "B1")
    line(L[1], "L1")
    # B2 plane: from "turn down" (-Z) tilted outward by the flare, about the current tangent
    down = np.array([0.0, 0.0, -1.0])
    outward = np.array([0.0, side, 0.0])
    n2 = down * math.cos(phi) + outward * math.sin(phi)
    n2 = n2 - np.dot(n2, t) * t                  # keep it normal to the tangent (sweep != 0)
    n2 /= np.linalg.norm(n2)
    s2 = arc(n2, p.r_hood, a2, "B2")
    line(L[2], "L2")
    _, _, n3 = s2.end()                          # same plane, continuing the turn
    arc(n3, p.r_drop, a3, "B3")
    line(L[3], "L3")
    return segs


def _measures(p, L):
    segs = _half(p, L, 1)
    pts = np.array([s.point(u)[0] for s in segs for u in np.linspace(0, s.length, 41)])
    end = np.array(segs[-1].end()[0])
    return np.array([2 * end[1], pts[:, 0].max(), -end[2]])


@dataclass(frozen=True)
class Solution:
    L0: float
    L1: float
    L2: float
    L3: float


def solve(p):
    """Straight lengths so that the centreline has p.width, p.reach and p.drop
    (Newton on the three unknowns; width ~ L0, reach ~ L1, drop ~ L2)."""
    target = np.array([p.width, p.reach, p.drop])
    L = np.array([p.width / 2 - p.r_shoulder, 0.0, p.drop - p.r_hood - p.r_drop])
    for _ in range(30):
        f = _measures(p, (*L, p.drop_len)) - target
        if np.abs(f).max() < 1e-7:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            d = np.zeros(3); d[k] = 1e-4
            J[:, k] = (_measures(p, (*(L + d), p.drop_len)) - _measures(p, (*L, p.drop_len))) / 1e-4
        L = L - np.linalg.solve(J, f)
    return Solution(float(L[0]), float(L[1]), float(L[2]), float(p.drop_len))


def path(p, side=1, sol=None):
    sol = sol or solve(p)
    return _half(p, (sol.L0, sol.L1, sol.L2, sol.L3), side)


def centreline_length(p, sol=None):
    return 2 * sum(s.length for s in path(p, 1, sol))


# ------------------------------------------------------------------ section along the tops (swaged taper + butt)

def zones(p):
    """Axial stations |y| along the tops (from the centre): clamp zone end, outside taper
    end, mandrel nose end, butt end, grip bore from."""
    a = p.clamp_zone_w / 2
    b = a + p.taper_len
    n = b + p.nose_len
    c = n + p.butt_len
    d = c + p.butt_taper_len
    return a, b, n, c, d


def sunk_wall(p, D):
    return p.wall_clamp * (p.clamp_d / D) ** p.sink_exp


def radii_at(p, y):
    """Outside and inside radius at |y| along the straight tops (the outside cone blends
    are left out)."""
    a, b, n, c, d = zones(p)
    Rc, Rg = p.clamp_d / 2, p.grip_d / 2
    y = abs(y)
    if y <= a:
        return Rc, Rc - p.wall_clamp
    if y <= b:
        ro = Rc + (y - a) / p.taper_len * (Rg - Rc)
        return ro, ro - sunk_wall(p, 2 * ro)
    if y <= n:
        t0 = sunk_wall(p, p.grip_d)
        return Rg, Rg - (t0 + (y - b) / p.nose_len * (p.wall_butt - t0))
    if y <= c:
        return Rg, Rg - p.wall_butt
    if y <= d:
        return Rg, Rg - (p.wall_butt + (y - c) / p.butt_taper_len * (p.wall_grip - p.wall_butt))
    return Rg, Rg - p.wall_grip


def _blend(pts, i, rho):
    """Replace the corner pts[i] of a polyline by a tangent arc of radius rho (sampled)."""
    p0, p1, p2 = (np.array(pts[k]) for k in (i - 1, i, i + 1))
    u, v = (p0 - p1) / np.linalg.norm(p0 - p1), (p2 - p1) / np.linalg.norm(p2 - p1)
    turn = math.pi - math.acos(np.clip(np.dot(u, v), -1, 1))
    if turn < 1e-6:
        return [tuple(p1)]
    tl = rho * math.tan(turn / 2)
    a, b = p1 + u * tl, p1 + v * tl
    bis = (u + v) / np.linalg.norm(u + v)
    c = p1 + bis * math.hypot(rho, tl)
    out = []
    for w in np.linspace(0, 1, 9):
        q = a + (b - a) * w
        out.append(tuple(c + (q - c) / np.linalg.norm(q - c) * rho))
    return out


def tops_profile(p, L0):
    """Closed (y, r) profile of the hollow tops from -L0 to +L0 as edges, in order:
    ("line", p0, p1) | ("arc", p0, mid, p1) | ("spline", [points]). Outside: clamp zone,
    cones with tangent arcs at both ends, grip; inside (back from +L0): grip bore, step,
    butt, mandrel nose, the sunk taper (a spline through the sinking law), clamp bore."""
    a, b, n, c, d = zones(p)
    Rc, Rg = p.clamp_d / 2, p.grip_d / 2
    outer = [(-L0, Rg), (-b, Rg), (-a, Rc), (a, Rc), (b, Rg), (L0, Rg)]
    edges, prev = [], outer[0]
    for i in range(1, len(outer) - 1):
        arc = _blend(outer, i, p.taper_blend_r)
        edges.append(("line", prev, arc[0]))
        edges.append(("arc", arc[0], arc[len(arc) // 2], arc[-1]))
        prev = arc[-1]
    edges.append(("line", prev, outer[-1]))
    ri = lambda y: (y, radii_at(p, y)[1])
    edges.append(("line", outer[-1], ri(L0)))
    taper = list(np.linspace(b, a, 7))
    for sgn, ys in ((1, [L0, d, c, n]), (-1, [n, c, d, L0])):
        if sgn < 0:
            edges.append(("line", ri(a), ri(-a)))
            edges.append(("spline", [ri(-y) for y in taper[::-1]]))
            pts = [ri(-y) for y in ys]
            for q0, q1 in zip([ri(-b)] + pts[:-1], pts):
                edges.append(("line", q0, q1))
        else:
            pts = [ri(y) for y in ys]
            for q0, q1 in zip(pts[:-1], pts[1:]):
                edges.append(("line", q0, q1))
            edges.append(("line", pts[-1], ri(b)))
            edges.append(("spline", [ri(y) for y in taper]))
    edges.append(("line", ri(-L0), outer[0]))
    return edges


# ------------------------------------------------------------------ solid

def _annulus(plane, ro, ri):
    return plane * Circle(ro) - plane * Circle(ri)


def _plane(P, t):
    t = Vector(*t)
    x = Vector(0, 0, 1) if abs(t.Z) < 0.9 else Vector(1, 0, 0)
    x = (x - t * x.dot(t)).normalized()
    return Plane(origin=Vector(*P), x_dir=x, z_dir=t)


def _seg_solid(s, ro, ri):
    if s.kind == "line":
        # built from its far end back to its start: fused onto the left end of the tops
        # the other way round, OCCT returns an invalid solid
        P1 = np.array(s.p0) + s.length * np.array(s.t0)
        back = tuple(-np.array(s.t0))
        return extrude(_annulus(_plane(tuple(P1), back), ro, ri), s.length, dir=Vector(*back))
    pl = _plane(s.p0, s.t0)
    face = _annulus(pl, ro, ri)
    c = np.array(s.p0) + s.R * np.array(s.n0)
    ax = np.cross(s.t0, s.n0)
    return revolve(face, Axis(tuple(c), tuple(ax)), s.angle_deg)


def tops_solid(p, L0):
    # profile drawn in the XZ plane as (x = r, z = axial), revolved about Z, then Z -> Y
    P = lambda q: Vector(q[1], 0, q[0])
    es = []
    for e in tops_profile(p, L0):
        if e[0] == "line":
            if np.hypot(e[2][0] - e[1][0], e[2][1] - e[1][1]) > 1e-9:
                es.append(Line(P(e[1]), P(e[2])))
        elif e[0] == "arc":
            es.append(ThreePointArc(P(e[1]), P(e[2]), P(e[3])))
        else:
            es.append(Spline(*[P(q) for q in e[1]]))
    face = make_face(Wire(es))
    sol = revolve(face, Axis.Z, 360)
    return Rot(-90, 0, 0) * sol


def build(p: HandlebarParams):
    sol = solve(p)
    Rg = p.grip_d / 2
    ri = Rg - p.wall_grip
    part = tops_solid(p, sol.L0)
    for side in (1, -1):
        for s in path(p, side, sol)[1:]:
            part = part + _seg_solid(s, Rg, ri)
    return part.clean()


def mass_g(p, shape):
    return shape.volume * density(p) / 1000.0


def checks(p, shape=None):
    """Geometry rules that do not need a FEA: interface sizes, zones fit on the tops,
    lengths are physical, mass target."""
    sol = solve(p)
    a, b, n, c, d = zones(p)
    out = dict(solution=asdict(sol), centreline_mm=round(centreline_length(p, sol), 2), zones=dict(
        clamp_half=a, taper_end=b, nose_end=n, butt_end=c, grip_bore_from=d),
        wall_taper_end=round(sunk_wall(p, p.grip_d), 3))
    rules = []

    def rule(name, ok, value, limit, basis):
        rules.append({"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis})
    rule("geometry.tops_hold_butting", d <= sol.L0 - 5.0, round(d, 2), round(sol.L0 - 5.0, 2),
         "swaged taper and butt end >= 5 mm before the shoulder bend tangent")
    rule("geometry.hood_straight", sol.L1 >= -1e-6, round(sol.L1, 3), 0.0, "reach reachable with R1 + R2")
    rule("geometry.drop_straight", sol.L2 >= -1e-6, round(sol.L2, 3), 0.0, "drop reachable with R2 + R3")
    t0 = sunk_wall(p, p.grip_d)
    rule("process.mandrel_withdrawable", t0 >= p.wall_butt >= p.wall_grip, [round(t0, 3), p.wall_butt, p.wall_grip],
         "sunk >= butt >= grip wall", "swaging mandrel pulled out through the end: the bore must not shrink outward")
    if shape is not None:
        m = mass_g(p, shape)
        out["mass_g"] = round(m, 1)
        rule("geometry.valid_solid", shape.is_valid and len(shape.solids()) == 1, len(shape.solids()), 1,
             "one valid closed solid")
        rule("mass.total_g", m <= p.mass_target_g, round(m, 1), p.mass_target_g, "parameter mass_target_g")
    out["rules"] = rules
    out["pass"] = all(r["pass"] for r in rules)
    return out
