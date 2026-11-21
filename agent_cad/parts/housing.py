"""Injection-moulded electrical housing (cover): a drafted open box with cross ribs,
four screw bosses and a connector window in the front (-Z) wall.

The mould opens along Y. The parting plane is the rim (Y = 0) and the box stands on
it with its opening toward -Y: the outside is formed by the fixed half (cavity) and
withdraws toward +Y, so the outer walls taper inward going up; the inside, ribs and
bosses are formed by the moving half (core) and withdraw toward -Y, so the pocket
narrows going up and ribs and bosses widen toward the floor. The window is an
undercut for both halves: a slide pulls it out along -Z, so it narrows from the
outer face toward the inside.

Sizes given at the rim (L, W) or at the face a feature starts from (boss and hole
diameters at the boss tip, window at the outer face); the draft runs from there.

    part = build(HousingParams())
"""
import math
from dataclasses import dataclass, asdict
from shapely.geometry import box
from build123d import Rot, Pos, Cone, Align, Plane, RectangleRounded, extrude
from ..materials import MATERIALS
from ..process.forging import drafted_prism, draft_sides

__all__ = ["HousingParams", "build", "density", "asdict", "GEOMETRY_REV"]

# Revision of the geometry build() makes; part of the document number (scripts/make_molding.py).
# Bump it whenever build() returns a different shape for the same parameters
# (tests/test_molding.py pins the default model's fingerprint to it).
GEOMETRY_REV = 1


@dataclass(frozen=True)
class HousingParams:
    length: float = 100.0         # X, outside at the rim
    width: float = 70.0           # Z, outside at the rim
    height: float = 30.0          # Y, rim to top
    wall: float = 2.5             # walls and floor (nominal)
    corner_r: float = 6.0         # outside vertical corners
    draft_deg: float = 1.0        # outer walls (cavity) and pocket walls (core)
    rib_t: float = 1.25           # at the root (floor); ~0.5 x wall against sink marks
    rib_h: float = 7.0            # below the floor; <= 3 x wall
    rib_draft_deg: float = 0.5    # per side
    boss_pitch_x: float = 76.0
    boss_pitch_z: float = 46.0
    boss_d: float = 8.0           # at the tip; 2 x screw diameter
    boss_recess: float = 1.0      # boss tips below the rim (toward the floor)
    hole_d: float = 3.4           # at the boss tip: thread-forming screw, nominal 4 mm
    hole_depth: float = 12.0
    boss_draft_deg: float = 0.5   # boss outside and hole
    window_w: float = 24.0        # X, at the outer face
    window_h: float = 10.0        # Y, at the outer face
    window_x: float = -18.0       # centre
    window_y: float = 12.0        # centre above the rim
    window_draft_deg: float = 1.0  # toward the slide's pull direction (-Z)
    material: str = "PA66-GF30"
    mass_target_g: float = 90.0


def density(p):
    return MATERIALS[p.material].density_g_cm3


def bosses(p):
    """(x, z) of the four screw bosses."""
    return [(sx * p.boss_pitch_x / 2, sz * p.boss_pitch_z / 2) for sx in (-1, 1) for sz in (-1, 1)]


def outline(p):
    """Outside of the box at the rim, (x, z)."""
    r = p.corner_r
    return box(-p.length / 2, -p.width / 2, p.length / 2, p.width / 2).buffer(-r).buffer(r, quad_segs=8)


def inner_outline(p, y=0.0):
    """Inside of the pocket at height y (it narrows going up by the draft)."""
    return outline(p).buffer(-p.wall - y * math.tan(math.radians(p.draft_deg)), quad_segs=8)


def floor_y(p):
    return p.height - p.wall


def rounded_prism(l, w, r, h, deg, y0=0.0):
    """Rounded rectangle l x w (corner radius r, true arcs) on Y = y0, centred on the Y
    axis, extruded by h along +Y with its sides drafted inward by `deg`."""
    pl = Plane(origin=(0, y0, 0), x_dir=(1, 0, 0), z_dir=(0, 1, 0))
    return draft_sides(extrude(pl * RectangleRounded(l, w, r), h), deg, y0)


def envelope(p):
    """The outside of the box as a solid (pocket and window filled)."""
    return rounded_prism(p.length, p.width, p.corner_r, p.height, p.draft_deg)


def pocket(p):
    """The inside of the box as a solid, from just below the rim to the floor."""
    g = 2 * (p.wall - math.tan(math.radians(p.draft_deg)))      # 1 mm below the rim it is that much wider
    return rounded_prism(p.length - g, p.width - g, max(p.corner_r - p.wall, 0.5), 1.0 + floor_y(p), p.draft_deg, y0=-1.0)


def window_face_z(p):
    """Z of the front outer face at the window centre height (the wall leans in going up)."""
    return -p.width / 2 + p.window_y * math.tan(math.radians(p.draft_deg))


def window_cutter(p, extra=1.0):
    """Through-cutter of the window: from `extra` outside the outer face to past the inner
    face, narrowing along +Z (inward) so the slide can pull it out along -Z. Its nominal
    size is at the outer face, at the window centre."""
    tan = math.tan(math.radians(p.window_draft_deg))
    z0 = window_face_z(p) - extra
    L = extra + p.wall + 2.0
    w, h = p.window_w + 2 * extra * tan, p.window_h + 2 * extra * tan
    # built along +Y in a local frame, then turned so local +Y -> +Z and local z -> -Y
    rect = box(p.window_x - w / 2, -p.window_y - h / 2, p.window_x + w / 2, -p.window_y + h / 2)
    return Rot(90, 0, 0) * drafted_prism(rect, L, p.window_draft_deg, y0=z0)


def _rib(p, a, b):
    """Rib from (x, z) a to b, hanging from the floor, widening toward it."""
    from shapely.geometry import LineString
    tan = math.tan(math.radians(p.rib_draft_deg))
    tip = floor_y(p) - p.rib_h
    t_tip = p.rib_t - 2 * p.rib_h * tan
    poly = LineString([a, b]).buffer(t_tip / 2, cap_style="flat").intersection(inner_outline(p, tip).buffer(0.3))
    return drafted_prism(poly, p.rib_h + 0.5, -p.rib_draft_deg, y0=tip)


def cone_y(x, z, y0, r0, h, deg):
    """Cone frustum on the circle (x, z, r0) at Y = y0, height h along +Y, its side
    narrowing by `deg` going up (negative: widening)."""
    r1 = r0 - h * math.tan(math.radians(deg))
    return Pos(x, y0, z) * Cone(r0, r1, h, rotation=(-90, 0, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))


def _boss(p, x, z):
    return cone_y(x, z, p.boss_recess, p.boss_d / 2, floor_y(p) - p.boss_recess + 0.5, -p.boss_draft_deg)


def _hole(p, x, z):
    tan = math.tan(math.radians(p.boss_draft_deg))
    return cone_y(x, z, p.boss_recess - 1.0, p.hole_d / 2 + tan, p.hole_depth + 1.0, p.boss_draft_deg)


def build(p: HousingParams):
    """The moulded part, standing on the parting plane Y = 0 (opening toward -Y)."""
    part = envelope(p) - pocket(p)
    L, W = p.length, p.width
    part += _rib(p, (-L / 2, 0.0), (L / 2, 0.0))
    part += _rib(p, (0.0, -W / 2), (0.0, W / 2))
    for x, z in bosses(p):
        part += _boss(p, x, z)
    for x, z in bosses(p):
        part -= _hole(p, x, z)
    part -= window_cutter(p)
    return part.clean()
