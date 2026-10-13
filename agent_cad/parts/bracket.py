"""Die-cast aluminium mounting bracket (ADC12): a ribbed plate with four bolt bosses
and a central bearing hub, cast near-net in a two-plate die that opens along Y.

As cast, every wall is drafted so the part leaves the ejector die along -Y: the
parting plane is the plate's back face (Y = 0) and the whole part stands on it, in
the ejector die. Walls taper inward going up; cored holes widen going up (the core
pins taper toward the parting plane). Afterwards the boss and hub tops are faced and
the hub bore reamed to H7; the bolt holes stay as cast.

    casting, finished = build(BracketParams())
"""
import math
from dataclasses import dataclass, asdict
from shapely.geometry import Point, LineString, box
from build123d import Box, Cylinder, Pos, Align
from ..materials import MATERIALS
from ..process.forging import drafted_prism

__all__ = ["BracketParams", "build", "density", "asdict"]


@dataclass(frozen=True)
class BracketParams:
    plate_l: float = 120.0        # X
    plate_w: float = 80.0         # Z
    plate_t: float = 4.0          # Y
    corner_r: float = 10.0
    boss_pitch_x: float = 96.0
    boss_pitch_z: float = 56.0
    boss_d: float = 18.0
    boss_h: float = 12.0          # finished height of the bolt boss tops above the back face
    hole_d: float = 9.0           # M8 clearance (JIS B 1001, class 2), at its narrowest
    hub_d: float = 30.0
    hub_h: float = 26.0           # finished hub height
    bore_d: float = 16.0          # H7, reamed
    rib_t: float = 4.0            # at the root
    rib_h: float = 11.0           # below the boss tops, so facing does not touch them
    draft_deg: float = 1.5        # outer walls and ribs
    core_draft_deg: float = 1.0   # cored holes
    stock_mm: float = 0.8         # machining stock on faced and reamed surfaces
    material: str = "ADC12"
    mass_target_g: float = 250.0


def density(p):
    return MATERIALS[p.material].density_g_cm3


def bosses(p):
    """(x, z) of the four bolt bosses."""
    return [(sx * p.boss_pitch_x / 2, sz * p.boss_pitch_z / 2) for sx in (-1, 1) for sz in (-1, 1)]


def _cored_hole(d, top, deg):
    """Cutter for a cored hole of diameter d at Y = 0, widening by `deg` going up to `top`."""
    return drafted_prism(Point(0, 0).buffer(d / 2, quad_segs=32), top + 2.0, -deg, y0=-1.0)


def build(p: BracketParams):
    """(casting, finished part). Both stand on the parting plane Y = 0."""
    s = p.stock_mm
    plate = box(-p.plate_l / 2, -p.plate_w / 2, p.plate_l / 2, p.plate_w / 2)
    plate = plate.buffer(-p.corner_r).buffer(p.corner_r, quad_segs=16)
    cast = drafted_prism(plate, p.plate_t, p.draft_deg)
    for x, z in bosses(p):
        cast += drafted_prism(Point(x, z).buffer(p.boss_d / 2, quad_segs=32), p.boss_h + s, p.draft_deg)
        rib = LineString([(0, 0), (x, z)]).buffer(p.rib_t / 2, cap_style="flat")
        cast += drafted_prism(rib, p.rib_h, p.draft_deg)
    cast += drafted_prism(Point(0, 0).buffer(p.hub_d / 2, quad_segs=32), p.hub_h + s, p.draft_deg)
    for x, z in bosses(p):
        cast -= Pos(x, 0, z) * _cored_hole(p.hole_d, p.boss_h + s, p.core_draft_deg)
    # the cored bore is widest at the hub top: there it leaves exactly the reaming stock
    tan = math.tan(math.radians(p.core_draft_deg))
    cast -= _cored_hole(p.bore_d - 2 * s - 2 * (p.hub_h + s + 1.0) * tan, p.hub_h + s, p.core_draft_deg)
    cast = cast.clean()

    fin = cast
    up = Align.CENTER, Align.MIN, Align.CENTER          # boxes standing on their Y = 0 face
    for x, z in bosses(p):
        fin -= Pos(x, p.boss_h, z) * Box(p.boss_d + 2, s + 2, p.boss_d + 2, align=up)
    fin -= Pos(0, p.hub_h, 0) * Box(p.hub_d + 2, s + 2, p.hub_d + 2, align=up)
    fin -= Pos(0, -1, 0) * (Cylinder(p.bore_d / 2, p.hub_h + 4, rotation=(-90, 0, 0),
                                     align=(Align.CENTER, Align.CENTER, Align.MIN)))
    return cast, fin.clean()
