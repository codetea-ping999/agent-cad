"""High-pressure die casting (cold chamber): gating, overflows, the die pair, ejector
pins and cooling channels for the die-cast bracket.

Layout. The parting plane is the bracket's back face (Y = 0). The moving (ejector)
die at Y >= 0 holds the whole cavity, the cored holes (its core pins) and the
gating; the fixed (cover) die at Y <= 0 is flat against the back face and carries
the shot sleeve. The part shrinks onto the core pins, so it stays in the moving die
when the die opens; the ejector pins then push it out along -Y.

    shot (cast part + gating)    casting + fan runner + gate + biscuit + overflows
    cavity                       shot x (1 + shrink); the die is cut at room
                                 temperature, the part shrinks on cooling
    moving die                   block (Y >= 0) - cavity - ejector pin holes - channels
    cover die                    block (Y <= 0) - shot sleeve bore - channels

Gating is sized from the NADCA fill-time formula and a target gate velocity; the
numbers (casting pressure, fill time, velocities, shrinkage) are the usual
empirical starting values for ADC12, to be confirmed by a filling/solidification
simulation and die trials before the die is cut.
"""
import math
from dataclasses import dataclass, asdict
import shapely
from shapely.geometry import Point, Polygon as SPolygon, box
from build123d import Box, Cylinder, Pos, Align, Compound, scale
from ..parts import bracket
from ..materials import MATERIALS
from .forging import drafted_prism

DIE_STEEL = ("SKD61", 7.80)        # JIS G 4404 hot-work tool steel (die inserts, pins)
MACHINES_T = (125, 250, 350, 500, 800, 1250, 1650, 2500)   # usual cold-chamber sizes, tf


@dataclass(frozen=True)
class DiecastParams:
    shrink_pct: float = 0.55          # ADC12 linear casting shrinkage
    pressure_MPa: float = 60.0        # cavity (intensification) pressure
    clamp_margin: float = 1.2         # machine clamp force >= margin x opening force
    # NADCA fill time t = K T (Ti - Tf + S Z) / (Tf - Td)
    fill_K: float = 0.0346            # s/mm, aluminium
    T_inject_C: float = 650.0
    T_flow_C: float = 570.0           # minimum flow temperature
    T_die_C: float = 220.0
    solid_pct: float = 20.0           # solid fraction allowed at the end of fill
    Z_C_per_pct: float = 3.8
    gate_v_m_s: float = 35.0          # target gate velocity
    gate_t_mm: float = 1.6            # gate thickness (~0.4 x the 4 mm wall)
    gate_land_mm: float = 2.0
    runner_ratio: float = 1.6         # runner section / gate section
    runner_depth_mm: float = 6.0
    runner_draft_deg: float = 5.0
    biscuit_d_mm: float = 40.0        # = shot sleeve bore
    biscuit_t_mm: float = 15.0
    sleeve_gap_mm: float = 25.0       # sleeve edge to the part outline (along -Z)
    overflow_mm: tuple = (12.0, 10.0, 5.0)   # length along the edge, width, depth
    overflow_gate_t_mm: float = 0.8
    overflow_gate_l_mm: float = 3.0
    pin_d_mm: float = 6.0             # ejector pins (JIS B 5103 type)
    pin_clear_mm: float = 1.5         # pin edge to any wall foot
    pin_tail_mm: float = 30.0         # pin length behind the die block (ejector plate)
    cool_d_mm: float = 8.0            # drilled cooling channels
    cool_cover_mm: float = 18.0       # channel axis beyond the deepest cavity point
    cool_pitch_mm: float = 24.0
    die_wall_mm: float = 45.0         # die block beyond the shot outline
    die_base_mm: float = 45.0         # moving die behind the deepest cavity point
    cover_t_mm: float = 60.0


def fill_time(dp, wall_mm):
    """NADCA fill time (s) for the governing wall thickness."""
    return dp.fill_K * wall_mm * (dp.T_inject_C - dp.T_flow_C + dp.solid_pct * dp.Z_C_per_pct) \
        / (dp.T_flow_C - dp.T_die_C)


def gating_sizes(p, dp, cast_volume):
    """Gate and runner sections from the fill time and the target gate velocity.
    The overflows fill through the cavity, so they count in the volume."""
    ox, oy, oz = dp.overflow_mm
    v = cast_volume + len(overflow_sites(p)) * ox * oy * oz
    t = fill_time(dp, p.plate_t)
    a_gate = v / (dp.gate_v_m_s * 1000 * t)
    gate_w = a_gate / dp.gate_t_mm
    runner_w = dp.runner_ratio * a_gate / dp.runner_depth_mm
    return dict(volume_mm3=v, fill_time_s=t, gate_area_mm2=a_gate, gate_w_mm=gate_w,
                runner_w_mm=max(runner_w, 2 * dp.pin_d_mm), runner_area_mm2=max(runner_w, 2 * dp.pin_d_mm) * dp.runner_depth_mm)


def overflow_sites(p):
    """(x, z, outward unit vector) of overflow wells on the plate edges away from the gate:
    the far edge (+Z) and both ends (+-X), where the metal front arrives last."""
    hx, hz = p.plate_l / 2, p.plate_w / 2
    return [(-p.boss_pitch_x / 4, hz, (0, 1)), (p.boss_pitch_x / 4, hz, (0, 1)),
            (-hx, 0.0, (-1, 0)), (hx, 0.0, (1, 0))]


def sleeve_center(p, dp):
    return 0.0, -(p.plate_w / 2 + dp.sleeve_gap_mm + dp.biscuit_d_mm / 2)


def gating(p, dp, cast_volume):
    """(runner + gate, biscuit, [overflows]) solids, all standing on the parting plane."""
    g = gating_sizes(p, dp, cast_volume)
    hz = p.plate_w / 2
    xs, zs = sleeve_center(p, dp)
    z_gate = -hz - dp.gate_land_mm
    rw, gw = g["runner_w_mm"], g["gate_w_mm"]
    # fan runner: from the biscuit centre to the gate land, widening (or narrowing) to the
    # gate width, so the modelled gate is the section the gate velocity is sized for
    fan = SPolygon([(xs - rw / 2, zs), (xs + rw / 2, zs), (gw / 2, z_gate), (-gw / 2, z_gate)])
    runner = drafted_prism(fan, dp.runner_depth_mm, dp.runner_draft_deg)
    land = box(-gw / 2, z_gate - 0.5, gw / 2, -hz + 0.5)       # overlaps runner and plate edge
    runner += drafted_prism(land, dp.gate_t_mm, dp.runner_draft_deg)
    biscuit = Pos(xs, 0, zs) * Cylinder(dp.biscuit_d_mm / 2, dp.biscuit_t_mm, rotation=(90, 0, 0),
                                       align=(Align.CENTER, Align.CENTER, Align.MIN))
    ox, oy, oz = dp.overflow_mm
    wells = []
    for x, z, (ux, uz) in overflow_sites(p):
        gl = dp.overflow_gate_l_mm
        a, b = (ox, oy) if uz else (oy, ox)          # well footprint: long side along the edge
        cx, cz = x + ux * (gl + oy / 2), z + uz * (gl + oy / 2)
        well = drafted_prism(box(cx - a / 2, cz - b / 2, cx + a / 2, cz + b / 2), oz, dp.runner_draft_deg)
        ga, gb = (ox * 0.6, gl + 1.0) if uz else (gl + 1.0, ox * 0.6)
        gx, gz = x + ux * (gl / 2), z + uz * (gl / 2)
        well += drafted_prism(box(gx - ga / 2, gz - gb / 2, gx + ga / 2, gz + gb / 2), dp.overflow_gate_t_mm,
                              dp.runner_draft_deg)
        wells.append(well)
    return runner, biscuit, wells


def shot(p, dp, casting):
    runner, biscuit, wells = gating(p, dp, casting.volume)
    s = casting + runner + biscuit
    for w in wells:
        s += w
    return s.clean()


def shrink_factor(dp):
    return 1 + dp.shrink_pct / 100


def top_faces(solid, y, tol=1e-3):
    """Planar faces of `solid` facing +Y at height y, as one shapely region in (x, z)."""
    tris = []
    for f in solid.faces():
        if f.geom_type.name != "PLANE":
            continue
        if f.normal_at().Y > 1 - 1e-9 and abs(f.center().Y - y) < tol:
            vs, ts = f.tessellate(0.01, 0.1)
            P = [(v.X, v.Z) for v in vs]
            tris += [SPolygon([P[i] for i in t]) for t in ts]
    return shapely.union_all(tris).buffer(0) if tris else SPolygon()


def pin_sites(p, dp, shot_solid):
    """Ejector pin positions (x, z, y_face): on flat +Y faces, every pin circle at least
    pin_clear from the face boundary. Candidates: the plate between the ribs, every
    overflow well, the runner at the biscuit."""
    r = dp.pin_d_mm / 2 + dp.pin_clear_mm
    xs, zs = sleeve_center(p, dp)
    ox, oy, oz = dp.overflow_mm
    cands = [(sx * p.boss_pitch_x / 2, 0.0, p.plate_t) for sx in (-1, 1)] + \
            [(0.0, sz * p.boss_pitch_z / 2, p.plate_t) for sz in (-1, 1)]
    for x, z, (ux, uz) in overflow_sites(p):
        g = dp.overflow_gate_l_mm + oy / 2
        cands.append((x + ux * g, z + uz * g, oz))
    cands.append((xs, zs + dp.biscuit_d_mm / 4, dp.runner_depth_mm))   # the runner, just off the biscuit
    out = []
    for x, z, y in cands:
        region = top_faces(shot_solid, y)
        if region.is_empty:
            continue
        # the drafted walls lean in by up to (wall height) tan(draft): measured at the face, not at Y = 0
        if region.buffer(-1e-3).contains(Point(x, z).buffer(r, quad_segs=16)):
            out.append((x, z, y))
    return out


def channel_sites(span, avoid, gap, pitch):
    """Positions along one axis inside `span` = (lo, hi), at least `gap` from every
    position in `avoid`, at least `pitch` apart (greedy from the low end)."""
    lo, hi = span
    out, z = [], lo
    while z <= hi:
        if all(abs(z - a) >= gap for a in avoid) and (not out or z - out[-1] >= pitch):
            out.append(z)
            z += pitch
        else:
            z += 0.5
    return out


def dies(p, dp, shot_solid):
    """dict(moving, cover, pins, channels_moving, channels_cover, cavity, pin_sites)."""
    k = shrink_factor(dp)
    cavity = scale(shot_solid, k)
    bb = cavity.bounding_box()
    w = dp.die_wall_mm
    x0, x1, z0, z1 = bb.min.X - w, bb.max.X + w, bb.min.Z - w, bb.max.Z + w
    hb = bb.max.Y + dp.die_base_mm
    cx, cz, lx, lz = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    moving = Pos(cx, hb / 2, cz) * Box(lx, hb, lz) - cavity
    cover = Pos(cx, -dp.cover_t_mm / 2, cz) * Box(lx, dp.cover_t_mm, lz)
    xs, zs = sleeve_center(p, dp)
    cover -= Pos(xs * k, 1, zs * k) * Cylinder(dp.biscuit_d_mm / 2 * k, dp.cover_t_mm + 2, rotation=(90, 0, 0),
                                               align=(Align.CENTER, Align.CENTER, Align.MIN))
    # ejector pins: from their seat on the cavity face through the block and out the back
    sites = pin_sites(p, dp, shot_solid)
    pins, rp = [], dp.pin_d_mm / 2
    for x, z, y in sites:
        L = hb - y * k + dp.pin_tail_mm
        pins.append(Pos(x * k, y * k, z * k) * Cylinder(rp, L, rotation=(-90, 0, 0),
                                                        align=(Align.CENTER, Align.CENTER, Align.MIN)))
    for pin in pins:
        moving -= pin
    # cooling: straight drilled channels along X, through the block, clear of pins and of the
    # shot sleeve (molten metal: the same 1.5 d as to the cavity)
    rc, d = dp.cool_d_mm / 2, dp.cool_d_mm
    gap_pin = rp + rc + 5.0
    ym = bb.max.Y + dp.cool_cover_mm
    zm = channel_sites((z0 + w / 2, z1 - w / 2), [z * k for _, z, _ in sites], gap_pin, dp.cool_pitch_mm)
    yc = -dp.cool_cover_mm
    zc = channel_sites((z0 + w / 2, z1 - w / 2), [zs * k], dp.biscuit_d_mm / 2 * k + rc + 1.5 * d + 0.5, dp.cool_pitch_mm)

    def drill(y, z):
        return Pos(cx, y, z) * Cylinder(rc, lx + 2, rotation=(0, 90, 0))
    ch_m = [drill(ym, z) for z in zm]
    ch_c = [drill(yc, z) for z in zc]
    for c in ch_m:
        moving -= c
    for c in ch_c:
        cover -= c
    return dict(moving=moving.clean(), cover=cover.clean(), pins=pins, channels_moving=ch_m,
                channels_cover=ch_c, cavity=cavity, pin_sites=sites, block_mm=(lx, hb, lz))


def die_set(p=bracket.BracketParams(), dp=DiecastParams()):
    casting, finished = bracket.build(p)
    s = shot(p, dp, casting)
    d = dies(p, dp, s)
    d.update(casting=casting, finished=finished, shot=s, pins_compound=Compound(d["pins"]))
    return d
