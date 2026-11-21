"""Checks on an injection-moulded part and its mould, measured on the B-rep.

    report = verify_molding(HousingParams())

- release: every surface of the shot (part, gate, runner, sprue, well) is formed by a
  mould component, found by probing the steel just outside the surface; each
  component has a pull direction (fixed half +Y, moving half and its pins -Y, slide
  -Z). At every sample point the wall must open toward that direction by at least
  the resin's minimum draft, and a ray along it must leave the shot without
  re-entering it (no undercut for that component)
- wall: thinnest measured wall against the resin minimum; rib / boss proportions
  against the usual sink-mark rules (design rules, from the parameters)
- mould: one valid solid per component, no overlap with the cavity or each other;
  ejector pins and sleeves sit fully on flat faces of the shot; cooling channels
  and bubblers keep their distance to the cavity and to every pin
- slide: the angle pin's stroke clears the window insert; once retracted the slide
  clears the part's ejection path; the locking heel is steeper than the pin
- process: projected area, clamp force and machine size; gate shear rate, flow
  length ratio, runner size; cooling time and cycle; masses and yield (estimates)
"""
import math
import numpy as np
from shapely.geometry import box as sbox, Point
from build123d import Pos, Cylinder, Align, Vector
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Pnt, gp_Dir, gp_Lin
from OCP.Precision import Precision
from ..parts import housing
from ..process import molding as mo
from .forging_check import surface_samples, _rule, TOL_DRAFT, TOL_UNDERCUT
from .wall import min_wall

COOL_TO_CAVITY = 1.5      # x channel diameter
COOL_TO_PIN_MM = 4.0
RIB_RATIO_MAX = 0.6       # rib root / wall (sink marks; PA-GF guideline 0.5-0.6)
RIB_HEIGHT_MAX = 3.0      # x wall
BOSS_RATIO_MIN = 2.0      # boss outside / hole
LOCK_MARGIN_DEG = 2.0     # locking heel steeper than the angle pin
ANGLE_PIN_RANGE = (10.0, 25.0)
PROBE_MM = 0.05

PULL = {"nozzle": (0, 1, 0), "fixed": (0, 1, 0), "moving": (0, -1, 0), "slide": (0, 0, -1), "core_pin": (0, -1, 0),
        "sleeve": (0, -1, 0), "ejector_pin": (0, -1, 0)}


def components(ms):
    """[(kind, solid)] of the mould, small pieces first (cheaper to classify)."""
    out = [("ejector_pin", s) for s in ms["pins"]] + [("sleeve", s) for s in ms["sleeves"]] + \
          [("core_pin", s) for s in ms["core_pins"]]
    return out + [("slide", ms["slide"]), ("moving", ms["moving"]), ("fixed", ms["fixed"]), ("nozzle", ms["nozzle"])]


def release(shot, comps, k, per_face=3):
    """Per sample point of the shot: the component forming it, the draft toward that
    component's pull direction and whether that direction is blocked (undercut).
    Returns dict(points=[(q, kind, draft_deg, undercut, face_index)], unformed=[q])."""
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shot.wrapped, 1e-6)
    pts, unformed = [], []
    for q, n, fi in surface_samples(shot, per_face):
        probe = (q + n * PROBE_MM) * k
        kind = next((kd for kd, s in comps if s.is_inside(Vector(*probe), 1e-6)), None)
        if kind is None:
            unformed.append(tuple(round(float(c), 2) for c in q))
            continue
        d = np.array(PULL[kind], float)
        c = float(n @ d)
        dr = math.degrees(math.asin(max(-1.0, min(1.0, c))))
        reach = TOL_UNDERCUT / math.tan(math.radians(max(dr, 0.5)))
        start = gp_Pnt(*(q + d * 1e-3 + n * 1e-4))
        inter.Perform(gp_Lin(start, gp_Dir(*d)), 0.0, Precision.Infinite_s())
        blocked = c < 1e-6 or (inter.IsDone() and any(inter.WParameter(i) > reach for i in range(1, inter.NbPnt() + 1)))
        pts.append((tuple(round(float(x), 2) for x in q), kind, dr, blocked, fi))
    return dict(points=pts, unformed=unformed)


def _min_dist(a_list, b_list):
    return min((a.distance_to(b) for a in a_list for b in b_list), default=float("inf"))


def face_sides(shot, rel):
    """Area of the shot formed by each component (each face goes to its samples' majority)."""
    by_face = {}
    for _, kind, _, _, fi in rel["points"]:
        by_face.setdefault(fi, []).append(kind)
    faces = shot.faces()
    area = {}
    for fi, kinds in by_face.items():
        kd = max(set(kinds), key=kinds.count)
        area[kd] = area.get(kd, 0.0) + faces[fi].area
    return area


def flow_length(p, g, part):
    """Longest flow path from the gate, estimated as rise along the walls plus the
    straight run across: |dy| + |dxz| from the gate to each surface sample."""
    gx, gy = p.length / 2, g["gate_t"] / 2
    best = 0.0
    for q, _, _ in surface_samples(part, 2):
        best = max(best, abs(q[1] - gy) + math.hypot(q[0] - gx, q[2]))
    return best


def projected_area(p, mp, g):
    out = housing.outline(p)
    x0 = p.length / 2
    xs = mo.sprue_x(p, mp)
    feed = sbox(x0 - 0.5, -g["gate_w"] / 2, x0 + mp.gate_land_mm + 0.5, g["gate_w"] / 2).union(
        sbox(x0 + mp.gate_land_mm, -mp.runner_w_mm / 2, xs, mp.runner_w_mm / 2)).union(
        Point(xs, 0).buffer(g["sprue_r0"], quad_segs=16))
    return out.union(feed).area * mo.shrink_factor(p) ** 2


def cooling_time(p):
    """One-dimensional slab: t = s^2 / (pi^2 a) ln(8/pi^2 (Tm - Tw)/(Te - Tw))."""
    r = mo.resin(p)
    s = p.wall
    return s * s / (math.pi ** 2 * r.diffusivity_mm2_s) * math.log(
        8 / math.pi ** 2 * (r.T_melt_C - r.T_mould_C) / (r.T_eject_C - r.T_mould_C))


def verify_molding(p=housing.HousingParams(), mp=mo.MouldParams(), ms=None):
    ms = ms or mo.mould_set(p, mp)
    part, shot, cav, g = ms["part"], ms["shot"], ms["cavity"], ms["gating"]
    k = mo.shrink_factor(p)
    rs = mo.resin(p)
    rules = []

    ok = part.is_valid and len(part.solids()) == 1
    rules.append(_rule("part.valid", ok, ok, True, "one valid solid"))
    rho = housing.density(p)
    m_part = part.volume / 1000 * rho
    rules.append(_rule("part.mass", m_part <= p.mass_target_g, round(m_part, 1), p.mass_target_g, "g, moulded part"))

    # ---- release, per mould component
    rel = release(shot, components(ms), k)
    pts = rel["points"]
    rules.append(_rule("release.formed", not rel["unformed"], len(rel["unformed"]), 0,
                       "shot surface points with no mould steel next to them", at_mm=rel["unformed"][:5]))
    dmin = min(pts, key=lambda t: t[2])
    rules.append(_rule("release.draft", dmin[2] >= rs.min_draft_deg - TOL_DRAFT, round(dmin[2], 2), rs.min_draft_deg,
                       "wall angle to the pull direction of the component forming it", at_mm=dmin[0], component=dmin[1]))
    under = [(q, kd) for q, kd, _, b, _ in pts if b]
    rules.append(_rule("release.undercut", not under, len(under), 0,
                       "points whose component cannot withdraw along its pull direction", at_mm=under[:5]))
    by_kind = {}
    for q, kd, dr, _, _ in pts:
        by_kind.setdefault(kd, []).append(dr)
    draft_by = {kd: round(min(v), 2) for kd, v in by_kind.items()}
    for kd, want in (("fixed", p.draft_deg), ("slide", p.window_draft_deg)):
        if kd in draft_by:
            rules.append(_rule(f"release.draft_{kd}", draft_by[kd] >= want - TOL_DRAFT, draft_by[kd], want,
                               f"deg, surfaces formed by the {kd} (outer walls / window)"))
    n_slide = len(by_kind.get("slide", []))
    rules.append(_rule("release.slide_forms_window", n_slide > 0, n_slide, 1, "sample points formed by the slide"))

    # ---- walls and proportions
    w = min_wall(part, per_face=4)
    rules.append(_rule("wall.min", w["min_mm"] >= rs.min_wall_mm - 0.05, w["min_mm"], rs.min_wall_mm,
                       "thinnest measured wall (mm)", at_mm=w["at_mm"]))
    rules.append(_rule("design.rib_thickness", p.rib_t / p.wall <= RIB_RATIO_MAX + 1e-9, round(p.rib_t / p.wall, 3),
                       RIB_RATIO_MAX, "rib root / wall (sink marks)"))
    rules.append(_rule("design.rib_height", p.rib_h <= RIB_HEIGHT_MAX * p.wall + 1e-9, p.rib_h,
                       RIB_HEIGHT_MAX * p.wall, "mm, rib height <= 3 x wall"))
    rules.append(_rule("design.boss", p.boss_d / p.hole_d >= BOSS_RATIO_MIN - 1e-9, round(p.boss_d / p.hole_d, 2),
                       BOSS_RATIO_MIN, "boss outside / hole"))

    # ---- mould components
    comps = {"fixed": ms["fixed"], "moving": ms["moving"], "slide": ms["slide"], "angle_pin": ms["angle_pin"]}
    comps.update({f"core_pin_{i + 1}": s for i, s in enumerate(ms["core_pins"])})
    comps.update({f"sleeve_{i + 1}": s for i, s in enumerate(ms["sleeves"])})
    comps.update({f"ejector_pin_{i + 1}": s for i, s in enumerate(ms["pins"])})
    for name, s in comps.items():
        if name in ("fixed", "moving", "slide"):
            ok = s.is_valid and len(s.solids()) == 1
            rules.append(_rule(f"mould.{name}.valid", ok, ok, True, "one valid solid"))
    ov_cav = {n: round((s & cav).volume, 4) for n, s in comps.items()}
    worst = max(ov_cav, key=ov_cav.get)
    rules.append(_rule("mould.clears_cavity", ov_cav[worst] < 1e-2, ov_cav[worst], 0.0,
                       "mm3, largest overlap of a component with the cavity", component=worst))
    names = list(comps)
    clash = []
    for i, a in enumerate(names):
        bba = comps[a].bounding_box()
        for b in names[i + 1:]:
            bbb = comps[b].bounding_box()
            if any(tuple(bba.max)[j] < tuple(bbb.min)[j] or tuple(bbb.max)[j] < tuple(bba.min)[j] for j in range(3)):
                continue
            v = (comps[a] & comps[b]).volume
            if v > 1e-2:
                clash.append((a, b, round(v, 3)))
    rules.append(_rule("mould.components_clear", not clash, len(clash), 0, "component pairs overlapping (> 0.01 mm3)",
                       pairs=clash[:5]))

    # ---- ejection: the disc just inside each pin / sleeve face is all plastic
    seat = []
    for (x, z, y), pin in zip(ms["pin_sites"], ms["pins"]):
        disc = Pos(x * k, y * k, z * k) * Cylinder(mp.pin_d_mm / 2, 0.2, rotation=(-90, 0, 0),
                                                   align=(Align.CENTER, Align.CENTER, Align.MIN))
        seat.append(round((disc & cav).volume / disc.volume, 4))
    for (x, z), sl in zip(housing.bosses(p), ms["sleeves"]):
        ring = Pos(x * k, p.boss_recess * k, z * k) * Cylinder((p.boss_d / 2 - mp.sleeve_inset_mm) * k, 0.2,
                                                               rotation=(-90, 0, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
        ring -= Pos(x * k, p.boss_recess * k - 1, z * k) * Cylinder(p.hole_d / 2 * k, 2, rotation=(-90, 0, 0),
                                                                    align=(Align.CENTER, Align.CENTER, Align.MIN))
        seat.append(round((ring & cav).volume / ring.volume, 4))
    rules.append(_rule("ejector.seated", min(seat) > 0.999, min(seat), 1.0,
                       "fraction of each pin / sleeve face backed by the shot", per_pin=seat))
    on_floor = sum(1 for _, _, y in ms["pin_sites"] if abs(y - housing.floor_y(p)) < 1e-6)
    rules.append(_rule("ejector.on_part", on_floor >= 4 and len(ms["sleeves"]) == 4, [on_floor, len(ms["sleeves"])],
                       [4, 4], "pins on the floor, sleeves on the bosses"))

    # ---- slide
    x0, x1, ys, zb, zf = mo.slide_frame(p, mp)
    travel = mo.slide_travel(p, mp)
    wb = ms["window_insert"].bounding_box()
    depth = wb.max.Z - wb.min.Z
    rules.append(_rule("slide.stroke", travel >= depth + mp.slide_safety_mm - 1e-6, round(travel, 2),
                       round(depth + mp.slide_safety_mm, 2), "mm, angle-pin stroke >= window insert depth + safety"))
    retracted = Pos(0, 0, -travel) * ms["slide"]
    hit = (retracted & _sweep_y(ms["envelope"], 100.0)).volume
    rules.append(_rule("slide.clears_ejection", hit < 1e-2, round(hit, 3), 0.0,
                       "mm3 of the retracted slide inside the part's ejection path (+Y)"))
    rules.append(_rule("slide.lock_angle", mp.lock_deg >= mp.angle_pin_deg + LOCK_MARGIN_DEG, mp.lock_deg,
                       mp.angle_pin_deg + LOCK_MARGIN_DEG, "deg, locking heel >= angle pin + 2"))
    rules.append(_rule("slide.angle_pin", ANGLE_PIN_RANGE[0] <= mp.angle_pin_deg <= ANGLE_PIN_RANGE[1], mp.angle_pin_deg,
                       list(ANGLE_PIN_RANGE), "deg, usual angle-pin range"))

    # ---- cooling
    chf, chm, bub = ms["channels_fixed"], ms["channels_moving"], ms["bubblers"]
    dch = _min_dist(chf + chm, [cav])
    rules.append(_rule("cooling.to_cavity", dch >= COOL_TO_CAVITY * mp.cool_d_mm - 1e-3, round(dch, 2),
                       COOL_TO_CAVITY * mp.cool_d_mm, "mm, plate channel wall to cavity (>= 1.5 d)"))
    dbb = _min_dist(bub, [cav])
    rules.append(_rule("cooling.bubbler_to_cavity", dbb >= COOL_TO_CAVITY * mp.bubbler_d_mm - 1e-3, round(dbb, 2),
                       COOL_TO_CAVITY * mp.bubbler_d_mm, "mm, bubbler wall to cavity (>= 1.5 d)"))
    moving_pins = ms["pins"] + ms["sleeves"] + ms["core_pins"]
    dpin = min(_min_dist(chm + bub, moving_pins), _min_dist(chf, [ms["angle_pin"]]))
    rules.append(_rule("cooling.to_pins", dpin >= COOL_TO_PIN_MM, round(dpin, 2), COOL_TO_PIN_MM,
                       "mm, channel wall to ejector pins, sleeves, core pins and the angle pin"))
    dsl = _min_dist(chf, [ms["slide_block"]])
    rules.append(_rule("cooling.to_slide", dsl >= COOL_TO_PIN_MM, round(dsl, 2), COOL_TO_PIN_MM,
                       "mm, fixed-half channel wall to the slide pocket"))
    rules.append(_rule("cooling.present", len(chf) >= 3 and len(chm) >= 2 and len(bub) >= 4,
                       [len(chf), len(chm), len(bub)], [3, 2, 4], "channels fixed / moving, bubblers in the core"))

    # ---- process estimates
    area = projected_area(p, mp, g)
    f_open = rs.cavity_MPa * area / 1000                       # kN
    t_clamp = mp.clamp_margin * f_open / 9.81
    machine = next((m for m in mo.MACHINES_T if m >= t_clamp), None)
    rules.append(_rule("process.machine", machine is not None, machine, mo.MACHINES_T[-1],
                       "smallest standard machine (tf) with clamp force >= margin x opening force"))
    q = (part.volume + g["gate"].volume) / mp.fill_time_s
    shear = 6 * q / (g["gate_w"] * g["gate_t"] ** 2)
    rules.append(_rule("process.gate_shear", shear <= rs.max_shear_1_s, round(shear), rs.max_shear_1_s,
                       "1/s, wall shear rate in the gate 6Q/(w h^2)"))
    fl = flow_length(p, g, part)
    rules.append(_rule("process.flow_ratio", fl / p.wall <= rs.flow_ratio, round(fl / p.wall, 1), rs.flow_ratio,
                       "longest flow path / wall (estimate)"))
    a_r = (mp.runner_w_mm + mp.runner_w_mm - 2 * mp.runner_d_mm * math.tan(math.radians(mp.runner_draft_deg))) / 2 * mp.runner_d_mm
    per = mp.runner_w_mm + (mp.runner_w_mm - 2 * mp.runner_d_mm * math.tan(math.radians(mp.runner_draft_deg))) + \
        2 * mp.runner_d_mm / math.cos(math.radians(mp.runner_draft_deg))
    dh = 4 * a_r / per
    rules.append(_rule("process.runner", dh >= p.wall + 1.5 - 1e-6, round(dh, 2), p.wall + 1.5,
                       "mm, runner hydraulic diameter >= wall + 1.5"))
    sides = face_sides(shot, rel)
    core_side = sum(v for kd, v in sides.items() if kd != "fixed")
    rules.append(_rule("process.stays_on_core", core_side > sides.get("fixed", 0.0), round(core_side), round(sides.get("fixed", 0.0)),
                       "mm2 of shot surface on the moving side (core, slide, pins) vs the fixed side: the part shrinks onto the core"))
    tc = cooling_time(p)
    side_area = (p.window_w + 2 * mp.slide_margin_mm) * (p.window_y + p.window_h / 2 + mp.slide_top_mm)
    m_shot = shot.volume / 1000 * rho
    steel, srho = mo.MOULD_STEEL
    steel_vol = sum(s.volume for s in comps.values())
    return {
        "rules": rules, "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]],
        "process": {
            "process": "injection moulding, single cavity, cold runner, edge gate, one slide",
            "resin": p.material, "iso_marking": rs.iso_mark,
            "parting_plane": "Y = 0 (housing rim); fixed half Y >= 0 (cavity), moving half Y <= 0 (core)",
            "shrink_pct": rs.shrink_pct, "shrink_factor": round(k, 5),
            "projected_area_mm2": round(area), "cavity_pressure_MPa": rs.cavity_MPa,
            "opening_force_kN": round(f_open), "clamp_force_required_t": round(t_clamp, 1), "machine_t": machine,
            "fill_time_s": mp.fill_time_s, "gate_mm": [round(g["gate_w"], 2), round(g["gate_t"], 2)],
            "gate_shear_1_s": round(shear), "flow_length_mm": round(fl, 1), "flow_ratio": round(fl / p.wall, 1),
            "runner_hydraulic_d_mm": round(dh, 2),
            "cooling_time_s": round(tc, 1), "cycle_time_s": round(mp.fill_time_s + tc + 6.0, 1),
            "cycle_basis": "fill + cooling (incl. packing) + 6 s mould open / eject / close (estimate)",
            "slide_stroke_mm": round(travel, 2), "slide_insert_depth_mm": round(depth, 2),
            "slide_side_force_kN": round(rs.cavity_MPa * side_area / 1000, 1),
            "draft_min_by_component_deg": draft_by,
            "surface_by_component_mm2": {kd: round(v) for kd, v in sides.items()},
            "ejector_pins": len(ms["pins"]), "ejector_sleeves": len(ms["sleeves"]),
            "cooling": {"fixed_channels": len(chf), "moving_channels": len(chm), "bubblers": len(bub)},
            "mass_part_g": round(m_part, 1), "mass_shot_g": round(m_shot, 1),
            "mass_feed_g": round(m_shot - m_part, 1), "yield_pct": round(100 * m_part / m_shot, 1),
            "mould_steel": steel, "mould_mass_kg": round(steel_vol / 1e6 * srho, 1),
            "mould_block_mm": [round(v, 1) for v in ms["block_mm"]],
        },
    }


def _sweep_y(solid, dy):
    """The region `solid` passes through when moved by dy along +Y (its XZ footprint
    extruded over its own Y range plus dy)."""
    from ..process.forging import section2d, face_at
    from build123d import extrude
    bb = solid.bounding_box()
    foot = section2d(solid, bb.min.Y + 1e-3)         # a drafted envelope is widest at its base
    return extrude(face_at(foot.buffer(0.01), bb.min.Y), bb.size.Y + dy, dir=(0, 1, 0))
