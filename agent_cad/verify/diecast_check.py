"""Checks on a die casting and its die, measured on the B-rep.

    report = verify_diecast(BracketParams())

- casting: one valid solid containing the finished part; machining stock on every
  machined surface (ray cast along the surface normal, as for the forging)
- draft / undercut: every wall of the casting and of the gating opens toward +Y
  (the moving die withdraws, the part leaves it along -Y)
- wall: thinnest measured wall of the casting against the ADC12 minimum
- die: one valid solid per half, no overlap with the cavity or each other; ejector
  pins sit fully on flat faces of the shot and do not cut into it; cooling channels
  keep their distance to the cavity, the pins and the shot sleeve
- process: projected area, opening force, clamp force and machine size; NADCA fill
  time, gate and runner sections, gate velocity; masses and yield (estimates)
"""
import numpy as np
from build123d import Box, Pos, Cylinder, Align, Compound, Vector
from ..parts import bracket
from ..process import diecast as dc
from ..process.forging import section2d
from .forging_check import allowance, draft_and_undercut, surface_samples, _rule, TOL_ALLOW, TOL_DRAFT
from .wall import min_wall

MIN_WALL_MM = 2.0          # ADC12 small castings (JIS H 5302 commentary / NADCA guideline)
COOL_TO_CAVITY = 1.5       # x channel diameter
COOL_TO_PIN_MM = 4.0
COOL_TO_SLEEVE = 1.5       # x channel diameter: the sleeve holds molten metal
COOL_TO_BACK = 1.0         # x channel diameter: steel behind the channel


def machined_faces(finished, casting, tol=1e-3):
    """Faces of the finished part that do not lie on the casting surface."""
    skin = Compound(casting.faces())
    out = []
    for f in finished.faces():
        pts = surface_samples(f, 2)
        if pts and all(skin.distance_to(Vector(*q)) > tol for q, _, _ in pts):
            out.append(f)
    return out


def _min_dist(a_list, b_list):
    return min((a.distance_to(b) for a in a_list for b in b_list), default=float("inf"))


def verify_diecast(p=bracket.BracketParams(), dp=dc.DiecastParams(), ds=None):
    ds = ds or dc.die_set(p, dp)
    cast, fin, shot, cav = ds["casting"], ds["finished"], ds["shot"], ds["cavity"]
    mv, cv, pins = ds["moving"], ds["cover"], ds["pins"]
    k = dc.shrink_factor(dp)
    rules = []

    ok = cast.is_valid and len(cast.solids()) == 1
    rules.append(_rule("casting.valid", ok, ok, True, "one valid solid"))
    unc = (fin - cast).volume
    rules.append(_rule("casting.contains_part", unc < 1e-3, round(unc, 4), 0.0, "mm3 of finished part outside the casting"))
    mf = machined_faces(fin, cast)
    amin, at = allowance(mf, cast)
    rules.append(_rule("casting.machining_stock", amin >= p.stock_mm - TOL_ALLOW, round(amin, 3), p.stock_mm,
                       "stock on machined surfaces (faced tops, reamed bore)", at_mm=at, machined_faces=len(mf)))

    (dmin, dat), under = draft_and_undercut(cast, per_face=3, draft_deg=p.core_draft_deg)
    rules.append(_rule("casting.draft", dmin >= p.core_draft_deg - TOL_DRAFT, round(dmin, 2), p.core_draft_deg,
                       "wall angle to the die stroke (+Y); cored holes 1.0 deg, outer walls 1.5 deg", at_mm=dat))
    rules.append(_rule("casting.undercut", not under, len(under), 0, "points from which the moving die cannot withdraw",
                       at_mm=under[:5]))
    bb = shot.bounding_box()
    upper = shot & Pos(0, (bb.max.Y + 1) / 2, (bb.min.Z + bb.max.Z) / 2) * Box(bb.size.X + 2, bb.max.Y + 1, bb.size.Z + 2)
    (gmin, gat), gunder = draft_and_undercut(upper, per_face=2, draft_deg=p.core_draft_deg)
    rules.append(_rule("shot.draft", gmin >= p.core_draft_deg - TOL_DRAFT and not gunder, round(gmin, 2),
                       p.core_draft_deg, "casting with runner, gates and overflows (above the parting plane) opens toward +Y, no undercut",
                       at_mm=gat, undercuts=len(gunder)))

    w = min_wall(cast, per_face=4)
    rules.append(_rule("casting.min_wall", w["min_mm"] >= MIN_WALL_MM - TOL_ALLOW, w["min_mm"], MIN_WALL_MM,
                       "thinnest measured wall", at_mm=w["at_mm"]))

    for name, die in (("moving", mv), ("cover", cv)):
        ok = die.is_valid and len(die.solids()) == 1
        rules.append(_rule(f"die.{name}.valid", ok, ok, True, "one valid solid"))
        ov = (die & cav).volume
        rules.append(_rule(f"die.{name}.clears_cavity", ov < 1e-2, round(ov, 4), 0.0, "mm3 overlap with the cavity"))
    ov = (mv & cv).volume
    rules.append(_rule("die.halves_clear", ov < 1e-2, round(ov, 4), 0.0, "mm3 overlap between moving and cover die"))

    # ejector pins: the disc just under each pin face is all metal (the pin pushes on a flat
    # face, nowhere on an edge or a wall), the pin itself never enters the cavity
    rp = dp.pin_d_mm / 2
    seat, cut = [], 0.0
    for (x, z, y), pin in zip(ds["pin_sites"], pins):
        disc = Pos(x * k, y * k, z * k) * Cylinder(rp, 0.2, rotation=(90, 0, 0), align=(Align.CENTER, Align.CENTER, Align.MIN))
        seat.append(round((disc & cav).volume / disc.volume, 4))
        cut += (pin & cav).volume + (pin & mv).volume
    # a pin is on the casting when the casting itself is under its face (overflow wells can
    # have the plate's height)
    on_part = sum(1 for x, z, y in ds["pin_sites"] if cast.is_inside(Vector(x, y - 0.05, z)))
    rules.append(_rule("ejector.pins_seated", min(seat, default=0.0) > 0.999, min(seat, default=0.0), 1.0,
                       "fraction of each pin face backed by the shot", per_pin=seat))
    rules.append(_rule("ejector.pins_clear", cut < 1e-2, round(cut, 4), 0.0, "mm3 of pin inside the cavity or the die"))
    rules.append(_rule("ejector.pins_on_part", on_part >= 4, on_part, 4,
                       "pins on the casting itself (the rest on overflows and runner)"))

    chm, chc = ds["channels_moving"], ds["channels_cover"]
    dcav = _min_dist(chm + chc, [cav])
    rules.append(_rule("cooling.to_cavity", dcav >= COOL_TO_CAVITY * dp.cool_d_mm - 1e-3, round(dcav, 2),
                       COOL_TO_CAVITY * dp.cool_d_mm, "mm, channel wall to cavity (>= 1.5 d)"))
    dpin = _min_dist(chm, pins)
    rules.append(_rule("cooling.to_pins", dpin >= COOL_TO_PIN_MM, round(dpin, 2), COOL_TO_PIN_MM,
                       "mm, channel wall to ejector pin"))
    xs, zs = dc.sleeve_center(p, dp)
    sleeve = Pos(xs * k, 1, zs * k) * Cylinder(dp.biscuit_d_mm / 2 * k, dp.cover_t_mm + 2, rotation=(90, 0, 0),
                                               align=(Align.CENTER, Align.CENTER, Align.MIN))
    dsl = _min_dist(chc, [sleeve])
    rules.append(_rule("cooling.to_sleeve", dsl >= COOL_TO_SLEEVE * dp.cool_d_mm - 1e-3, round(dsl, 2),
                       COOL_TO_SLEEVE * dp.cool_d_mm, "mm, channel wall to shot sleeve bore (>= 1.5 d)"))
    # steel left between each channel and the back face of its block (else it breaks through)
    rc = dp.cool_d_mm / 2
    hb = ds["block_mm"][1]
    back = min([hb - (c.center().Y + rc) for c in chm] + [dp.cover_t_mm + (c.center().Y - rc) for c in chc],
               default=float("inf"))
    rules.append(_rule("cooling.enclosed", back >= COOL_TO_BACK * dp.cool_d_mm - 1e-3, round(back, 2),
                       COOL_TO_BACK * dp.cool_d_mm, "mm, channel wall to the back face of its die block (>= 1 d)"))
    rules.append(_rule("cooling.present", len(chm) >= 3 and len(chc) >= 3, [len(chm), len(chc)], [3, 3],
                       "channels in the moving / cover die"))

    # process estimates
    proj = section2d(shot, 1e-3).union(section2d(shot, -1e-3))
    area = proj.area * k ** 2                                   # the hot cavity, not the cold shot
    f_open = dp.pressure_MPa * area / 1000                     # kN
    f_clamp = dp.clamp_margin * f_open
    t_clamp = f_clamp / 9.81
    machine = next((m for m in dc.MACHINES_T if m >= t_clamp), None)
    g = dc.gating_sizes(p, dp, cast.volume)
    rho = bracket.density(p)
    steel, srho = dc.DIE_STEEL
    v_gate = g["volume_mm3"] / (g["gate_area_mm2"] * g["fill_time_s"]) / 1000
    rules.append(_rule("process.machine", machine is not None, machine, dc.MACHINES_T[-1],
                       "smallest standard machine (tf) with clamp force >= margin x opening force"))
    m_fin, m_cast, m_shot = (s.volume / 1000 * rho for s in (fin, cast, shot))
    rules.append(_rule("part.mass_g", m_fin <= p.mass_target_g, round(m_fin, 1), p.mass_target_g,
                       "finished part mass, parameter mass_target_g"))
    return {
        "rules": rules, "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]],
        "process": {
            "process": "cold-chamber high-pressure die casting",
            "parting_plane": "Y = 0 (bracket back face); moving die at Y >= 0 holds the cavity and cores",
            "shrink_pct": dp.shrink_pct, "shrink_factor": round(k, 5),
            "projected_area_mm2": round(area, 0),
            "cavity_pressure_MPa": dp.pressure_MPa,
            "opening_force_kN": round(f_open, 0), "clamp_force_required_t": round(t_clamp, 0),
            "machine_t": machine, "clamp_basis": f"{dp.clamp_margin} x pressure x projected area incl. gating (estimate)",
            "fill_time_s": round(g["fill_time_s"], 4),
            "fill_time_basis": "NADCA t = K T (Ti - Tf + S Z) / (Tf - Td), T = plate wall (estimate)",
            "gate_area_mm2": round(g["gate_area_mm2"], 1), "gate_w_mm": round(g["gate_w_mm"], 1),
            "gate_t_mm": dp.gate_t_mm, "gate_velocity_m_s": round(v_gate, 1),
            "runner_area_mm2": round(g["runner_area_mm2"], 1),
            "ejector_pins": len(pins), "cooling_channels": [len(chm), len(chc)],
            "mass_finished_g": round(m_fin, 1), "mass_casting_g": round(m_cast, 1),
            "mass_shot_g": round(m_shot, 1), "casting_yield_pct": round(100 * m_cast / m_shot, 1),
            "material_yield_pct": round(100 * m_fin / m_shot, 1),
            "die_steel": steel,
            "die_mass_kg": round((mv.volume + cv.volume) / 1e6 * srho, 1),
            "die_block_mm": [round(v, 1) for v in (ds["block_mm"][0], ds["block_mm"][1] + dp.cover_t_mm, ds["block_mm"][2])],
        },
    }
