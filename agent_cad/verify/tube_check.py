"""Checks of the handlebar's manufacturing route and tooling (process/tube_bending.py).

Each rule records value, limit and basis like the part checks. The rules of thumb (swaging
reduction per pass, mandrel chart, clamp length) are what a tube-bending shop would ask
for first; the springback and torque are estimates to set the machine, not results.
"""
import math
import numpy as np
from ..parts import handlebar as hb
from ..process import tube_bending as tb

MAX_PASSES = 4                 # rotary swaging passes we accept for one end
MAX_DIE_RELIEF = 3.0           # relief cut into the bend die for a leading part (mm)
MIN_TABLE_CLEAR = 10.0
MIN_TUBE_CLEAR = 5.0
MAX_OVERBEND_DEG = 5.0
BENDER_TORQUE_NM = 600.0       # small CNC rotary-draw bender (e.g. 32 mm class)


def _overlap(a, b):
    r = a & b
    return 0.0 if r is None else float(r.volume)


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def verify_process(p, tp=tb.TubeProcessParams(), tools=None):
    rules = []
    bl, sw = tb.blank(p, tp), tb.swaging(p, tp)
    rules.append(_rule("swage.passes", sw["passes"] <= MAX_PASSES, sw["passes"], MAX_PASSES,
                       f"reduction of area {sw['reduction_of_area']:.0%}, <= {tb.MAX_RA_PER_PASS:.0%} per pass"))
    t0 = hb.sunk_wall(p, p.grip_d)
    rules.append(_rule("swage.mandrel_withdrawable", t0 >= p.wall_butt >= p.wall_grip,
                       [round(t0, 3), p.wall_butt, p.wall_grip], "sunk >= butt >= grip wall",
                       "the mandrel leaves through the open end: the bore never shrinks toward it"))
    plan = tb.bending_plan(p, tp)
    for b in plan:
        key = f"bend.{b['side']}{b['bend']}"
        rules.append(_rule(f"{key}.clamp_grip", max(b["straight_grip"], b["contoured_grip"]) >= b["clamp_needed"] - 1e-6,
                           round(max(b["straight_grip"], b["contoured_grip"]), 2), b["clamp_needed"],
                           ("contoured clamp pair following the leading part" if b["contoured_clamp"]
                            else "straight clamp") + f", {tp.clamp_len_D:g} D"))
        rules.append(_rule(f"{key}.trailing_straight", b["trailing_straight"] >= b["pressure_die"],
                           b["trailing_straight"], b["pressure_die"],
                           "unbent tube behind the bend >= pressure die (arc + "
                           f"{tp.pd_extra_D:g} D); mandrel enters from the open end"))
        sw_ = b["sweep"][b["machine_up"]] if b["machine_up"] else b["sweep"][1]
        rules.append(_rule(f"{key}.machine_hand", b["machine_up"] != 0, b["machine_up"], "+1 or -1",
                           "a machine orientation (right/left-hand bending) where the swinging part clears"))
        rules.append(_rule(f"{key}.clear_table", (sw_["table"] or 1e9) >= MIN_TABLE_CLEAR, sw_["table"], MIN_TABLE_CLEAR,
                           "swinging part above the machine table (mm)"))
        rules.append(_rule(f"{key}.clear_tube", (sw_["trailing_tube"] or 1e9) >= MIN_TUBE_CLEAR, sw_["trailing_tube"],
                           MIN_TUBE_CLEAR, "swinging part to the straight still to be bent (mm)"))
        rules.append(_rule(f"{key}.die_relief", sw_["die_relief"] <= MAX_DIE_RELIEF,
                           sw_["die_relief"], MAX_DIE_RELIEF,
                           "relief cut into the bend die rim for the leading part (mm)"))
        rules.append(_rule(f"{key}.overbend", b["springback"]["overbend_deg"] <= MAX_OVERBEND_DEG,
                           b["springback"]["overbend_deg"], MAX_OVERBEND_DEG, "springback in the bending temper"))
        rules.append(_rule(f"{key}.torque", b["torque_Nm"] <= BENDER_TORQUE_NM, b["torque_Nm"], BENDER_TORQUE_NM,
                           "1.3 x UTS x plastic modulus (bending temper)"))
    if tools is not None:
        for k in ("bend_die", "clamp", "insert", "clamp_straight", "insert_straight", "pressure_die", "wiper",
                  "mandrel_shank", "swage_die", "swage_mandrel"):
            s = tools[k]
            rules.append(_rule(f"tool.{k}.valid", s.is_valid and len(s.solids()) == 1, len(s.solids()), 1,
                               "one valid closed solid"))
        # tools must not cut into the tube at the set-up (groove clearance), and the clamp pair
        # must not overlap each other or the die
        tube = tools["tube"]
        for k in ("bend_die", "clamp", "insert", "pressure_die", "wiper"):
            v = _overlap(tools[k], tube)
            rules.append(_rule(f"tool.{k}.no_tube_overlap", v < 1e-3, round(v, 4), 0.0, "mm3 overlap with the tube"))
        for a, b in (("clamp", "insert"), ("clamp", "bend_die"), ("insert", "bend_die"), ("wiper", "bend_die"),
                     ("pressure_die", "bend_die"), ("wiper", "pressure_die")):
            v = _overlap(tools[a], tools[b])
            rules.append(_rule(f"tool.{a}_vs_{b}", v < 1e-3, round(v, 4), 0.0, "mm3 overlap between tools"))
        bore = p.grip_d - 2 * p.wall_grip
        balls = tools["mandrel_balls"]
        d_ball = 2 * (balls[0].bounding_box().max.X - balls[0].bounding_box().min.X) / 2 if balls else 0
        rules.append(_rule("tool.mandrel_ball_fit", (not balls) or d_ball < bore, round(d_ball, 3), round(bore, 3),
                           "ball diameter < bore"))
    mr = plan[0]["mandrel"]
    rows, verts = tb.lra_table(p, 1)
    return {"blank": bl, "swaging": sw, "bending": plan, "lra": rows, "vertices": verts, "mandrel": mr,
            "rules": rules, "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]]}
