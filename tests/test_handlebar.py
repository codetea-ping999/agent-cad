import dataclasses
import numpy as np
import pytest
from agent_cad.parts import handlebar as hb, handlebar_drawing
from agent_cad.process import tube_bending as tb
from agent_cad.verify import handlebar_verify as hv
from agent_cad.verify.tube_check import verify_process
from agent_cad.drawing import tool
from agent_cad.drawing.check import check_sheet

P = hb.HandlebarParams()
TP = tb.TubeProcessParams()


@pytest.fixture(scope="module")
def bar():
    return hb.build(P)


@pytest.fixture(scope="module")
def tools():
    return tb.tool_set(P, TP)


def test_fit_numbers_are_met():
    sol = hb.solve(P)
    m = hb._measures(P, (sol.L0, sol.L1, sol.L2, sol.L3))
    assert np.allclose(m, (P.width, P.reach, P.drop), atol=1e-6)


def test_bar_is_one_valid_solid_near_target_mass(bar):
    assert bar.is_valid and len(bar.solids()) == 1
    m = hb.mass_g(P, bar)
    assert 260 < m <= P.mass_target_g
    bb = bar.bounding_box()
    assert abs(bb.size.Y - (P.width + P.grip_d)) < 1.0          # width c-c + grip OD


def test_parameters_change_the_bar():
    wide = dataclasses.replace(P, width=440.0)
    s0, s1 = hb.solve(P), hb.solve(wide)
    assert s1.L0 > s0.L0
    assert abs(hb._measures(wide, (s1.L0, s1.L1, s1.L2, s1.L3))[0] - 440.0) < 1e-6
    assert hb.density(dataclasses.replace(P, material="A6061-T6")) < hb.density(P)


def test_wall_steps_let_the_mandrel_out():
    assert hb.sunk_wall(P, P.grip_d) >= P.wall_butt >= P.wall_grip
    rep = hv.verify(P, with_shape=False)                       # beam model only (fast)
    assert {"fit.clamp_zone", "fit.clamp_d", "wall.extrados_after_bending"} <= {r["check"] for r in rep["rules"]}
    assert all(r["pass"] for r in rep["rules"] if r["check"].startswith(("fit.", "wall.")))


def test_6061_at_the_same_walls_is_weaker():
    a = hv.allowable(P, "fatigue")[0]
    b = hv.allowable(dataclasses.replace(P, material="A6061-T6"), "fatigue")[0]
    assert b < a


def test_process_and_tools_pass(tools):
    rep = verify_process(P, TP, tools)
    assert rep["pass"], rep["failed"]
    names = {r["check"] for r in rep["rules"]}
    assert {"swage.passes", "bend.RB2.clamp_grip", "bend.RB2.machine_hand", "tool.clamp_vs_bend_die",
            "tool.wiper.no_tube_overlap", "tool.mandrel_ball_fit"} <= names
    b2 = [b for b in rep["bending"] if b["side"] == "R" and b["bend"] == "B2"][0]
    assert b2["contoured_clamp"]                               # the hood straight is too short to clamp


def test_tool_part_drawings(tools):
    parts = tb.tool_parts(P, TP, tools)
    assert set(parts) >= {"bend_die", "clamp", "insert", "pressure_die", "wiper", "mandrel", "swage_die"}
    for k in ("bend_die", "clamp", "mandrel"):
        t = parts[k]
        assert t["shape"].is_valid and len(t["shape"].solids()) == 1
        info = dict(number="AC-TEST", title=k, material=t["material"][0], finish=t["material"][1], up=1,
                    block=t["block"], holes=t["holes"], density_g_cm3=t["density"])
        sh = tool.component_sheet(t["shape"], info)
        assert check_sheet(sh)["pass"], k


def test_part_drawing_passes(bar):
    sh = handlebar_drawing.part_sheet(P, bar, "AC-TEST")
    chk = check_sheet(sh)
    assert chk["pass"], [c["check"] for c in chk["checks"] if not c["pass"]]
    names = {d["name"] for d in sh.meta["dims"]}
    assert {"width", "drop", "reach", "clamp_d", "grip_d"} <= names
