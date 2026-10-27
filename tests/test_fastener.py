import dataclasses, math
import pytest
from agent_cad import fastener as fa
from agent_cad.fastener import BOLTS, Plate, Joint, calc
from agent_cad.parts import stem
from agent_cad.parts.handlebar import HandlebarParams
from agent_cad.parts.bracket import BracketParams
from agent_cad.parts.housing import HousingParams
from agent_cad.verify import fastener_check as fc


def test_thread_geometry_matches_jis():
    b = BOLTS["M5-10.9"]
    assert abs(b.d2 - 4.480) < 1e-3 and abs(b.d3 - 4.019) < 1e-3
    assert abs(b.As - 14.2) < 0.05                       # JIS B 1082: M5 14.2 mm2
    assert abs(BOLTS["M8-8.8"].As - 36.6) < 0.05         # M8 36.6 mm2


def test_allowed_preload_puts_assembly_stress_at_nu():
    b = BOLTS["M8-8.8"]
    F = fa.FM_allowed(b, 0.12, 0.9)
    s, _, _ = fa.assembly_stress(b, F, 0.12)
    assert abs(s / b.Rp02_MPa - 0.9) < 1e-9
    # the order of VDI 2230 Table A1 for M8 8.8 (about 18 kN at mu_G 0.12)
    assert 16_000 < F < 20_000


def test_torque_coefficient_is_the_usual_0_2():
    b = BOLTS["M8-8.8"]
    K = fa.torque_factor(b, (13 + 9) / 2, 0.15, 0.15) / b.d
    assert 0.18 < K < 0.22


def _joint(**kw):
    base = dict(name="t", bolt=BOLTS["M5-10.9"], n=1, torque_Nm=(5.0, 6.0), plates=(Plate(6.0, 70_000, 14.0),),
                hole_d=5.5, tapped=True, engagement_mm=8.0, thread_Rm_MPa=310, bearing_Rm_MPa=310)
    base.update(kw)
    return Joint(**base)


def test_more_torque_more_preload_less_margin_on_the_bolt():
    lo, hi = calc(_joint(torque_Nm=(4, 5))), calc(_joint(torque_Nm=(8, 9)))
    assert hi["preload"]["FM_min"] > lo["preload"]["FM_min"]
    assert hi["stress"]["nu"] > lo["stress"]["nu"]


def test_window_edges_are_where_the_checks_flip():
    j = _joint(F_req_N=3000.0)
    w = calc(j)["torque_window_Nm"]
    assert w["feasible"]
    assert calc(dataclasses.replace(j, torque_Nm=(w["min"] * 1.01, w["min"] * 1.02)))["rules"][2]["pass"]
    assert not calc(dataclasses.replace(j, torque_Nm=(w["min"] * 0.98, w["min"] * 0.99)))["rules"][2]["pass"]
    assert calc(dataclasses.replace(j, torque_Nm=(w["max"] * 0.98, w["max"] * 0.99)))["rules"][0]["pass"]
    assert not calc(dataclasses.replace(j, torque_Nm=(w["max"] * 1.01, w["max"] * 1.02)))["rules"][0]["pass"]


def test_compliant_clamp_side_loses_less_preload_to_embedding():
    stiff, soft = calc(_joint()), calc(_joint(extra_compliance=2e-5))
    assert soft["preload"]["FZ"] < stiff["preload"]["FZ"]
    assert soft["load_factor"]["Phi_n"] > stiff["load_factor"]["Phi_n"]


def test_short_engagement_strips_before_the_bolt_breaks():
    assert not calc(_joint(engagement_mm=4.0))["pass"]
    assert calc(_joint(engagement_mm=10.0))["rules"][-1]["pass"]


def test_stem_pinch_holds_the_steerer_torque_with_the_drawing_torque():
    r = fc.stem_joints(stem.StemParams(), HandlebarParams(), fea=False)
    pj = r["joints"]["pinch"]
    assert pj["pass"], pj["failed"]
    assert pj["torque_Nm"] == list(fc.TORQUE_SPEC["pinch"])
    assert pj["capacity"]["slip_safety"] >= fc.SG_SLIP_STATIC
    assert r["bar_interface"]["pass"]


def test_pinch_thread_in_6061_is_marginal_for_a_12_9_bolt():
    """Why the calculation specifies 10.9: 7.5 mm of A6061 thread strips just before a 12.9
    bolt breaks, a 10.9 bolt breaks first."""
    p = stem.StemParams()
    assert not fc.calc(fc.pinch_joint(p, bolt="M5-12.9"))["rules"][-1]["pass"]
    assert fc.calc(fc.pinch_joint(p, bolt="M5-10.9"))["rules"][-1]["pass"]


def test_faceplate_needs_about_5_Nm():
    """The bar torque demand (provisional test torque, hood fatigue peak) needs the
    faceplate bolts at >= ~4.9 N.m with the FEA clamp compliance."""
    p = stem.StemParams()
    c = fc.faceplate_clamp_compliance(p, h=2.0)
    assert 4e-6 < c < 7e-6
    j, need = fc.faceplate_joint(p, HandlebarParams(), fea_compliance=c)
    assert 70 < need["fatigue_peak_Nm"] < 80
    w = fc.calc(j)["torque_window_Nm"]
    assert 4.5 < w["min"] < 5.2 and w["max"] > 7.5
    assert fc.calc(dataclasses.replace(j, torque_Nm=(5.0, 6.0)))["pass"]


def test_torque_notes_follow_the_spec():
    t, bolts = fc.torque_notes()
    (a, b), (c, d) = fc.TORQUE_SPEC["pinch"], fc.TORQUE_SPEC["faceplate"]
    assert f"{a:g}〜{b:g} N·m" in t and f"{c:g}〜{d:g} N·m" in t
    assert "10.9" in bolts


def test_stem_joints_pass_with_the_drawing_spec():
    """AC-1001 rev A: faceplate 5-6 N·m and 10.9 bolts pass every joint check."""
    r = fc.stem_joints(stem.StemParams(), HandlebarParams())
    assert r["pass"], r["failed"]


def test_bracket_and_housing():
    r = fc.bracket_joints(BracketParams())
    assert r["pass"], r["failed"]
    j = r["joints"]["bracket"]
    assert j["torque_window_Nm"]["min"] <= fc.TORQUE_SPEC["bracket"][0]
    assert fc.housing_screws(HousingParams())["pass"]
    # a hole too wide for the screw fails the guide
    assert not fc.housing_screws(dataclasses.replace(HousingParams(), hole_d=3.8))["pass"]
