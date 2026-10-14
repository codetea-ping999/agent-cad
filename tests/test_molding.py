import dataclasses
import pytest
from build123d import Vector
from agent_cad.parts import housing, housing_drawing
from agent_cad.process import molding as mo
from agent_cad.verify import molding_check as mc
from agent_cad.drawing.check import check_sheet

P = housing.HousingParams()
MP = mo.MouldParams()


@pytest.fixture(scope="module")
def made():
    ms = mo.mould_set(P, MP)
    return ms, mc.verify_molding(P, MP, ms=ms)


def test_housing_and_mould_pass(made):
    _, rep = made
    assert rep["pass"], rep["failed"]
    assert {r["check"] for r in rep["rules"]} >= {
        "release.formed", "release.draft", "release.undercut", "release.slide_forms_window", "wall.min",
        "mould.components_clear", "mould.clears_cavity", "ejector.seated", "slide.stroke", "slide.clears_ejection",
        "cooling.to_cavity", "cooling.to_pins", "process.machine", "process.gate_shear"}


def test_process_numbers_are_sane(made):
    _, rep = made
    pr = rep["process"]
    assert pr["mass_part_g"] < pr["mass_shot_g"] <= P.mass_target_g
    assert 80 < pr["yield_pct"] < 100
    assert pr["machine_t"] == 50                          # ~7 300 mm2 at 40 MPa x 1.2 = 36 t
    assert 1.0 < pr["cooling_time_s"] < 10.0               # 2.5 mm PA66-GF30
    assert pr["gate_shear_1_s"] <= mo.RESINS["PA66-GF30"].max_shear_1_s
    assert pr["slide_stroke_mm"] > pr["slide_insert_depth_mm"]
    assert pr["draft_min_by_component_deg"]["fixed"] >= P.draft_deg - 0.05
    assert pr["surface_by_component_mm2"]["slide"] > 0


def test_material_sets_density_and_shrinkage():
    assert abs(housing.density(P) - 1.37) < 1e-9
    pp = dataclasses.replace(P, material="PP")
    assert abs(housing.density(pp) - 0.905) < 1e-9
    assert abs(mo.shrink_factor(pp) - 1.016) < 1e-12 and abs(mo.shrink_factor(P) - 1.005) < 1e-12


def test_part_features_are_where_the_parameters_put_them(made):
    ms, _ = made
    part = ms["part"]
    zw = -P.width / 2 + P.wall / 2
    assert not part.is_inside(Vector(P.window_x, P.window_y, zw))           # the window goes through the wall
    assert part.is_inside(Vector(P.window_x + P.window_w, P.window_y, zw))
    x, z = housing.bosses(P)[0]
    assert not part.is_inside(Vector(x, P.boss_recess + 1, z))               # screw hole
    assert part.is_inside(Vector(x + P.hole_d / 2 + 1, P.boss_recess + 1, z))
    assert part.is_inside(Vector(x, P.boss_recess + P.hole_depth + 0.5, z))      # blind: ends in the boss
    assert part.is_inside(Vector(x, housing.floor_y(P) + 0.5, z))


def test_window_is_an_undercut_without_the_slide(made):
    """The same mould with the slide merged into the fixed half: the window can no
    longer be released (its faces see the fixed half's pull direction)."""
    ms, _ = made
    k = mo.shrink_factor(P)
    comps = [(kd, s) for kd, s in mc.components(ms) if kd != "slide"]
    comps = [(kd, (s + ms["slide"]) if kd == "fixed" else s) for kd, s in comps]
    rel = mc.release(ms["part"], comps, k, per_face=2)
    blocked = [q for q, kd, _, b, _ in rel["points"] if b]
    assert blocked
    wx0, wx1 = P.window_x - P.window_w / 2 - 0.5, P.window_x + P.window_w / 2 + 0.5
    assert all(wx0 <= q[0] <= wx1 and q[2] < -P.width / 2 + P.wall + 0.5 for q in blocked)


def test_slide_stroke_follows_the_angle_pin():
    assert mo.slide_travel(P, dataclasses.replace(MP, angle_pin_deg=22, lock_deg=24)) > mo.slide_travel(P, MP)
    short = dataclasses.replace(MP, angle_pin_deg=8, lock_deg=10)
    assert mo.slide_travel(P, short) < P.wall + MP.slide_safety_mm      # too shallow a pin cannot clear the window


def test_ejectors_and_cooling(made):
    ms, rep = made
    r = {x["check"]: x for x in rep["rules"]}
    assert len(ms["sleeves"]) == 4 and len(ms["pins"]) >= 8
    assert r["ejector.seated"]["value"] > 0.999
    assert r["cooling.to_cavity"]["value"] >= 1.5 * MP.cool_d_mm
    assert r["cooling.bubbler_to_cavity"]["value"] >= 1.5 * MP.bubbler_d_mm
    assert r["cooling.to_pins"]["value"] >= mc.COOL_TO_PIN_MM


def test_gate_is_sized_by_shear_rate():
    h, w = mo.gate_size(P, MP, 45_000)
    h2, w2 = mo.gate_size(P, dataclasses.replace(MP, fill_time_s=0.4), 45_000)
    assert h == h2 == pytest.approx(0.6 * P.wall) and w2 > w        # faster fill -> wider gate


def test_drawing_passes_and_catches_a_model_mismatch(made):
    ms, _ = made
    sh = housing_drawing.part_sheet(P, ms["part"], dwg_no="AC-HSG-TEST")
    rep = check_sheet(sh)
    assert rep["pass"], [c for c in rep["checks"] if not c["pass"]]
    names = {d["name"] for d in sh.meta["dims"]}
    assert {"length", "width", "height", "window_w", "window_h", "boss_pitch_x", "hole_depth", "rib_h", "floor"} <= names
    other = housing.build(dataclasses.replace(P, window_w=20.0, hole_depth=10.0))
    bad = check_sheet(housing_drawing.part_sheet(P, other, dwg_no="AC-HSG-TEST"))
    failed = {c["check"] for c in bad["checks"] if not c["pass"]}
    assert {"dim.window_w", "dim.hole_depth"} <= failed


# default housing's part volume per geometry revision: a change to housing.build() that moves
# it must bump housing.GEOMETRY_REV (it is in the document number)
GEOMETRY_FINGERPRINT = {1: 44975.6}


def test_document_number_follows_the_geometry_revision(made, monkeypatch):
    from scripts.make_molding import doc_no
    ms, _ = made
    assert GEOMETRY_FINGERPRINT[housing.GEOMETRY_REV] == round(ms["part"].volume, 1)
    before = doc_no(P)
    monkeypatch.setattr(housing, "GEOMETRY_REV", housing.GEOMETRY_REV + 1)
    assert doc_no(P) != before
