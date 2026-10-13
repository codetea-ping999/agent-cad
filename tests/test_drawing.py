"""Drawings: dimensions read from the model match the design intent, layout is
clean, and the outputs open."""
import pytest
import ezdxf
from agent_cad.parts import stem, stem_drawing
from agent_cad.drawing.check import check_sheet
from agent_cad.drawing.render import to_pdf, to_dxf
from agent_cad.standards import jis

CASES = [stem.StemParams(), stem.StemParams(length=110, angle_deg=-17, material="A7075-T6", mass_target_g=165)]


@pytest.fixture(scope="module", params=range(len(CASES)))
def built(request):
    p = CASES[request.param]
    b, f = stem.build(p)
    return p, b, f


def failed(rep):
    return [c for c in rep["checks"] if not c["pass"]]


def test_body_sheet_checks(built):
    p, b, f = built
    rep = check_sheet(stem_drawing.body_sheet(p, b))
    assert rep["pass"], failed(rep)


def test_faceplate_sheet_checks(built):
    p, b, f = built
    rep = check_sheet(stem_drawing.faceplate_sheet(p, f))
    assert rep["pass"], failed(rep)


def test_model_drawing_mismatch_is_caught():
    """A model built with a different bore than the parameters claim must fail."""
    p = stem.StemParams()
    b, f = stem.build(stem.StemParams(steerer_d=28.2))
    rep = check_sheet(stem_drawing.body_sheet(p, b))
    assert any(c["check"] == "dim.steerer_bore_d" for c in failed(rep))


def test_outputs_open(tmp_path):
    p = CASES[0]
    b, f = stem.build(p)
    sh = stem_drawing.faceplate_sheet(p, f)
    to_pdf(sh, tmp_path / "f.pdf"); to_dxf(sh, tmp_path / "f.dxf")
    assert (tmp_path / "f.pdf").read_bytes()[:4] == b"%PDF"
    doc = ezdxf.readfile(tmp_path / "f.dxf")
    auditor = doc.audit()
    assert not auditor.has_errors
    assert len(doc.modelspace().query("LWPOLYLINE")) > 50 and len(doc.modelspace().query("TEXT")) > 20


def test_jis_tables():
    assert jis.fit_deviations(28.6, "F8") == (0.053, 0.020)
    assert jis.fit_deviations(31.8, "F8") == (0.064, 0.025)
    assert jis.fit_deviations(25, "H7") == (0.021, 0.0)
    assert jis.fit_deviations(25, "k6") == (0.015, 0.002)
    assert jis.general_linear(40, "m") == 0.3 and jis.general_linear(5, "m") == 0.1


def _no_nan(sheet):
    import numpy as np
    for it in sheet.items:
        if it[0] in ("line", "fill"):
            assert np.isfinite(it[1]).all(), it


def test_zero_angle_stem_draws():
    p = stem.StemParams(angle_deg=0)
    b, f = stem.build(p)
    sh = stem_drawing.body_sheet(p, b)
    _no_nan(sh)
    rep = check_sheet(sh)
    assert rep["pass"], failed(rep)


def test_window_is_measured_from_the_solid():
    p = stem.StemParams()
    b, f = stem.build(p)
    dims = {d["name"]: d for d in stem_drawing.faceplate_sheet(p, f).meta["dims"]}
    assert abs(dims["window_w"]["measured"] - p.fp_window_w) < 1e-3 and abs(dims["window_h"]["measured"] - p.fp_window_h) < 1e-3
    # parameters claim a window, but the model has none: caught
    b2, f2 = stem.build(stem.StemParams(fp_window_w=0))
    assert {"dim.window_w", "dim.window_h"} <= {c["check"] for c in failed(check_sheet(stem_drawing.faceplate_sheet(p, f2)))}
    # w > 0 with h == 0 builds no window: no callout and nothing to check
    p3 = stem.StemParams(fp_window_h=0)
    sh = stem_drawing.faceplate_sheet(p3, stem.build(p3)[1])
    assert not any("窓" in it[3] for it in sh.items if it[0] == "text") and check_sheet(sh)["pass"]


def test_depths_and_notes_follow_the_model():
    p = stem.StemParams(fp_gap=2.0, bar_d=35.0)
    b, f = stem.build(p)
    body = stem_drawing.body_sheet(p, b)
    dims = {d["name"]: d for d in body.meta["dims"]}
    assert abs(dims["fp_tap_drill_depth"]["measured"] - p.tap_depth) < 1e-3
    assert abs(dims["pinch_cbore_depth"]["measured"] - p.cbore_depth) < 1e-3
    assert check_sheet(body)["pass"]
    notes = " ".join(stem_drawing.faceplate_sheet(p, f).meta["notes"])
    assert "φ35 " in notes and "すき間 2 " in notes and "31.8" not in notes


def test_variants_get_distinct_drawing_numbers():
    nos = {stem_drawing.doc_no(p) for p in CASES}
    assert len(nos) == len(CASES)
    assert stem_drawing.doc_no(stem.StemParams()) == stem_drawing.doc_no(stem.StemParams())


def test_large_faceplate_falls_back_to_fit():
    p = stem.StemParams(fp_bolt_off=40, boss_margin=8, fp_bolt_w=20)
    sh = stem_drawing.faceplate_sheet(p, stem.build(p)[1])
    rep = check_sheet(sh)
    assert rep["pass"], failed(rep)


def test_pdf_requires_japanese_font(monkeypatch, tmp_path):
    from agent_cad.drawing import render
    monkeypatch.setattr(render, "_JP_FONTS", ["No Such Font JP"])
    p = CASES[0]
    sh = stem_drawing.faceplate_sheet(p, stem.build(p)[1])
    with pytest.raises(RuntimeError, match="Japanese font"):
        to_pdf(sh, tmp_path / "x.pdf")


def test_missing_features_fail_the_check():
    """Parameters claim M5 taps (φ4.2), the model was drilled φ3.3: the tap pattern is
    not found, and the sheet must fail instead of silently dropping the callouts."""
    p = stem.StemParams()
    b, f = stem.build(stem.StemParams(tap_drill_d=3.3))
    assert "feature.fp_taps" in {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b)))}


def test_fastener_diameters_are_checked():
    p = stem.StemParams()
    b, f = stem.build(stem.StemParams(bolt_clear_d=5.54, cbore_d=8.84))
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.faceplate_sheet(p, f)))}
    assert {"dim.bolt_clear_d", "dim.cbore_d"} <= bad


def test_body_sections_follow_the_loft():
    p = stem.StemParams()
    b, f = stem.build(p)
    dims = {d["name"] for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    assert {"section_D_w", "section_D_h", "section_E_w", "section_E_h"} <= dims
    b2, _ = stem.build(stem.StemParams(body_w=23.0))
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b2)))}
    assert "dim.section_D_w" in bad


def test_projection_symbol_is_third_angle():
    from agent_cad.drawing.sheet import Sheet
    import numpy as np
    sh = Sheet(); sh.projection_symbol((100, 100))
    lines = [it[1] for it in sh.items if it[0] == "line" and it[2] == "thin"]
    circles = [l for l in lines if len(l) > 20]
    cone = [l for l in lines if len(l) == 5][0]
    assert min(c[:, 0].min() for c in circles) > cone[:, 0].max()      # end view right of the cone
    left = cone[cone[:, 0] < cone[:, 0].mean()]
    right = cone[cone[:, 0] > cone[:, 0].mean()]
    assert np.ptp(right[:, 1]) < np.ptp(left[:, 1])                       # small end faces the circles


def test_zero_gap_faceplate_draws():
    p = stem.StemParams(fp_gap=0.0)
    rep = check_sheet(stem_drawing.faceplate_sheet(p, stem.build(p)[1]))
    assert rep["pass"], failed(rep)


def test_zero_angle_text_clears_reference_line():
    from agent_cad.drawing.sheet import Sheet, text_box
    sh = Sheet(); sh.angle_dim((100, 100), 0, 0, 50, "0°")
    t = [it for it in sh.items if it[0] == "text"][0]
    b0, _ = text_box((t[1], t[2]), t[3], t[4], t[5], t[6], t[7])
    assert b0[1] > 100                       # the whole text sits above the horizontal line


def test_unknown_material_needs_a_finish():
    p = stem.StemParams(material="S45C", density_g_cm3=7.85)
    rep = check_sheet(stem_drawing.body_sheet(p, stem.build(p)[0]))
    assert "content.finish" in {c["check"] for c in failed(rep)}


def test_dxf_uses_the_selected_font(tmp_path):
    from agent_cad.drawing import render
    p = CASES[0]
    sh = stem_drawing.faceplate_sheet(p, stem.build(p)[1])
    to_dxf(sh, tmp_path / "f.dxf")
    assert ezdxf.readfile(tmp_path / "f.dxf").styles.get("JP").dxf.font == render._font_file()


def test_drawing_numbers_do_not_collide_on_reported_pair():
    a = stem.StemParams(length=80.3, angle_deg=-8.3)
    b = stem.StemParams(length=81.7, angle_deg=-4.4)
    assert stem_drawing.doc_no(a) != stem_drawing.doc_no(b)


def test_bolt_pattern_needs_all_four_corners():
    ok = [(-12.5, -20.2), (12.5, -20.2), (-12.5, 20.2), (12.5, 20.2)]
    moved = ok[:3] + [(5.0, 10.0)]                      # spans unchanged, one hole inside the rectangle
    assert stem_drawing._corners(ok, 12.5, 20.2) == 4
    assert stem_drawing._corners(moved, 12.5, 20.2) == 3


def test_pinch_taps_are_measured():
    p = stem.StemParams()
    dims = {d["name"]: d for d in stem_drawing.body_sheet(p, stem.build(p)[0]).meta["dims"]}
    assert abs(dims["pinch_tap_drill_d"]["measured"] - p.tap_drill_d) < 1e-3
    assert abs(dims["pinch_tap_length"]["measured"] - (p.ear_half_w - p.slot_w / 2)) < 1e-3   # drilled through


def test_fit_deviations_follow_the_printed_nominal():
    """30.004 prints as φ30: the deviations must be those of the 18-30 band, as check_sheet reads them."""
    text, tol = stem_drawing._fit_text(30.004, "F8")
    up, lo = jis.fit_deviations(30.0, "F8")
    assert text == "φ30 F8" and tol == (f"{up:+.3f}", f"{lo:+.3f}")


def test_lightening_bore_of_another_size_is_reported():
    p = stem.StemParams()
    b, _ = stem.build(stem.StemParams(light_d=p.light_d + 2))
    rep = check_sheet(stem_drawing.body_sheet(p, b))          # must not raise
    bad = {c["check"] for c in failed(rep)}
    assert "dim.light_d" in bad and "feature.light_bore" not in bad


def test_body_retries_at_a_smaller_scale(monkeypatch):
    from agent_cad.drawing.sheet import A2
    p = CASES[0]
    b = stem.build(p)[0]
    monkeypatch.setattr(stem_drawing, "BODY_TRIES", [((150.0, 105.0), 1.0), (A2, 0.5)])   # 1:1 cannot fit
    sh = stem_drawing.body_sheet(p, b)
    assert sh.meta["title_block"]["尺度"] == "1:2" and len(sh.views) >= 4
    rep = check_sheet(sh)
    assert rep["pass"], failed(rep)
    monkeypatch.setattr(stem_drawing, "BODY_TRIES", [((150.0, 105.0), 1.0)])
    with pytest.raises(ValueError, match="do not fit"):
        stem_drawing.body_sheet(p, b)


def test_counterbores_must_sit_on_the_holes():
    from build123d import Cylinder, Plane, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    import numpy as np
    p = CASES[0]
    f = stem.build(p)[1]
    d, n, S, B = stem.frame(p)
    dv, nv = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z])
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    e0, e1 = c.axial_extent()
    base = c.loc + dv * (e0 - c.loc @ dv)

    def cyl(at):
        return Plane(origin=Vector(*at), z_dir=Vector(*dv)) * Cylinder(c.r, e1 - e0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = (f + cyl(base)) - cyl(base + nv * 2.0)             # one counterbore moved 2 mm off its hole
    rep = check_sheet(stem_drawing.faceplate_sheet(p, bad))
    assert "feature.cbore_pattern" in {c["check"] for c in failed(rep)}
    assert "feature.bolt_pattern" not in {c["check"] for c in failed(rep)}


def test_local_roughness_is_shown_and_checked():
    p = CASES[0]
    b, f = stem.build(p)
    fp = stem_drawing.faceplate_sheet(p, f)
    body = stem_drawing.body_sheet(p, b)
    assert {(r["name"], r["drawn"]) for r in fp.meta["roughness_local"]} >= {("mating_face", "1.6"), ("bar_bore", "1.6")}
    assert {r["name"] for r in body.meta["roughness_local"]} >= {"mating_face", "bar_bore", "steerer_bore"}
    fp.meta["roughness_local"][0]["drawn"] = None
    assert any(c["check"].startswith("content.roughness.") for c in failed(check_sheet(fp)))


def test_general_tolerance_bands_are_unambiguous():
    p = CASES[0]
    sh = stem_drawing.faceplate_sheet(p, stem.build(p)[1])
    texts = [it[3] for it in sh.items if it[0] == "text"]
    assert {"0.5以上", "3以下", "30を超え", "120以下"} <= set(texts)
    import re
    assert not any(re.fullmatch(r"[\d.]+〜[\d.]+", t) for t in texts)


def test_missing_primary_bore_gives_a_failed_sheet():
    from build123d import Box
    p = CASES[0]
    rep = check_sheet(stem_drawing.body_sheet(p, Box(60, 40, 40)))      # nothing to recognise, must not raise
    assert "feature.bar_bore" in {c["check"] for c in failed(rep)}
    rep = check_sheet(stem_drawing.faceplate_sheet(p, Box(10, 40, 40)))
    assert "feature.bar_bore" in {c["check"] for c in failed(rep)}


def test_stack_height_ignores_proud_pinch_bosses():
    p = stem.StemParams(stack_h=40, pinch_pitch=35)
    rep = check_sheet(stem_drawing.body_sheet(p, stem.build(p)[0]))
    assert "dim.stack_h" not in {c["check"] for c in failed(rep)}, failed(rep)


def test_every_hole_of_a_4x_callout_is_measured():
    from build123d import Cylinder, Plane, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    import numpy as np
    p = CASES[0]
    f = stem.build(p)[1]
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    hs = find_cyl(f, dv, r=p.bolt_clear_d / 2, inner=True)
    c = hs[-1]                                    # not the first one the old code looked at
    e0, e1 = c.axial_extent()
    base = c.loc + dv * (e0 - c.loc @ dv)
    bad = f - Plane(origin=Vector(*base), z_dir=Vector(*dv)) * Cylinder(c.r + 0.02, e1 - e0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    sh = stem_drawing.faceplate_sheet(p, bad)
    assert len([d_ for d_ in sh.meta["dims"] if d_["name"].startswith("bolt_clear_d")]) == 4
    assert any(c_["check"].startswith("dim.bolt_clear_d") for c_ in failed(check_sheet(sh)))


def test_round_window_of_bolt_size_is_not_a_bolt_hole():
    p = stem.StemParams(fp_window_w=5.5, fp_window_h=5.5)
    rep = check_sheet(stem_drawing.faceplate_sheet(p, stem.build(p)[1]))
    bad = {c["check"] for c in failed(rep)}
    assert not bad & {"feature.bolt_holes", "feature.bolt_pattern"}, failed(rep)


def test_slot_centre_and_faceplate_model_note():
    p = CASES[0]
    b, f = stem.build(p)
    dims = {d["name"]: d for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    assert abs(dims["slot_center_y"]["measured"]) < 1e-3 and dims["slot_center_y"]["intent"] == 0.0
    assert any("3Dモデル" in n for n in stem_drawing.faceplate_sheet(p, f).meta["notes"])


def test_window_radius_is_checked():
    p = CASES[0]
    dims = {d["name"]: d for d in stem_drawing.faceplate_sheet(p, stem.build(p)[1]).meta["dims"]}
    assert abs(dims["window_r"]["measured"] - min(p.fp_window_w, p.fp_window_h) / 2) < 1e-3


def test_step_products_carry_the_drawing_number():
    from scripts.make_stem import part_specs
    p = CASES[1]
    b, f = stem.build(p)
    body, fp = part_specs(p, b, f)
    assert body.name.startswith(stem_drawing.body_sheet(p, b).meta["title_block"]["図番"])
    assert fp.name.startswith(stem_drawing.doc_no(p) + "-02")


def _refill_and_move(shape, c, shift):
    """Fill the cylindrical hole `c` and cut the same hole again, moved by `shift`."""
    from build123d import Cylinder, Plane, Align, Vector
    import numpy as np
    e0, e1 = c.axial_extent()
    base = c.loc + c.axis * (e0 - c.loc @ c.axis)

    def cyl(at):
        return Plane(origin=Vector(*at), z_dir=Vector(*c.axis)) * Cylinder(
            c.r, e1 - e0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return (shape + cyl(base)) - cyl(base + np.asarray(shift, float))


def test_steerer_bore_must_be_concentric_with_the_clamp():
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b = stem.build(p)[0]
    bore = max(find_cyl(b, (0, 0, 1), near=(0, 0, 0), inner=True), key=lambda c: c.r)
    rep = check_sheet(stem_drawing.body_sheet(p, _refill_and_move(b, bore, (0.4, 0, 0))))
    bad = {c["check"] for c in failed(rep)}
    assert "dim.steerer_bore_axis_offset" in bad and "dim.steerer_bore_d" not in bad


def test_both_pinch_holes_are_located():
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b = stem.build(p)[0]
    hole = max(find_cyl(b, (0, 1, 0), r=p.bolt_clear_d / 2, inner=True), key=lambda c: c.loc[2])
    rep = check_sheet(stem_drawing.body_sheet(p, _refill_and_move(b, hole, (-1.0, 0, 0))))
    bad = {c["check"] for c in failed(rep)}
    assert {"dim.pinch_hole_x#2", "dim.pinch_cbore_coaxial#2"} <= bad
    assert "dim.pinch_pitch" not in bad and "dim.pinch_bolt_x" not in bad


def test_engagement_rule_uses_the_called_out_thread_depth():
    p = CASES[0]
    b, f = stem.build(p)
    dims = {d["name"]: d for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    eng = stem.checks(p, b, f)["thread_engagement_mm"]
    assert eng["faceplate"] == dims["fp_thread_depth"]["measured"] == p.tap_depth - stem.TAP_RUNOUT
    assert eng["pinch"] == dims["pinch_thread_depth"]["measured"]
    # an 8 mm drill gives 6 mm of full thread: below 1.5 d for M5, whatever the drill depth
    assert stem.usable_thread(8.0, through=False) == 6.0 < 7.5


def test_window_must_be_centred(monkeypatch):
    from build123d import Pos
    orig = stem._window_profile
    monkeypatch.setattr(stem, "_window_profile", lambda p: Pos(1.0, 0) * orig(p))   # 1 mm across (Y)
    p = CASES[0]
    rep = check_sheet(stem_drawing.faceplate_sheet(p, stem.build(p)[1]))
    bad = {c["check"] for c in failed(rep)}
    assert "dim.window_center_y" in bad
    assert not bad & {"dim.window_w", "dim.window_h", "dim.window_r"}


def test_faceplate_bore_and_lightening_bore_are_on_their_axes():
    from agent_cad.drawing.measure import find_cyl
    import numpy as np
    p = CASES[0]
    b, f = stem.build(p)
    d, n, S, B = stem.frame(p)
    dv, nv = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z])
    good_f = {d_["name"]: d_ for d_ in stem_drawing.faceplate_sheet(p, f).meta["dims"]}
    good_b = {d_["name"]: d_ for d_ in stem_drawing.body_sheet(p, b).meta["dims"]}
    assert good_f["bar_bore_axis_offset"]["measured"] < 1e-3 and good_b["light_axis_offset"]["measured"] < 1e-3
    bore = max((c for c in find_cyl(f, (0, 1, 0), inner=True) if c.r > 10), key=lambda c: c.r)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.faceplate_sheet(p, _refill_and_move(f, bore, nv * 0.3))))}
    assert "dim.bar_bore_axis_offset" in bad and "dim.bar_bore_d" not in bad
    light = max((c for c in find_cyl(b, dv, inner=True) if c.r > p.tap_drill_d / 2 + 0.1), key=lambda c: c.r)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, _refill_and_move(b, light, nv * 0.5))))}
    assert "dim.light_axis_offset" in bad and "dim.light_d" not in bad


def test_narrow_lightening_bore_and_small_bar_are_recognised():
    import dataclasses
    p = dataclasses.replace(CASES[0], light_d=4.0)
    dims = {d["name"]: d for d in stem_drawing.body_sheet(p, stem.build(p)[0]).meta["dims"]}
    assert abs(dims["light_d"]["measured"] - 4.0) < 1e-3
    p = dataclasses.replace(CASES[0], bar_d=20.0)
    b, f = stem.build(p)
    for sh in (stem_drawing.body_sheet(p, b), stem_drawing.faceplate_sheet(p, f)):
        dims = {d["name"]: d for d in sh.meta["dims"]}
        assert abs(dims["bar_bore_d"]["measured"] - 20.0) < 1e-3, sh.meta["part"]


def test_slot_must_split_the_whole_clamp(monkeypatch):
    p = CASES[0]
    box = stem.Box

    def short_slot(x, y, z, **kw):       # the slot cutter (3 x stack high) only 20 mm high
        return box(x, y, p.stack_h - 20 if abs(y - p.slot_w) < 1e-9 and abs(z - 3 * p.stack_h) < 1e-9 else z, **kw)
    monkeypatch.setattr(stem, "Box", short_slot)
    b = stem.build(p)[0]
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b)))}
    assert {"dim.slot_lo_uncovered", "dim.slot_hi_uncovered"} <= bad and "dim.slot_w" not in bad


def test_faceplate_bolt_holes_must_be_through():
    from build123d import Cylinder, Plane, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    import numpy as np
    p = CASES[0]
    f = stem.build(p)[1]
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    c = find_cyl(f, dv, r=p.bolt_clear_d / 2, inner=True)[0]
    e = np.array(c.axial_extent()) * np.sign(c.axis @ dv)
    start = c.loc + dv * (e.min() - c.loc @ dv)
    plug = Plane(origin=Vector(*start), z_dir=Vector(*dv)) * Cylinder(c.r + 0.01, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + plug)))}
    assert any(x.startswith("dim.bolt_hole_from_mating_face") for x in bad)


def test_thin_lines_follow_the_jis_line_group():
    from agent_cad.drawing.sheet import STYLES
    assert STYLES["outline"][0] == 0.5
    assert all(STYLES[k][0] == 0.25 for k in ("thin", "center", "hatch", "hidden"))


def test_holes_run_where_the_drawing_says(monkeypatch):
    """Through bores span their clamp, pinch holes reach the slot, faceplate taps open
    on the mating face: each fails when the cutter is shortened or moved."""
    import numpy as np
    from build123d import Cylinder, Plane, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b = stem.build(p)[0]
    good = {d["name"]: d for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    for k in ("steerer_bore_bottom", "steerer_bore_top", "bar_bore_y_lo", "pinch_hole_from_slot",
              "pinch_hole_to_cbore#2", "fp_tap_on_mating_face#4"):
        assert abs(good[k]["measured"]) < 1e-3, k
    # steerer bore blind for its lowest 3 mm
    plug = Cylinder(p.steerer_d / 2 + 0.01, 3.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b + plug)))}
    assert "dim.steerer_bore_bottom" in bad and "dim.steerer_bore_d" not in bad
    # one pinch clearance hole stops 2 mm short of the slot
    h = min(find_cyl(b, (0, 1, 0), r=p.bolt_clear_d / 2, inner=True), key=lambda c: c.loc[2])
    plug = Plane(origin=Vector(h.loc[0], p.slot_w / 2, h.loc[2]), z_dir=(0, 1, 0)) * Cylinder(
        h.r + 0.01, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b + plug)))}
    assert any(c.startswith("dim.pinch_hole_from_slot") for c in bad) and "dim.slot_w" not in bad
    # faceplate tap drills start 1 mm inside the body
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    taps = find_cyl(b, dv, r=p.tap_drill_d / 2, inner=True)
    for c in taps:
        top = c.loc + dv * (max(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
        b = b + Plane(origin=Vector(*top), z_dir=Vector(*-dv)) * Cylinder(
            c.r + 0.01, 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b)))}
    assert "dim.fp_tap_on_mating_face" in bad


def test_lightening_bore_runs_the_whole_body_and_is_not_a_tap(monkeypatch):
    """A lightening bore that stops short of the bar bore is reported; one of the tap
    drill size is not counted as a faceplate tap; a bar bore is measured against its
    clamp even where the pinch ears are wider."""
    import dataclasses
    p = CASES[0]
    b = stem.build(p)[0]
    good = {d["name"]: d for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    assert good["light_to_steerer_bore"]["measured"] < 1e-3 and good["light_to_bar_bore"]["measured"] < 1e-3
    for q in (dataclasses.replace(p, light_d=p.tap_drill_d), dataclasses.replace(p, ear_half_w=p.bar_clamp_w / 2 + 2)):
        bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(q, stem.build(q)[0])))}
        assert not bad, (q, bad)
    ext = stem.extrude

    def short_bore(obj, amount=None, *a, **kw):      # the bore cutter (p.length long) 20 mm short
        return ext(obj, amount - 20 if amount == p.length else amount, *a, **kw)
    monkeypatch.setattr(stem, "extrude", short_bore)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, stem.build(p)[0])))}
    assert "dim.light_to_bar_bore" in bad and "dim.light_d" not in bad


def test_features_are_checked_over_their_whole_length():
    """A web across a bore or the slot, a pinch tap that no longer opens on the ear
    face, a counterbore under a skin: each is reported, however thin."""
    import numpy as np
    from build123d import Box, Cylinder, Plane, Pos, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b, f = stem.build(p)
    good = {d["name"]: d for d in stem_drawing.body_sheet(p, b).meta["dims"]}
    assert abs(good["pinch_tap_to_slot#2"]["measured"]) < 1e-3 and good["pinch_tap_open#2"]["measured"] == 0
    zm = p.stack_h / 2 + 3
    web = Pos(0, 0, zm) * Cylinder(p.steerer_d / 2 + 0.01, 0.5)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b + web)))}
    assert {"dim.steerer_bore_bottom", "dim.steerer_bore_top"} & bad
    xm = -(p.steerer_d / 2 + p.steerer_d / 2 + p.steerer_wall) / 2
    bridge = Pos(xm, 0, zm) * Box(2, p.slot_w + 0.02, 0.3)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b + bridge)))}
    assert {"dim.slot_lo_uncovered", "dim.slot_hi_uncovered"} <= bad
    tap = min(find_cyl(b, (0, 1, 0), r=p.tap_drill_d / 2, inner=True), key=lambda c: c.loc[2])
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, _refill_and_move(b, tap, (0, 1.0, 0)))))}
    assert any(x.startswith("dim.pinch_tap_open") for x in bad)
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    top = c.loc + dv * (max(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
    skin = Plane(origin=Vector(*top), z_dir=Vector(*-dv)) * Cylinder(c.r + 0.01, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + skin)))}
    assert any(x.startswith("dim.cbore_open") for x in bad)


def test_pdf_hatch_uses_the_declared_width(tmp_path):
    import matplotlib
    from agent_cad.drawing.sheet import STYLES
    from agent_cad.drawing.render import PT_PER_MM
    p = CASES[0]
    to_pdf(stem_drawing.body_sheet(p, stem.build(p)[0]), tmp_path / "b.pdf")
    assert abs(matplotlib.rcParams["hatch.linewidth"] - STYLES["hatch"][0] * PT_PER_MM) < 1e-9


def test_bar_bore_of_fastener_size_is_not_a_pinch_feature_and_thin_skins_show():
    import dataclasses
    import numpy as np
    from build123d import Cylinder, Plane, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    for bar in (p.cbore_d, p.bolt_clear_d, p.tap_drill_d):
        q = dataclasses.replace(p, bar_d=bar)
        feats = {f["name"]: f for f in stem_drawing.body_sheet(q, stem.build(q)[0]).meta["features"]}
        assert all(feats[k]["found"] == 2 for k in ("pinch_holes", "pinch_cbores", "pinch_taps")), (bar, feats)
    f = stem.build(p)[1]
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    top = c.loc + dv * (max(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
    skin = Plane(origin=Vector(*top), z_dir=Vector(*-dv)) * Cylinder(c.r + 0.01, 0.02, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + skin)))}
    assert any(x.startswith("dim.cbore_open") for x in bad)


def test_open_sections_slot_bridges_and_wide_faceplate_gap():
    """A ring skin over a counterbore (open at the centre), a bridge across the slot away
    from the probe line, both reported; a wide faceplate gap still finds the mating face."""
    import dataclasses
    import numpy as np
    from build123d import Box, Cylinder, Plane, Pos, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b, f = stem.build(p)
    pin = min(find_cyl(b, (0, 1, 0), r=p.bolt_clear_d / 2, inner=True), key=lambda c: c.loc[2])
    bridge = Pos(pin.loc[0] - p.bolt_clear_d, 0, pin.loc[2]) * Box(1.0, p.slot_w + 0.02, 1.0)
    bad = {c["check"] for c in failed(check_sheet(stem_drawing.body_sheet(p, b + bridge)))}
    assert "dim.slot_bridged_mm3" in bad
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    top = c.loc + dv * (max(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
    pl = Plane(origin=Vector(*top), z_dir=Vector(*-dv))
    ring = pl * (Cylinder(c.r + 0.01, 0.3, align=(Align.CENTER, Align.CENTER, Align.MIN))
                 - Cylinder(p.bolt_clear_d / 2 + 1.0, 0.3, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + ring)))}
    assert any(x.startswith("dim.cbore_open") for x in bad)
    q = dataclasses.replace(p, fp_gap=12.0)          # mating face 6 mm behind the bar axis
    dims = {x["name"]: x for x in stem_drawing.body_sheet(q, stem.build(q)[0]).meta["dims"]}
    assert "fp_tap_on_mating_face#4" in dims
    assert all(abs(dims[k]["measured"]) < 1e-3 for k in dims if k.startswith("fp_tap_on_mating_face")), dims


def test_window_and_counterbore_openings_are_clear_to_their_walls():
    """An island in one quadrant of the window, a ring in the outermost 1 % of a
    counterbore's radius: both reported."""
    import numpy as np
    from build123d import Box, Cylinder, Plane, Pos, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    f = stem.build(p)[1]
    good = {d["name"]: d for d in stem_drawing.faceplate_sheet(p, f).meta["dims"]}
    assert good["window_material_mm3"]["measured"] < 1e-3
    d, n, S, B = stem.frame(p)
    dv, nv, Bm = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z]), np.array([B.X, B.Y, B.Z])
    at = Bm + np.array([0, 1, 0]) * p.fp_window_w * 0.3 + nv * p.fp_window_h * 0.3
    island = Plane(origin=Vector(*(at - dv * 50)), z_dir=Vector(*dv)) * Cylinder(0.4, 100, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + (island & _hull_box(f)))))}
    assert "dim.window_material_mm3" in bad
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    top = c.loc + dv * (max(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
    pl = Plane(origin=Vector(*top), z_dir=Vector(*-dv))
    ring = pl * (Cylinder(c.r + 0.01, 0.3, align=(Align.CENTER, Align.CENTER, Align.MIN))
                 - Cylinder(c.r * 0.99, 0.3, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + ring)))}
    assert any(x.startswith("dim.cbore_open") for x in bad)


def _hull_box(shape):
    """The shape's bounding box as a solid, to trim test add-ons to the part's extent."""
    from build123d import Box, Pos
    bb = shape.bounding_box()
    return Pos(*bb.center()) * Box(bb.size.X, bb.size.Y, bb.size.Z)


def test_holes_are_clear_over_their_full_section_and_taps_fit_m5():
    """A ledge at a counterbore bottom, a radial bridge in a clearance hole: reported. A tap
    drill outside the M5 minor-diameter band fails the design check and the drawing."""
    import dataclasses
    import numpy as np
    from build123d import Box, Cylinder, Plane, Pos, Align, Vector
    from agent_cad.drawing.measure import find_cyl
    p = CASES[0]
    b, f = stem.build(p)
    d, n, S, B = stem.frame(p)
    dv = np.array([d.X, d.Y, d.Z])
    c = find_cyl(f, dv, r=p.cbore_d / 2, inner=True)[0]
    bot = c.loc + dv * (min(np.array(c.axial_extent()) * np.sign(c.axis @ dv)) - c.loc @ dv)
    ledge = Plane(origin=Vector(*bot), z_dir=Vector(*dv), x_dir=Vector(0, 1, 0)) * Pos(c.r - 0.5, 0, 0.25) * Box(1.0, 1.0, 0.5)
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + ledge)))}
    assert any(x.startswith("dim.cbore_open") for x in bad), bad
    h = find_cyl(f, dv, r=p.bolt_clear_d / 2, inner=True)[0]
    mid = h.loc + dv * (np.mean(h.axial_extent()) * np.sign(h.axis @ dv) - h.loc @ dv)
    rib = Plane(origin=Vector(*mid), z_dir=Vector(*dv), x_dir=Vector(0, 1, 0)) * Pos(h.r - 0.4, 0, 0) * Box(0.8, 0.6, 0.6)
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.faceplate_sheet(p, f + rib)))}
    assert any(x.startswith("dim.bolt_hole_clear") for x in bad), bad
    q = dataclasses.replace(p, tap_drill_d=3.3)
    bq, fq = stem.build(q)
    assert not stem.checks(q, bq, fq)["tap_drill_fits_M5"]
    bad = {x["check"] for x in failed(check_sheet(stem_drawing.body_sheet(q, bq)))}
    assert any(x.startswith("dim.fp_tap_drill_vs_M5") for x in bad) and any(x.startswith("dim.pinch_tap_drill_vs_M5") for x in bad)
