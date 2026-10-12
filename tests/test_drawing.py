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
    assert max(c[:, 0].max() for c in circles) < cone[:, 0].min()      # end view left of the cone
    left = cone[cone[:, 0] < cone[:, 0].mean()]
    right = cone[cone[:, 0] > cone[:, 0].mean()]
    assert np.ptp(left[:, 1]) < np.ptp(right[:, 1])                       # small end faces the circles


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
