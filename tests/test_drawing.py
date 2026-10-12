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
