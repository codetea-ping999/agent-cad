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
