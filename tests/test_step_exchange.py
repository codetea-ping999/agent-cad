"""STEP exchange check must pass on good files and catch broken ones."""
import re
import pytest
from agent_cad.parts import stem
from agent_cad.step_io import write_step, read_step, PartSpec
from agent_cad.verify.step_check import check_step


@pytest.fixture(scope="module")
def parts():
    p = stem.StemParams()
    b, f = stem.build(p)
    return [PartSpec("STEM_BODY", b, (0.6, 0.64, 0.69), p.material, 2.7),
            PartSpec("STEM_FACEPLATE", f, (0.79, 0.54, 0.24), p.material, 2.7)]


@pytest.fixture(scope="module")
def asm_file(parts, tmp_path_factory):
    path = tmp_path_factory.mktemp("s") / "asm.step"
    write_step(parts, str(path), "STEM_ASM")
    return path


def failed(rep):
    return {c["check"] for c in rep["checks"] if not c["pass"]}


def test_good_assembly_passes(parts, asm_file):
    rep = check_step(str(asm_file), parts, "STEM_ASM")
    assert rep["pass"], failed(rep)


def test_names_material_and_validation_props_are_written(asm_file):
    txt = asm_file.read_text(encoding="latin-1")
    assert "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING" in txt
    for s in ("'STEM_BODY'", "'STEM_FACEPLATE'", "'A6061-T6'", "geometric validation property", "'density'"):
        assert s in txt
    asm, rp = read_step(str(asm_file))
    assert asm == "STEM_ASM" and all(p.validation for p in rp)


def test_missing_face_is_detected(parts, asm_file, tmp_path):
    txt = asm_file.read_text(encoding="latin-1")
    m = re.search(r"CLOSED_SHELL\('',\((#\d+),", txt)
    bad = tmp_path / "bad.step"
    bad.write_text(txt[:m.start(1)] + txt[m.end(1) + 1:], encoding="latin-1")  # drop first face
    rep = check_step(str(bad), parts, "STEM_ASM")
    assert not rep["pass"]
    assert any(c.endswith(("topo.closed", "geom.vs_reference", "geom.vs_validation_props")) for c in failed(rep))


def test_tampered_validation_property_is_detected(parts, asm_file, tmp_path):
    txt = asm_file.read_text(encoding="latin-1")
    txt2, n = re.subn(r"(VOLUME_MEASURE\(\s*)([\d.E+-]+)", lambda m: m.group(1) + repr(float(m.group(2)) * 1.01), txt, count=1)
    assert n == 1
    bad = tmp_path / "bad.step"; bad.write_text(txt2, encoding="latin-1")
    assert any(c.endswith("geom.vs_validation_props") for c in failed(check_step(str(bad), parts, "STEM_ASM")))


def test_units_are_honoured(parts, asm_file, tmp_path):
    """Relabel the file as centimetres: the reader must scale it (geometry no longer
    matches) and the header check must flag the non-mm unit."""
    txt = asm_file.read_text(encoding="latin-1").replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT(.CENTI.,.METRE.)")
    bad = tmp_path / "cm.step"; bad.write_text(txt, encoding="latin-1")
    f = failed(check_step(str(bad), parts, "STEM_ASM"))
    assert "header.unit_mm" in f and "STEM_BODY.geom.vs_reference" in f
