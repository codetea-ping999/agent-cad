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


def test_material_and_density_round_trip(parts, asm_file, tmp_path):
    _, rp = read_step(str(asm_file))
    assert {(p.material, p.density_g_cm3) for p in rp} == {("A6061-T6", 2.7)}
    txt = asm_file.read_text(encoding="latin-1")
    txt2, n = re.subn(r"(POSITIVE_RATIO_MEASURE\(\s*)2\.7\b", r"\g<1>2.8", txt)
    assert n >= 1
    bad = tmp_path / "dens.step"; bad.write_text(txt2, encoding="latin-1")
    assert any(c.endswith(".material") for c in failed(check_step(str(bad), parts, "STEM_ASM")))


def test_unexpected_assembly_wrapper_is_detected(parts, tmp_path):
    path = tmp_path / "wrapped.step"
    write_step(parts[:1], str(path), "WRAPPER")   # a single part wrapped in an assembly
    assert "struct.assembly_name" in failed(check_step(str(path), parts[:1], None))


def test_duplicate_part_names_are_detected(parts, tmp_path):
    path = tmp_path / "dup.step"
    dup = [parts[0], PartSpec("STEM_BODY", parts[1].shape, parts[1].color, parts[1].material, 2.7)]
    write_step(dup, str(path), "STEM_ASM")
    assert "struct.part_names" in failed(check_step(str(path), parts, "STEM_ASM"))


def test_header_without_schema_fails_cleanly(parts, asm_file, tmp_path):
    txt = re.sub(r"FILE_SCHEMA\s*\(.*?\);", "FILE_SCHEMA(());", asm_file.read_text(encoding="latin-1"), flags=re.S)
    bad = tmp_path / "noschema.step"; bad.write_text(txt, encoding="latin-1")
    f = failed(check_step(str(bad), parts, "STEM_ASM"))   # must not raise
    assert "header.schema_ap242" in f


def test_hash_inside_string_is_not_a_reference(tmp_path):
    from build123d import Box
    path = tmp_path / "hash.step"
    part = [PartSpec("BRACKET #99999", Box(10, 10, 10))]
    write_step(part, str(path))
    rep = check_step(str(path), part, None)
    assert rep["pass"], failed(rep)


def test_material_without_density_round_trips(tmp_path):
    from build123d import Box
    path = tmp_path / "nodens.step"
    part = [PartSpec("BLOCK", Box(10, 10, 10), material="UNKNOWN-ALLOY")]
    write_step(part, str(path))
    _, rp = read_step(str(path))
    assert rp[0].material == "UNKNOWN-ALLOY"
    rep = check_step(str(path), part, None)
    assert rep["pass"], failed(rep)


def test_duplicate_instances_are_each_validated(tmp_path):
    from build123d import Box, Pos
    a = Box(10, 10, 10)
    path = tmp_path / "dups.step"
    # the first of two identical instances is displaced; the last one is intact
    write_step([PartSpec("BOLT", Pos(0, 7, 0) * a), PartSpec("BOLT", a)], str(path), "ASM")
    f = failed(check_step(str(path), [PartSpec("BOLT", a), PartSpec("BOLT", a)], "ASM"))
    assert "BOLT.geom.vs_reference" in f


def test_header_keeps_only_the_file_name(tmp_path):
    from build123d import Box
    path = tmp_path / "dir\\part.step"            # a Windows-style separator in the name
    write_step([PartSpec("P", Box(5, 5, 5))], str(path))
    head = path.read_text(encoding="latin-1").split("ENDSEC;")[0]
    assert "FILE_NAME('part.step'" in head.replace(" ", "")


def test_empty_tessellation_does_not_crash():
    from build123d import Compound
    from agent_cad.verify.step_check import _mesh_volume
    assert _mesh_volume(Compound([])) is None


def test_comment_with_hash_is_not_a_reference(parts, asm_file, tmp_path):
    txt = asm_file.read_text(encoding="latin-1").replace("DATA;", "DATA;\n/* inspect #999999 */", 1)
    p = tmp_path / "comment.step"; p.write_text(txt, encoding="latin-1")
    assert "text.refs_resolve" not in failed(check_step(str(p), parts, "STEM_ASM"))


def test_default_assembly_name_matches(parts, tmp_path):
    p = tmp_path / "default.step"
    write_step(parts, str(p))
    rep = check_step(str(p), parts)
    assert rep["pass"], failed(rep)
