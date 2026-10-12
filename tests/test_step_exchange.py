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


def test_part21_lexing():
    from agent_cad.verify.step_check import _scrub, _entities
    data = """
  #1 = PRODUCT('owner''s part #77','',(#2));   /* owner's note #999 */
#2=PRODUCT_CONTEXT('',#3,
   'mechanical');
#3 = ( LENGTH_UNIT() NAMED_UNIT(*)
       SI_UNIT(.MILLI.,
               .METRE.) );
#4 = BREP_WITH_VOIDS('',#5,());
#5 = CLOSED_SHELL('',());
"""
    ents = _entities(_scrub(data))
    assert set(ents) == {"1", "2", "3", "4", "5"}                    # indented / multi-line definitions
    assert ents["1"][1].count("#") == 1                              # only #2: string and comment ignored
    assert "SI_UNIT(.MILLI.,.METRE.)" in ents["3"][1] and ents["4"][0] == "BREP_WITH_VOIDS"


def test_unit_text_in_a_string_does_not_pass(parts, asm_file, tmp_path):
    txt = asm_file.read_text(encoding="latin-1").replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT(.CENTI.,.METRE.)")
    txt = txt.replace("PRODUCT('STEM_BODY'", "PRODUCT('STEM_BODY SI_UNIT(.MILLI.,.METRE.)'", 1)
    p = tmp_path / "fake.step"; p.write_text(txt, encoding="latin-1")
    assert "header.unit_mm" in failed(check_step(str(p), parts, "STEM_ASM"))


def test_same_shape_with_different_metadata(tmp_path):
    from build123d import Box
    b = Box(10, 10, 10)
    parts = [PartSpec("A", b, (1, 0, 0)), PartSpec("B", b, (0, 1, 0))]
    path = tmp_path / "same.step"; write_step(parts, str(path), "ASM")
    _, rp = read_step(str(path))
    assert sorted((p.name, p.color) for p in rp) == [("A", (1.0, 0.0, 0.0)), ("B", (0.0, 1.0, 0.0))]
    assert check_step(str(path), parts, "ASM")["pass"]


def test_repeated_instances_share_one_definition(tmp_path):
    from build123d import Box
    b = Box(10, 10, 10)
    parts = [PartSpec("BOLT", b), PartSpec("BOLT", b)]
    path = tmp_path / "rep.step"; write_step(parts, str(path), "ASM")
    rep = check_step(str(path), parts, "ASM")
    assert rep["pass"], failed(rep)


def test_multiple_roots_are_rejected(tmp_path):
    from build123d import Box, Pos
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    w = STEPControl_Writer()
    a, b = Box(10, 10, 10), Pos(30, 0, 0) * Box(10, 10, 10)
    w.Transfer(a.wrapped, STEPControl_AsIs); w.Transfer(b.wrapped, STEPControl_AsIs)
    path = tmp_path / "roots.step"; w.Write(str(path))
    _, rp = read_step(str(path))
    assert {p.n_roots for p in rp} == {2}
    assert "struct.single_root" in failed(check_step(str(path), [PartSpec("", a), PartSpec("", b)]))


def test_rotation_about_the_centroid_is_detected(tmp_path):
    from build123d import Box
    a, turned = Box(10, 20, 30), Box(20, 10, 30)   # a turned 90 deg about its centroid: same V, A, centroid
    path = tmp_path / "rot.step"; write_step([PartSpec("P", turned)], str(path))
    rep = check_step(str(path), [PartSpec("P", a)])
    assert failed(rep) == {"P.geom.orientation"}, failed(rep)


def test_interleaved_definitions_keep_component_order(tmp_path):
    from build123d import Box
    a, b = Box(10, 10, 10), Box(20, 10, 10)
    parts = [PartSpec("P", a), PartSpec("P", b), PartSpec("P", a)]
    path = tmp_path / "aba.step"; write_step(parts, str(path), "ASM")
    _, rp = read_step(str(path))
    assert [round(p.shape.volume) for p in rp] == [1000, 2000, 1000]
    rep = check_step(str(path), parts, "ASM")
    assert rep["pass"], failed(rep)


def test_unreadable_file_gives_a_failed_report(tmp_path):
    from build123d import Box
    path = tmp_path / "junk.step"; path.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\n", encoding="latin-1")
    rep = check_step(str(path), [PartSpec("P", Box(1, 1, 1))])
    assert not rep["pass"] and "struct.import" in failed(rep)


def test_mesh_volume_far_from_the_origin():
    from build123d import Box, Pos
    from agent_cad.verify.step_check import _mesh_volume
    far = Pos(1e7, -1e7, 1e7) * Box(10, 20, 30)
    assert abs(_mesh_volume(far) - 6000.0) < 1e-6 * 6000.0


def test_empty_shape_has_no_bounds():
    from build123d import Compound
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from agent_cad.verify.step_check import _bbox
    empty = TopoDS_Compound(); BRep_Builder().MakeCompound(empty)   # what a damaged component imports as
    assert _bbox(Compound(empty)) is None


def _tetra(pts):
    from build123d import Face, Wire, Shell, Solid
    faces = [Face(Wire.make_polygon([pts[i] for i in f], close=True)) for f in ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))]
    return Solid(Shell(faces))


def test_symmetric_part_turned_is_detected(tmp_path):
    """A regular tetrahedron: isotropic inertia and a cubic bounding box, so only the
    surface sampling sees the 90 degree turn."""
    a = _tetra([(10, 10, 10), (10, -10, -10), (-10, 10, -10), (-10, -10, 10)])
    turned = _tetra([(-10, 10, 10), (10, 10, -10), (-10, -10, -10), (10, -10, 10)])
    path = tmp_path / "tet.step"; write_step([PartSpec("T", turned)], str(path))
    assert "T.geom.orientation" in failed(check_step(str(path), [PartSpec("T", a)]))
    path2 = tmp_path / "tet_ok.step"; write_step([PartSpec("T", a)], str(path2))
    rep = check_step(str(path2), [PartSpec("T", a)])
    assert rep["pass"], failed(rep)


def test_placed_instances_share_one_definition(tmp_path):
    from build123d import Box, Pos, Rot
    b = Box(10, 20, 30)
    parts = [PartSpec("BOLT", Pos(0, 0, 0) * b), PartSpec("BOLT", Pos(50, 0, 0) * Rot(0, 0, 30) * b)]
    path = tmp_path / "placed.step"; write_step(parts, str(path), "ASM")
    rep = check_step(str(path), parts, "ASM")
    assert rep["pass"], failed(rep)
    assert next(c for c in rep["checks"] if c["check"] == "text.solid_count")["found"] == 1


def test_placed_bare_part_round_trips(tmp_path):
    from build123d import Box, Pos, Rot
    part = PartSpec("P", Pos(5, 6, 7) * Rot(0, 0, 90) * Box(10, 20, 30))
    path = tmp_path / "bare.step"; write_step([part], str(path))
    rep = check_step(str(path), [part])
    assert rep["pass"], failed(rep)


def test_header_strings_cannot_fake_sections(parts, tmp_path):
    path = tmp_path / "hdr.step"
    write_step(parts, str(path), "STEM_ASM", author="x DATA; FILE_SCHEMA(('OTHER'));", organization="it''s")
    rep = check_step(str(path), parts, "STEM_ASM")
    assert rep["pass"], failed(rep)


def test_nested_assembly_reads_leaf_parts(tmp_path):
    """TOP -> SUB (placed at x=100) -> LEAF (placed at y=10): the leaf comes back by
    name, at the composed placement."""
    from build123d import Box
    from OCP.TDataStd import TDataStd_Name
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TopLoc import TopLoc_Location
    from OCP.gp import gp_Trsf, gp_Vec
    from agent_cad.step_io import _new_doc

    def at(x, y, z):
        t = gp_Trsf(); t.SetTranslation(gp_Vec(x, y, z)); return TopLoc_Location(t)
    doc = _new_doc(); st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)
    named = lambda lab, s: TDataStd_Name.Set_s(lab, TCollection_ExtendedString(s))
    leaf = st.AddShape(Box(10, 20, 30).wrapped, False); named(leaf, "LEAF")
    sub = st.NewShape(); named(sub, "SUB"); st.AddComponent(sub, leaf, at(0, 10, 0))
    top = st.NewShape(); named(top, "TOP"); st.AddComponent(top, sub, at(100, 0, 0))
    st.UpdateAssemblies()
    w = STEPCAFControl_Writer(); w.SetNameMode(True); w.Transfer(doc, STEPControl_AsIs)
    path = tmp_path / "nested.step"; w.Write(str(path))
    asm, rp = read_step(str(path))
    assert asm == "TOP" and [p.name for p in rp] == ["LEAF"] and rp[0].path == ("TOP", "SUB")
    c = rp[0].shape.center()
    assert abs(c.X - 100) < 1e-6 and abs(c.Y - 10) < 1e-6 and abs(c.Z) < 1e-6
    # same leaf, name and placement as a flat export, but an extra level: not what write_step wrote
    from build123d import Pos
    assert "struct.flat" in failed(check_step(str(path), [PartSpec("LEAF", Pos(100, 10, 0) * Box(10, 20, 30))], "TOP"))


def test_reversed_instance_is_its_own_definition():
    from build123d import Box, Shape
    from agent_cad.step_io import definitions
    b = Box(10, 10, 10)
    parts = [PartSpec("P", b), PartSpec("P", Shape.cast(b.wrapped.Reversed()))]
    assert len(definitions(parts)) == 2


def test_reversed_solid_mesh_volume(tmp_path):
    from build123d import Box, Shape
    from agent_cad.verify.step_check import _mesh_volume
    r = Shape.cast(Box(10, 20, 30).wrapped.Reversed())
    assert abs(abs(_mesh_volume(r)) - 6000.0) < 1.0
    path = tmp_path / "rev.step"; write_step([PartSpec("R", r)], str(path))
    assert "R.mesh.volume" not in failed(check_step(str(path), [PartSpec("R", r)]))



def test_loose_vertex_beside_the_solid_is_rejected():
    from build123d import Box, Compound, Vertex
    from agent_cad.verify.step_check import _stray
    b = Box(10, 10, 10)
    assert _stray(b.wrapped) == 0
    assert _stray(Compound([b, Vertex(0, 0, 0)]).wrapped) == 1


def test_mirrored_part_keeps_its_material(tmp_path):
    from build123d import Box, Pos, mirror, Plane
    from agent_cad.step_io import write_step, PartSpec
    from agent_cad.verify.step_check import check_step
    m = mirror(Pos(5, 5, 5) * Box(10, 10, 10), Plane.XZ)       # nested compound with a mirror location
    s = PartSpec("MIRRORED", m, (0.5, 0.5, 0.5), "SKD61", 7.8)
    for parts, asm in (([s], None), ([s, PartSpec("OTHER", Box(3, 3, 3), (0.2, 0.2, 0.2), "A6061-T6", 2.7)], "ASM")):
        path = tmp_path / f"m{len(parts)}.step"
        write_step(parts, path, asm)
        rep = check_step(path, parts, asm)
        assert rep["pass"], [c for c in rep["checks"] if not c["pass"]]


def test_tangent_seam_point_is_on_surface(tmp_path):
    """A sample point on a tangent seam of the -17 deg body is called IN by the solid
    classifier although it lies on the surface; the distance fallback settles it."""
    from scripts.make_stem import CASES, part_specs
    from agent_cad.parts import stem
    from agent_cad.step_io import write_step
    from agent_cad.verify.step_check import check_step
    p = CASES["stem110_-17_7075"]
    body, _ = part_specs(p, *stem.build(p))
    path = tmp_path / "b.step"
    write_step([body], path)
    rep = check_step(path, [body], None)
    assert rep["pass"], [c for c in rep["checks"] if not c["pass"]]
