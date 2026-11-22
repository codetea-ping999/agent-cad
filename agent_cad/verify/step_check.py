"""Self-contained STEP exchange check (stands in for a CATIA import test).

Checks run on a written STEP file, against the in-memory reference parts:

1. text    – independent of OpenCascade: header (schema AP242, mm units),
             every #ref resolves, entity counts per product.
2. struct  – product tree, part names, colours, material and density
             survive the round trip.
3. topo    – each part reads back as one valid, closed solid (no free edges).
4. geom    – volume / area / centroid match the reference and the
             geometric validation properties embedded in the file.
5. mesh    – volume of an independent triangulation (divergence theorem)
             agrees with the B-rep volume, i.e. no face is missing.
6. tol     – CATIA-oriented quality: max vertex/edge tolerance and minimum
             edge length against CATIA's 0.001 mm modelling resolution.
7. rewrite – the re-read model writes and reads again with identical mass
             properties (stable round trip).
"""
import re, math, json, tempfile, os
from collections import Counter
import numpy as np
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_ShapeTolerance
from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_ON
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from build123d import Shape
from ..step_io import read_step, write_step, PartSpec, mass_props, DEFAULT_ASSEMBLY, definitions

# Thresholds. CATIA V5 works at 0.001 mm resolution; geometry tolerances well
# below its "sharpness"/merging tolerance avoid healing on import.
REL_TOL = 1e-5            # mass-property agreement (relative)
CENTROID_TOL_MM = 1e-3
INERTIA_REL_TOL = 1e-5     # inertia tensor about the centroid, relative to its largest entry
BBOX_TOL_MM = 1e-3
MESH_REL_TOL = 5e-3       # triangulation volume vs exact volume
MAX_GEOM_TOL_MM = 1e-2    # largest vertex/edge tolerance allowed
MIN_EDGE_MM = 1e-2        # shortest edge allowed (>> CATIA 0.001 mm resolution)
DENSITY_REL_TOL = 1e-6

SCHEMA_AP242 = "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"


def _check(name, ok, **detail):
    return {"check": name, "pass": bool(ok), **detail}


# ---------------------------------------------------------------- 1. text
SOLID_TYPES = ("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")


def _scrub(txt):
    """Part 21 lexing: drop /* comments */ and blank out 'strings' ('' escapes) in one
    pass, so neither can fake or hide #references or entity names."""
    out, i, n = [], 0, len(txt)
    while i < n:
        c = txt[i]
        if c == "'":
            j = i + 1
            while j < n:
                if txt[j] == "'":
                    if j + 1 < n and txt[j + 1] == "'":
                        j += 2; continue
                    break
                j += 1
            out.append("''"); i = j + 1
        elif txt.startswith("/*", i):
            j = txt.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c); i += 1
    return "".join(out)


def _mask(txt):
    """Same lexing as _scrub, but keeps every offset: comments and string contents
    become spaces, so positions found in the result index the original text."""
    out, i, n = list(txt), 0, len(txt)
    while i < n:
        if txt[i] == "'":
            j = i + 1
            while j < n:
                if txt[j] == "'":
                    if j + 1 < n and txt[j + 1] == "'":
                        j += 2; continue
                    break
                j += 1
            out[i + 1:j] = " " * (min(j, n) - i - 1); i = j + 1
        elif txt.startswith("/*", i):
            j = txt.find("*/", i + 2); j = n if j < 0 else j + 2
            out[i:j] = " " * (j - i); i = j
        else:
            i += 1
    return "".join(out)


def _string_at(txt, i):
    """The Part 21 string whose opening quote is at txt[i]."""
    j, buf = i + 1, []
    while j < len(txt):
        if txt[j] == "'":
            if txt[j + 1:j + 2] == "'":
                buf.append("'"); j += 2; continue
            break
        buf.append(txt[j]); j += 1
    return "".join(buf)


def _entities(data):
    """{id: (type, body)} of the DATA section; whitespace and line breaks are not significant."""
    ents = {}
    for stmt in data.split(";"):
        m = re.match(r"\s*#(\d+)\s*=\s*(.*)\Z", stmt, re.S)
        if m:
            body = re.sub(r"\s+", "", m.group(2))
            t = re.match(r"\(?([A-Z_0-9]+)\(", body)
            ents[m.group(1)] = (t.group(1) if t else "", body)
    return ents


def text_checks(path):
    raw = open(path, encoding="latin-1").read()
    masked = _mask(raw)                   # find section markers and keywords outside strings/comments only
    d = re.search(r"(?<![A-Z_0-9])DATA\s*;", masked)
    head, data = (raw[:d.start()], raw[d.end():]) if d else (raw, "")
    out = []
    m = re.search(r"(?<![A-Z_0-9])FILE_SCHEMA\s*\(\s*\(\s*'", masked[:len(head)])
    schema = _string_at(raw, m.end() - 1) if m else None
    out.append(_check("header.schema_ap242", schema is not None and SCHEMA_AP242 in schema, schema=schema))
    ents = _entities(_scrub(data))
    # the geometry's length unit (assigned by GLOBAL_UNIT_ASSIGNED_CONTEXT) must be the millimetre;
    # other length units may legitimately exist, e.g. the cm inside a g/cm3 density unit
    ctx_units = {r for _, b in ents.values() if "GLOBAL_UNIT_ASSIGNED_CONTEXT(" in b
                 for r in re.findall(r"#(\d+)", b.split("GLOBAL_UNIT_ASSIGNED_CONTEXT(", 1)[1])}
    lunits = [ents[r][1] for r in ctx_units if r in ents and "LENGTH_UNIT(" in ents[r][1]]
    out.append(_check("header.unit_mm", bool(lunits) and all("SI_UNIT(.MILLI.,.METRE.)" in b for b in lunits),
                      length_units=len(lunits)))
    used = {r for _, b in ents.values() for r in re.findall(r"#(\d+)", b)}
    missing = sorted(used - set(ents), key=int)[:5]
    out.append(_check("text.refs_resolve", not missing, entities=len(ents), missing=missing))
    types = Counter(t for t, _ in ents.values())
    nsol = sum(types[t] for t in SOLID_TYPES)
    out.append(_check("text.has_solids", nsol > 0, solids=nsol, advanced_face=types["ADVANCED_FACE"],
                      products=types["PRODUCT"]))
    types["_solids"] = nsol
    return out, types


# ---------------------------------------------------------------- helpers
def _mass(shape):
    v, a, c = mass_props(shape.wrapped)
    return v, a, (c.X(), c.Y(), c.Z())


def _inertia(shape, eps=1e-12):
    """Inertia tensor about the centroid, in model axes (unit density).
    Unlike volume/area/centroid it changes when the part is rotated. eps 1e-12: at 1e-9 the
    adaptive integration of a rotated torus groove with B-spline relief cuts differs by
    6e-4 between the model and its read-back (1e-12: 3e-6)."""
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, g, Eps=eps)
    m = g.MatrixOfInertia()
    return np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])


def _bbox(shape):
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, b, False, False)
    if b.IsVoid():                        # empty import: no bounds to compare
        return None
    lo, hi = b.CornerMin(), b.CornerMax()
    return np.array([lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()])


def _off_surface(a, b, n=100, tol=1e-3):
    """How many of ~n points sampled on a's faces (mesh nodes lie on the exact
    surface) are not on b's boundary: catches placements that symmetric parts
    hide from mass properties and bounds."""
    try:
        v, _ = a.tessellate(0.05, 0.2)
    except AttributeError:          # a face OCCT cannot mesh (degenerate): every sample counts as off
        return n
    if not v:
        return None
    pts = v[::max(1, len(v) // n)]
    c = BRepClass3d_SolidClassifier(b.wrapped)      # built once, queried per point
    shell = None
    off = 0
    for p in pts:
        q = gp_Pnt(p.X, p.Y, p.Z)
        c.Perform(q, tol)
        if c.State() == TopAbs_ON:
            continue
        # the classifier can call a point on a tangent seam IN or OUT: settle it by
        # the distance to b's faces, which a real misplacement does not pass
        if shell is None:
            from OCP.TopoDS import TopoDS_Compound
            from OCP.BRep import BRep_Builder
            shell = TopoDS_Compound(); bb = BRep_Builder(); bb.MakeCompound(shell)
            for f in _iter(b.wrapped, TopAbs_FACE):
                bb.Add(shell, f)
        ds = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(q).Vertex(), shell)
        off += not (ds.IsDone() and ds.Value() <= tol)
    return off


def _mesh_volume(shape, tol=0.02):
    try:
        v, t = shape.tessellate(tol, 0.1)
    except AttributeError:          # a face OCCT cannot mesh: no mesh volume, the check fails
        return None
    if not v or not t:
        return None
    V = np.array([(p.X, p.Y, p.Z) for p in v]); T = np.array(t)
    V -= V.mean(axis=0)                   # local origin: no cancellation for parts far from the global one
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _stray(shape):
    """Faces, edges or vertices outside any solid (e.g. a loose vertex in the compound)."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID
    n = 0
    for kind, avoid in ((TopAbs_FACE, TopAbs_SOLID), (TopAbs_EDGE, TopAbs_FACE), (TopAbs_VERTEX, TopAbs_EDGE)):
        ex = TopExp_Explorer(shape, kind, avoid)
        while ex.More():
            n += 1; ex.Next()
    return n


def _iter(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    ex = TopExp_Explorer(shape, kind)
    while ex.More():
        yield ex.Current(); ex.Next()


def _tol_stats(shape):
    sat = ShapeAnalysis_ShapeTolerance()
    tmax = max(sat.Tolerance(shape, 1, TopAbs_VERTEX), sat.Tolerance(shape, 1, TopAbs_EDGE))
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    lens = []
    for e in _iter(shape, TopAbs_EDGE):
        e = TopoDS.Edge(e)
        if BRep_Tool.Degenerated_s(e):
            continue
        lens.append(GCPnts_AbscissaPoint.Length_s(BRepAdaptor_Curve(e)))
    return tmax, min(lens) if lens else 0.0


def _free_edges(shape):
    fb = ShapeAnalysis_FreeBounds(shape, 1e-7, False, False)
    return sum(1 for _ in _iter(fb.GetClosedWires(), TopAbs_EDGE)) + \
        sum(1 for _ in _iter(fb.GetOpenWires(), TopAbs_EDGE))


def _close(a, b, rel=REL_TOL):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


# ---------------------------------------------------------------- main
def check_step(path, reference, assembly_name=None):
    """reference: list[PartSpec] that were written to `path`; assembly_name:
    the expected assembly product; None means what write_step does by default
    (bare part for one part, DEFAULT_ASSEMBLY for several)."""
    if assembly_name is None and len(reference) > 1:
        assembly_name = DEFAULT_ASSEMBLY
    results, types = text_checks(path)
    try:
        asm, parts = read_step(path)
    except IOError as ex:                 # unreadable file: a failed report, not an exception
        results.append(_check("struct.import", False, error=str(ex)))
        return {"file": os.path.basename(path), "pass": False, "checks": results}
    by_name = {}
    for p in parts:                       # every instance kept: duplicates are paired in file order
        by_name.setdefault(p.name, []).append(p)
    got, want = sorted(p.name for p in parts), sorted(r.name for r in reference)  # multisets: duplicates count
    results.append(_check("struct.part_names", got == want, read=got, expected=want))
    results.append(_check("struct.assembly_name", asm == assembly_name, read=asm, expected=assembly_name))
    ndef = len(definitions(reference))      # repeated instances share one solid definition
    results.append(_check("text.solid_count", types["_solids"] == ndef, found=types["_solids"], expected=ndef))
    roots = {p.n_roots for p in parts} or {0}
    results.append(_check("struct.single_root", roots == {1}, roots=sorted(roots)))
    # write_step exports flat: every part directly under the root (or a bare part)
    want_path = (assembly_name,) if assembly_name else ()
    paths = sorted({p.path for p in parts})
    results.append(_check("struct.flat", all(p.path == want_path for p in parts), paths=[list(q) for q in paths]))
    for ref in reference:
        rp = by_name[ref.name].pop(0) if by_name.get(ref.name) else None
        if rp is None:
            continue
        s, n = rp.shape, ref.name
        col_ok = rp.color is not None and all(abs(a - b) < 2e-3 for a, b in zip(rp.color, ref.color))
        results.append(_check(f"{n}.color", col_ok, read=rp.color))
        if ref.material:
            mat_ok = rp.material == ref.material and rp.density_g_cm3 is not None and \
                _close(rp.density_g_cm3, ref.density_g_cm3, DENSITY_REL_TOL)
        else:
            mat_ok = rp.material is None
        results.append(_check(f"{n}.material", mat_ok, read=rp.material, density_g_cm3=rp.density_g_cm3,
                              expected=ref.material or None, expected_density_g_cm3=ref.density_g_cm3 or None))
        results.append(_check(f"{n}.topo.valid", BRepCheck_Analyzer(s.wrapped).IsValid()))
        stray = _stray(s.wrapped)
        results.append(_check(f"{n}.topo.one_solid", len(s.solids()) == 1 and stray == 0,
                              solids=len(s.solids()), stray_subshapes=stray))
        fe = _free_edges(s.wrapped)
        results.append(_check(f"{n}.topo.closed", fe == 0, free_edges=fe))
        (v, a, c), (v0, a0, c0) = _mass(s), _mass(ref.shape)
        dc = math.dist(c, c0)
        results.append(_check(f"{n}.geom.vs_reference", _close(v, v0) and _close(a, a0) and dc < CENTROID_TOL_MM,
                              volume_mm3=round(v, 3), ref_volume_mm3=round(v0, 3),
                              area_mm2=round(a, 3), ref_area_mm2=round(a0, 3), centroid_shift_mm=round(dc, 6)))
        # orientation: volume, area and centroid survive a rotation about the centroid; these do not
        I, I0 = _inertia(s), _inertia(ref.shape)
        di = float(np.abs(I - I0).max() / max(np.abs(I0).max(), 1e-12))
        bb, bb0 = _bbox(s), _bbox(ref.shape)
        db = float(np.abs(bb - bb0).max()) if bb is not None and bb0 is not None else float("inf")
        off = (_off_surface(s, ref.shape), _off_surface(ref.shape, s))
        results.append(_check(f"{n}.geom.orientation", di <= INERTIA_REL_TOL and db <= BBOX_TOL_MM and off == (0, 0),
                              inertia_rel_err=round(di, 9), bbox_shift_mm=round(db, 6), points_off_surface=off))
        vp = rp.validation
        lacking = [k for k in ("volume", "area", "centroid") if k not in vp]
        if not lacking:
            dcv = math.dist(c, vp["centroid"])
            ok = _close(v, vp["volume"]) and _close(a, vp["area"]) and dcv < CENTROID_TOL_MM
            results.append(_check(f"{n}.geom.vs_validation_props", ok, file_volume=vp["volume"],
                                  file_area=vp["area"], centroid_shift_mm=round(dcv, 6)))
        else:
            results.append(_check(f"{n}.geom.vs_validation_props", False, missing=lacking))
        mv = _mesh_volume(s)
        if abs(v) > 0 and mv is not None:   # magnitudes: a reversed solid has negative signed volumes
            results.append(_check(f"{n}.mesh.volume", abs(abs(mv) - abs(v)) <= MESH_REL_TOL * abs(v),
                                  mesh_volume_mm3=round(mv, 1), rel_err=round((abs(mv) - abs(v)) / abs(v), 6)))
        else:
            results.append(_check(f"{n}.mesh.volume", False, volume_mm3=v, note="no volume or no triangles"))
        tmax, emin = _tol_stats(s.wrapped)
        results.append(_check(f"{n}.tol.max_tolerance", tmax <= MAX_GEOM_TOL_MM, max_tol_mm=tmax))
        results.append(_check(f"{n}.tol.min_edge", emin >= MIN_EDGE_MM, min_edge_mm=round(emin, 5)))
    # 7. rewrite the re-read model and read it again
    with tempfile.TemporaryDirectory() as td:
        p2 = os.path.join(td, "rt.step")
        try:
            write_step([PartSpec(rp.name, rp.shape, rp.color or (0.5, 0.5, 0.5), rp.material or "", rp.density_g_cm3 or 0.0)
                        for rp in parts], p2, asm)
            asm2, parts2 = read_step(p2)
        except IOError as ex:
            results.append(_check("rewrite.stable", False, error=str(ex)))
            return {"file": os.path.basename(path), "pass": False, "checks": results}
        # pair parts in file order so duplicate names are compared too
        m1 = [(p.name, p.material, _mass(p.shape)) for p in parts]
        m2 = [(p.name, p.material, _mass(p.shape)) for p in parts2]
        ok = asm2 == asm and len(m1) == len(m2) and all(
            n1 == n2 and mat1 == mat2 and _close(a[0], b[0], 1e-7) and _close(a[1], b[1], 1e-7) and math.dist(a[2], b[2]) < 1e-6
            for (n1, mat1, a), (n2, mat2, b) in zip(m1, m2))
        results.append(_check("rewrite.stable", ok))
    return {"file": os.path.basename(path), "pass": all(r["pass"] for r in results), "checks": results}


def summary(report):
    lines = [f"{report['file']}: {'PASS' if report['pass'] else 'FAIL'}"]
    for r in report["checks"]:
        if not r["pass"]:
            lines.append("  FAIL " + json.dumps(r, ensure_ascii=False))
    return "\n".join(lines)
