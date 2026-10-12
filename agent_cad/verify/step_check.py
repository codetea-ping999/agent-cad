"""Self-contained STEP exchange check (stands in for a CATIA import test).

Checks run on a written STEP file, against the in-memory reference parts:

1. text    – independent of OpenCascade: header (schema AP242, mm units),
             every #ref resolves, entity counts per product.
2. struct  – product tree, part names and colours survive the round trip.
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
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_ShapeTolerance
from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.BRepAdaptor import BRepAdaptor_Curve
from build123d import Shape
from ..step_io import read_step, write_step, PartSpec, mass_props

# Thresholds. CATIA V5 works at 0.001 mm resolution; geometry tolerances well
# below its "sharpness"/merging tolerance avoid healing on import.
REL_TOL = 1e-5            # mass-property agreement (relative)
CENTROID_TOL_MM = 1e-3
MESH_REL_TOL = 5e-3       # triangulation volume vs exact volume
MAX_GEOM_TOL_MM = 1e-2    # largest vertex/edge tolerance allowed
MIN_EDGE_MM = 1e-2        # shortest edge allowed (>> CATIA 0.001 mm resolution)

SCHEMA_AP242 = "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING"


def _check(name, ok, **detail):
    return {"check": name, "pass": bool(ok), **detail}


# ---------------------------------------------------------------- 1. text
def text_checks(path):
    txt = open(path, encoding="latin-1").read()
    head, _, data = txt.partition("DATA;")
    out = []
    out.append(_check("header.schema_ap242", SCHEMA_AP242 in head,
                      schema=re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", head).group(1)))
    out.append(_check("header.unit_mm", "SI_UNIT(.MILLI.,.METRE.)" in data.replace(" ", "")))
    defined = set(re.findall(r"^#(\d+)\s*=", data, re.M))
    used = set(re.findall(r"#(\d+)", re.sub(r"^#\d+\s*=", "", data, flags=re.M)))
    missing = sorted(used - defined, key=int)[:5]
    out.append(_check("text.refs_resolve", not missing, entities=len(defined), missing=missing))
    types = Counter(re.findall(r"^#\d+\s*=\s*([A-Z_0-9]+)\s*\(", data, re.M))
    out.append(_check("text.has_solids", types["MANIFOLD_SOLID_BREP"] > 0,
                      manifold_solid_brep=types["MANIFOLD_SOLID_BREP"], advanced_face=types["ADVANCED_FACE"],
                      products=types["PRODUCT"]))
    return out, types


# ---------------------------------------------------------------- helpers
def _mass(shape):
    v, a, c = mass_props(shape.wrapped)
    return v, a, (c.X(), c.Y(), c.Z())


def _mesh_volume(shape, tol=0.02):
    v, t = shape.tessellate(tol, 0.1)
    V = np.array([(p.X, p.Y, p.Z) for p in v]); T = np.array(t)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


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
    """reference: list[PartSpec] that were written to `path`."""
    results, types = text_checks(path)
    asm, parts = read_step(path)
    by_name = {p.name: p for p in parts}
    want = [r.name for r in reference]
    results.append(_check("struct.part_names", sorted(by_name) == sorted(want), read=sorted(by_name), expected=sorted(want)))
    if assembly_name:
        results.append(_check("struct.assembly_name", asm == assembly_name, read=asm))
    results.append(_check("text.solid_count", types["MANIFOLD_SOLID_BREP"] == len(reference),
                          found=types["MANIFOLD_SOLID_BREP"], expected=len(reference)))
    for ref in reference:
        rp = by_name.get(ref.name)
        if rp is None:
            continue
        s, n = rp.shape, ref.name
        col_ok = rp.color is not None and all(abs(a - b) < 2e-3 for a, b in zip(rp.color, ref.color))
        results.append(_check(f"{n}.color", col_ok, read=rp.color))
        results.append(_check(f"{n}.topo.valid", BRepCheck_Analyzer(s.wrapped).IsValid()))
        results.append(_check(f"{n}.topo.one_solid", len(s.solids()) == 1, solids=len(s.solids())))
        fe = _free_edges(s.wrapped)
        results.append(_check(f"{n}.topo.closed", fe == 0, free_edges=fe))
        (v, a, c), (v0, a0, c0) = _mass(s), _mass(ref.shape)
        dc = math.dist(c, c0)
        results.append(_check(f"{n}.geom.vs_reference", _close(v, v0) and _close(a, a0) and dc < CENTROID_TOL_MM,
                              volume_mm3=round(v, 3), ref_volume_mm3=round(v0, 3),
                              area_mm2=round(a, 3), ref_area_mm2=round(a0, 3), centroid_shift_mm=round(dc, 6)))
        vp = rp.validation
        if vp:
            dcv = math.dist(c, vp["centroid"]) if "centroid" in vp else 0.0
            ok = _close(v, vp.get("volume", v)) and _close(a, vp.get("area", a)) and dcv < CENTROID_TOL_MM
            results.append(_check(f"{n}.geom.vs_validation_props", ok, file_volume=vp.get("volume"),
                                  file_area=vp.get("area"), centroid_shift_mm=round(dcv, 6)))
        else:
            results.append(_check(f"{n}.geom.vs_validation_props", False, note="no validation properties in file"))
        mv = _mesh_volume(s)
        results.append(_check(f"{n}.mesh.volume", abs(mv - v) <= MESH_REL_TOL * v,
                              mesh_volume_mm3=round(mv, 1), rel_err=round((mv - v) / v, 6)))
        tmax, emin = _tol_stats(s.wrapped)
        results.append(_check(f"{n}.tol.max_tolerance", tmax <= MAX_GEOM_TOL_MM, max_tol_mm=tmax))
        results.append(_check(f"{n}.tol.min_edge", emin >= MIN_EDGE_MM, min_edge_mm=round(emin, 5)))
    # 7. rewrite the re-read model and read it again
    with tempfile.TemporaryDirectory() as td:
        p2 = os.path.join(td, "rt.step")
        write_step([PartSpec(rp.name, rp.shape, rp.color or (0.5, 0.5, 0.5)) for rp in parts], p2, asm)
        _, parts2 = read_step(p2)
        m1 = {p.name: _mass(p.shape) for p in parts}
        m2 = {p.name: _mass(p.shape) for p in parts2}
        ok = m1.keys() == m2.keys() and all(_close(m1[k][0], m2[k][0], 1e-7) and _close(m1[k][1], m2[k][1], 1e-7) for k in m1)
        results.append(_check("rewrite.stable", ok))
    return {"file": os.path.basename(path), "pass": all(r["pass"] for r in results), "checks": results}


def summary(report):
    lines = [f"{report['file']}: {'PASS' if report['pass'] else 'FAIL'}"]
    for r in report["checks"]:
        if not r["pass"]:
            lines.append("  FAIL " + json.dumps(r, ensure_ascii=False))
    return "\n".join(lines)
