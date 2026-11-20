"""Semantic PMI (STEP AP242): the dimensions and tolerances of the 2D drawing written
into the 3D model, each attached to the faces it measures, and read back for checking.

A drawing dimension becomes PMI when the faces it measures can be found on the B-rep:

- diameter / radius: the cylindrical faces of that radius (Size_Diameter / Size_Radius)
- linear: a pair of parallel planar faces that far apart (Location_LinearDistance), or
  two cylinders of one radius whose axes are that far apart (axis to axis)

Dimensions taken on sections of drafted walls (sizes at the root of a boss, say) have no
such faces; they stay on the drawing only and are listed as not linked. Tolerances are
the drawing's: the fit deviations where the drawing gives them, otherwise the general
tolerance class of the title block (JIS B 0405).
"""
import re
import numpy as np
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_Dimension
from OCP.XCAFDimTolObjects import (XCAFDimTolObjects_DimensionObject as _DimObj,
                                   XCAFDimTolObjects_DimensionType as _DT)
from OCP.TCollection import TCollection_HAsciiString
from OCP.collections import Sequence_TDF_Label
from build123d import Shape, GeomType
from .drawing.measure import cylinders
from .standards import jis

KINDS = {"diameter": _DT.XCAFDimTolObjects_DimensionType_Size_Diameter,
         "radius": _DT.XCAFDimTolObjects_DimensionType_Size_Radius,
         "linear": _DT.XCAFDimTolObjects_DimensionType_Location_LinearDistance}
_NAMES = {v: k for k, v in KINDS.items()}
TOL = 1e-3


def general_tol(value, cls):
    for a, b, t in jis.b0405_table(cls):
        if value <= b:
            return t
    return jis.b0405_table(cls)[-1][2]


def from_sheet(sheet):
    """PMI requests from a checked drawing: [dict(name, kind, value, upper, lower, text)].
    Values are the measured ones (what the drawing shows, unrounded)."""
    m = re.search(r"JIS B 0405-([a-z])", sheet.meta.get("title_block", {}).get("普通公差", ""))
    cls = m.group(1) if m else "m"
    out, seen = [], set()
    for d in sheet.meta.get("dims", []):
        base = d["name"].split("#")[0]
        t = d["text"]
        kind = "diameter" if "φ" in t else "radius" if re.match(r"(\d+×)?R", t) else "linear"
        v = d["measured"]
        if d.get("tol"):
            up, lo = (float(x) for x in d["tol"])
        else:
            g = general_tol(v, cls); up, lo = g, -g
        key = (base, kind, round(v, 3))
        if key in seen:                      # N× callouts: one PMI per distinct feature set
            continue
        seen.add(key)
        out.append(dict(name=base, kind=kind, value=float(v), upper=up, lower=lo, text=t))
    return out


def _faces_for(shape, req):
    """Faces the dimension measures: ([faces], [faces]) or None."""
    v = req["value"]
    if req["kind"] in ("diameter", "radius"):
        r = v / 2 if req["kind"] == "diameter" else v
        fs = [f for c in cylinders(shape) if abs(c.r - r) < TOL for f in c.faces]
        return (fs, None) if fs else None
    groups = {}
    for f in shape.faces():
        if f.geom_type == GeomType.PLANE:
            n = f.normal_at(); c = f.center()
            n = np.array([n.X, n.Y, n.Z])
            n = n if n[np.argmax(np.abs(n))] > 0 else -n
            groups.setdefault(tuple(np.round(n, 6)), []).append((float(n @ np.array([c.X, c.Y, c.Z])), f))
    for planes in groups.values():
        planes.sort(key=lambda t: t[0])
        offs = np.array([t[0] for t in planes])
        for i, (o, fa) in enumerate(planes):
            j = np.searchsorted(offs, o + v - TOL)
            if j < len(planes) and abs(offs[j] - o - v) < TOL:
                return [fa], [planes[j][1]]
    cs = cylinders(shape)
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            if abs(a.r - b.r) < 1e-6 and abs(abs(a.axis @ b.axis) - 1) < 1e-9:
                w = b.loc - a.loc
                if abs(np.linalg.norm(w - (w @ a.axis) * a.axis) - v) < TOL:
                    return list(a.faces), list(b.faces)
    return None


def attach(doc, label, shape_wrapped, reqs):
    """Add the PMI dimensions to the part `label` of an XCAF doc. Returns the requests
    that could not be linked to faces (they are not written)."""
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    dt = XCAFDoc_DocumentTool.DimTolTool_s(doc.Main())
    shape = Shape.cast(shape_wrapped)
    skipped = []
    for r in reqs:
        faces = _faces_for(shape, r)
        if faces is None:
            skipped.append(r); continue
        a, b = faces
        la = Sequence_TDF_Label()
        for f in a:
            la.Append(st.AddSubShape(label, f.wrapped))
        dl = dt.AddDimension()
        if b is None:
            lb = Sequence_TDF_Label()
        else:
            lb = Sequence_TDF_Label()
            for f in b:
                lb.Append(st.AddSubShape(label, f.wrapped))
        dt.SetDimension(la, lb, dl)
        o = _DimObj()
        o.SetType(KINDS[r["kind"]])
        o.SetValue(r["value"])
        o.SetUpperTolValue(r["upper"]); o.SetLowerTolValue(r["lower"])
        o.SetSemanticName(TCollection_HAsciiString(r["name"]))
        XCAFDoc_Dimension.Set_s(dl).SetObject(o)
    return skipped


def read(path):
    """[dict(kind, value, upper, lower, n_faces)] of the semantic dimensions in a STEP file."""
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from .step_io import _new_doc
    doc = _new_doc()
    XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 0.001)
    r = STEPCAFControl_Reader(); r.SetGDTMode(True); r.SetNameMode(True)
    if r.ReadFile(str(path)) != IFSelect_RetDone or not r.Transfer(doc):
        raise IOError(f"STEP read failed: {path}")
    dt = XCAFDoc_DocumentTool.DimTolTool_s(doc.Main())
    L = Sequence_TDF_Label(); dt.GetDimensionLabels(L)
    out = []
    for i in range(1, L.Length() + 1):
        o = XCAFDoc_Dimension.Set_s(L.Value(i)).GetObject()
        f1, f2 = Sequence_TDF_Label(), Sequence_TDF_Label()
        dt.GetRefShapeLabel_s(L.Value(i), f1, f2)
        out.append(dict(kind=_NAMES.get(o.GetType(), str(o.GetType())), value=o.GetValue(),
                        upper=o.GetUpperTolValue(), lower=o.GetLowerTolValue(), n_faces=f1.Length() + f2.Length()))
    return out


def check(path, reqs, skipped):
    """Every linked drawing dimension is in the file once, with its value and tolerance
    (the lower deviation is compared by magnitude: the reader returns it unsigned)."""
    got = read(path)
    want = [r for r in reqs if r not in skipped]
    missing = []
    pool = list(got)
    for r in want:
        hit = next((g for g in pool if g["kind"] == r["kind"] and abs(g["value"] - r["value"]) < 1e-6
                    and abs(g["upper"] - r["upper"]) < 1e-9 and abs(abs(g["lower"]) - abs(r["lower"])) < 1e-9
                    and g["n_faces"] > 0), None)
        if hit is None:
            missing.append(r["name"])
        else:
            pool.remove(hit)
    return {"check": "pmi", "pass": not missing and not pool, "written": len(want), "read": len(got),
            "missing": missing, "extra": len(pool), "not_linked": [r["name"] for r in skipped]}
