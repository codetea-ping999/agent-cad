"""Wall thickness measured on the B-rep (not the design-rule arithmetic): from
sample points on every face, a ray along the inward normal to the next wall."""
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCP.gp import gp_Lin, gp_Pnt, gp_Dir, gp_Pnt2d
from OCP.Precision import Precision


def _outward(face, u, v):
    s = BRepAdaptor_Surface(face)
    pr = BRepLProp_SLProps(s, u, v, 1, 1e-7)
    if not pr.IsNormalDefined():
        return None
    n = pr.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        n.Reverse()
    return np.array([n.X(), n.Y(), n.Z()])


def wall_samples(shape, per_face=6, min_gap=1e-3, max_angle_deg=30.0):
    """[(thickness_mm, point, face_index)] for a per_face x per_face grid on each face.
    A ray counts only if it leaves through a face roughly facing it (within
    max_angle_deg): a ray grazing along an adjacent face at an edge is not a wall."""
    cos_max = np.cos(np.radians(max_angle_deg))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shape.wrapped, 1e-6)
    out = []
    for fi, f in enumerate(shape.faces()):
        s = BRepAdaptor_Surface(f.wrapped)
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        props = BRepLProp_SLProps(s, 1, 1e-7)
        cls = BRepClass_FaceClassifier()
        rev = f.wrapped.Orientation() == TopAbs_REVERSED
        for a in (np.arange(per_face) + 0.5) / per_face:
            for b in (np.arange(per_face) + 0.5) / per_face:
                u, v = u0 + a * (u1 - u0), v0 + b * (v1 - v0)
                cls.Perform(f.wrapped, gp_Pnt2d(u, v), 1e-7)
                if cls.State() != TopAbs_IN:          # trimmed-away part of the surface
                    continue
                props.SetParameters(u, v)
                if not props.IsNormalDefined():
                    continue
                n = props.Normal()
                if rev:
                    n.Reverse()
                p = props.Value()
                inward = gp_Dir(-n.X(), -n.Y(), -n.Z())
                start = gp_Pnt(p.X() + inward.X() * min_gap, p.Y() + inward.Y() * min_gap, p.Z() + inward.Z() * min_gap)
                inter.Perform(gp_Lin(start, inward), 0.0, Precision.Infinite_s())
                if not inter.IsDone() or inter.NbPnt() == 0:
                    continue
                k = min(range(1, inter.NbPnt() + 1), key=inter.WParameter)
                ne = _outward(inter.Face(k), inter.UParameter(k), inter.VParameter(k))
                if ne is None or ne @ np.array([inward.X(), inward.Y(), inward.Z()]) < cos_max:
                    continue
                out.append((inter.WParameter(k) + min_gap, (p.X(), p.Y(), p.Z()), fi))
    return out


def min_wall(shape, per_face=6, ignore_below=0.05):
    """Thinnest measured wall (ignoring grazing hits at edges below `ignore_below`)."""
    s = [w for w in wall_samples(shape, per_face) if w[0] > ignore_below]
    if not s:
        return None
    t, pt, fi = min(s, key=lambda w: w[0])
    return {"min_mm": round(t, 3), "at_mm": [round(c, 2) for c in pt], "face": fi, "samples": len(s)}
