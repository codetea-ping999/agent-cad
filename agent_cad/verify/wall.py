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


def _measure(inter, f, s, props, cls, rev, u, v, min_gap, cos_max):
    """Wall thickness from face f at (u, v) along the inward normal, or None."""
    cls.Perform(f.wrapped, gp_Pnt2d(u, v), 1e-7)
    if cls.State() != TopAbs_IN:          # trimmed-away part of the surface
        return None
    props.SetParameters(u, v)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    if rev:
        n.Reverse()
    p = props.Value()
    inward = gp_Dir(-n.X(), -n.Y(), -n.Z())
    start = gp_Pnt(p.X() + inward.X() * min_gap, p.Y() + inward.Y() * min_gap, p.Z() + inward.Z() * min_gap)
    inter.Perform(gp_Lin(start, inward), 0.0, Precision.Infinite_s())
    if not inter.IsDone() or inter.NbPnt() == 0:
        return None
    k = min(range(1, inter.NbPnt() + 1), key=inter.WParameter)
    ne = _outward(inter.Face(k), inter.UParameter(k), inter.VParameter(k))
    if ne is None or ne @ np.array([inward.X(), inward.Y(), inward.Z()]) < cos_max:
        return None
    return inter.WParameter(k) + min_gap, (p.X(), p.Y(), p.Z())


def wall_samples(shape, per_face=6, min_gap=1e-3, max_angle_deg=30.0, refine=0.0, levels=3):
    """[(thickness_mm, point, face_index)] for a per_face x per_face grid on each face.
    A ray counts only if it leaves through a face roughly facing it (within
    max_angle_deg): a ray grazing along an adjacent face at an edge is not a wall.
    With refine = r, each face whose thinnest sample is within r x the thinnest of all is
    refined around it: `levels` times a 5 x 5 grid over the cell round the current best,
    the cell shrinking by 3 each time, so a minimum between the grid points is found.
    Refined points keep the grid's half-cell distance from the face boundary, and wrap
    across the seam of a periodic surface."""
    cos_max = np.cos(np.radians(max_angle_deg))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shape.wrapped, 1e-6)
    out, ctx = [], []
    for fi, f in enumerate(shape.faces()):
        s = BRepAdaptor_Surface(f.wrapped)
        u0, u1, v0, v1 = s.FirstUParameter(), s.LastUParameter(), s.FirstVParameter(), s.LastVParameter()
        c = (f, s, BRepLProp_SLProps(s, 1, 1e-7), BRepClass_FaceClassifier(), f.wrapped.Orientation() == TopAbs_REVERSED,
             (u0, u1, v0, v1))
        ctx.append(c)
        for a in (np.arange(per_face) + 0.5) / per_face:
            for b in (np.arange(per_face) + 0.5) / per_face:
                u, v = u0 + a * (u1 - u0), v0 + b * (v1 - v0)
                m = _measure(inter, *c[:5], u, v, min_gap, cos_max)
                if m:
                    out.append((m[0], m[1], fi, u, v))
    # per face its thinnest sample, if within `refine` x the thinnest of all
    face_best = {}
    for w in out:
        if w[2] not in face_best or w[0] < face_best[w[2]][0]:
            face_best[w[2]] = w
    tmin = min((w[0] for w in out), default=0.0)
    for t, pt, fi, u, v in (w for w in face_best.values() if refine and w[0] <= refine * tmin):
        f, s, props, cls, rev, (u0, u1, v0, v1) = ctx[fi]
        du, dv = (u1 - u0) / per_face, (v1 - v0) / per_face
        # stay as far from the face boundary as the grid does: half a cell
        lo_u, hi_u, lo_v, hi_v = u0 + du / 2, u1 - du / 2, v0 + dv / 2, v1 - dv / 2
        # a periodic direction (round a cylinder) has no boundary there: wrap across the seam
        if s.IsUPeriodic():
            wrap_u = lambda x, s=s: u0 + (x - u0) % s.UPeriod()
        else:
            wrap_u = lambda x: min(max(x, lo_u), hi_u)
        if s.IsVPeriodic():
            wrap_v = lambda x, s=s: v0 + (x - v0) % s.VPeriod()
        else:
            wrap_v = lambda x: min(max(x, lo_v), hi_v)
        best = (t, pt, fi, u, v)
        for _ in range(levels):
            cu, cv = best[3], best[4]
            for a in np.linspace(-1, 1, 5):
                for b in np.linspace(-1, 1, 5):
                    uu, vv = wrap_u(cu + a * du), wrap_v(cv + b * dv)
                    m = _measure(inter, f, s, props, cls, rev, uu, vv, min_gap, cos_max)
                    if m and m[0] < best[0]:
                        best = (m[0], m[1], fi, uu, vv)
            du, dv = du / 3, dv / 3
        out.append(best)
    return [w[:3] for w in out]


def min_wall(shape, per_face=6, refine=2.0):
    """Thinnest measured wall, the thinnest samples refined locally. Grazing hits at
    edges are already dropped by the exit-face angle, so every sample counts, however
    thin the wall."""
    s = wall_samples(shape, per_face, refine=refine)
    if not s:
        return None
    t, pt, fi = min(s, key=lambda w: w[0])
    return {"min_mm": round(t, 3), "at_mm": [round(c, 2) for c in pt], "face": fi, "samples": len(s)}
