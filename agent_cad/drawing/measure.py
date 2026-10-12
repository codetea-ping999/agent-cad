"""Read feature geometry back from a B-rep so drawings dimension the model as
built, not the parameters that were meant to build it."""
from dataclasses import dataclass
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from build123d import Shape


@dataclass
class Cyl:
    r: float
    loc: np.ndarray     # a point on the axis
    axis: np.ndarray    # unit vector
    faces: list
    inner: bool         # material outside the cylinder (a hole)

    def axial_extent(self):
        """Extent of the face along the axis, as coordinates along `axis` from the model origin."""
        t = []
        for f in self.faces:
            for e in f.edges():
                for k in range(9):
                    p = e.position_at(k / 8)
                    t.append(np.array([p.X, p.Y, p.Z]) @ self.axis)
        return min(t), max(t)


def cylinders(shape: Shape):
    out = []
    for f in shape.faces():
        s = BRepAdaptor_Surface(f.wrapped)
        if s.GetType() != GeomAbs_Cylinder:
            continue
        c = s.Cylinder(); ax = c.Axis()
        loc = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        # hole if the outward face normal points toward the axis
        p = f.center(); n = f.normal_at(p)
        v = np.array([p.X, p.Y, p.Z]) - loc; v -= (v @ d) * d
        out.append(Cyl(c.Radius(), loc, d, [f], float(v @ np.array([n.X, n.Y, n.Z])) < 0))
    return out


def find_cyl(shape, axis, r=None, near=None, inner=None, rtol=0.05):
    """Cylinders whose axis is parallel to `axis`, optionally of radius ~r,
    with axis passing within 1 mm of `near`. Duplicates (split faces) merged."""
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    res = []
    for c in cylinders(shape):
        if abs(abs(c.axis @ axis) - 1) > 1e-6:
            continue
        if r is not None and abs(c.r - r) > rtol:
            continue
        if inner is not None and c.inner != inner:
            continue
        if near is not None:
            v = np.asarray(near, float) - c.loc
            if np.linalg.norm(v - (v @ axis) * axis) > 1.0:
                continue
        c.axis = axis if c.axis @ axis > 0 else -c.axis
        res.append(c)
    uniq = []
    for c in res:  # merge faces of the same cylinder
        same = [u for u in uniq if abs(c.r - u.r) < 1e-6 and np.linalg.norm(np.cross(c.loc - u.loc, axis)) < 1e-6]
        if same:
            same[0].faces += c.faces
        else:
            uniq.append(c)
    return uniq


def planes(shape, normal, near_center=None, radius=None):
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    out = []
    for f in shape.faces():
        if BRepAdaptor_Surface(f.wrapped).GetType() != GeomAbs_Plane:
            continue
        n = f.normal_at(); n = np.array([n.X, n.Y, n.Z])
        if abs(abs(n @ normal) - 1) > 1e-6:
            continue
        c = f.center(); c = np.array([c.X, c.Y, c.Z])
        if near_center is not None and np.linalg.norm(c - near_center) > radius:
            continue
        out.append((float(c @ normal), f, n))
    return out
