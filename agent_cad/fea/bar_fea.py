"""Solid FEA (gmsh P2 tetrahedra + scikit-fem) of one half of the handlebar.

The stem clamp holds the bar rigidly over its width, so the two halves do not
interact: each load case is solved on the right half (y >= 0) with the outer surface
inside the stem clamp fixed. A load is a traction on a band of the outer surface
around the load point whose resultant is the force F and the moment M about the band
centre (a hand on the hood acts above and ahead of the bar: that offset is M). One
mesh, one factorisation, one back-substitution per load case.
"""
import numpy as np
from dataclasses import dataclass
from .static import mesh_shape


@dataclass
class BandLoad:
    centre: tuple              # point on the centreline
    tangent: tuple             # centreline direction there
    radius: float              # outer radius of the tube there
    half_len: float            # band half length along the centreline
    F: tuple
    M: tuple = (0.0, 0.0, 0.0)  # about `centre`


def solve_cases(shape, material, fixed, cases, h=2.0):
    """cases: {name: [BandLoad]}. Returns {name: dict(vm_max, vm_p999, at, max_disp)} and the mesh size."""
    from skfem import Basis, FacetBasis, ElementTetP1, ElementTetP2, ElementVector, asm, LinearForm, Functional
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import sym_grad
    from scipy.sparse.linalg import factorized
    mesh = mesh_shape(shape, h)
    e = ElementVector(ElementTetP2())
    ib = Basis(mesh, e)
    lam, mu = lame_parameters(material.E_MPa, material.nu)
    K = asm(linear_elasticity(lam, mu), ib)
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    D = ib.get_dofs(bf[fixed(fc)]).all()
    I = ib.complement_dofs(D)
    lu = factorized(K[I][:, I].tocsc())
    xq = ib.mapping.F(ib.X)

    def rhs(bl):
        q, t = np.array(bl.centre, float), np.array(bl.tangent, float)
        rel = fc - q[:, None]
        ax = t @ rel
        rad = np.linalg.norm(rel - np.outer(t, ax), axis=0)
        facets = bf[(np.abs(ax) <= bl.half_len) & (np.abs(rad - bl.radius) < 0.25)]
        if len(facets) == 0:
            raise ValueError("no facets in the load band")
        sb = FacetBasis(mesh, ElementTetP1(), facets=facets)
        A = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), sb)
        cen = np.array([asm(Functional(lambda w, i=i: w.x[i] - q[i]), sb) for i in range(3)]) / A
        S = np.array([[asm(Functional(lambda w, i=i, j=j: (w.x[i] - q[i] - cen[i]) * (w.x[j] - q[j] - cen[j])), sb)
                       for j in range(3)] for i in range(3)])
        Jm = np.trace(S) * np.eye(3) - S
        F, M = np.array(bl.F, float), np.array(bl.M, float)
        W = np.linalg.lstsq(Jm, M - np.cross(cen, F), rcond=None)[0]   # t = F / A + W x r

        @LinearForm
        def L(v, w):
            r = [w.x[i] - q[i] - cen[i] for i in range(3)]
            tr = [F[i] / A + (W[(i + 1) % 3] * r[(i + 2) % 3] - W[(i + 2) % 3] * r[(i + 1) % 3]) for i in range(3)]
            return v[0] * tr[0] + v[1] * tr[1] + v[2] * tr[2]
        return asm(L, FacetBasis(mesh, e, facets=facets))

    out = {}
    for name, bands in cases.items():
        f = sum(rhs(b) for b in bands)
        u = np.zeros(K.shape[0])
        u[I] = lu(f[I])
        eps = sym_grad(ib.interpolate(u))
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        s = 2 * mu * eps + lam * np.einsum("ij,...->ij...", np.eye(3), tr)
        vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                     + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))
        vm_el = vm.max(axis=1)
        i = int(np.argmax(vm_el)); qn = int(np.argmax(vm[i]))
        umag = np.sqrt(sum(u[ib.nodal_dofs[k]] ** 2 for k in range(3)))
        out[name] = dict(vm_max_MPa=round(float(vm_el.max()), 1), vm_p999_MPa=round(float(np.percentile(vm_el, 99.9)), 1),
                         at_mm=[round(float(c), 1) for c in xq[:, i, qn]], max_disp_mm=round(float(umag.max()), 3),
                         vm_el=vm_el, centroids=mesh.p[:, mesh.t].mean(axis=1))
    return out, dict(tets=int(mesh.t.shape[1]), dofs=int(K.shape[0]), h=h)
