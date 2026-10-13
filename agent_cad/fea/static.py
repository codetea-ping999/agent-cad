"""Linear static FEA (gmsh P2 tetrahedra + scikit-fem) of a solid.

Boundary conditions are picked on boundary facets by geometric predicates on the
facet centres, so the caller can say "the steerer bore is fixed, a force acts on the
bar bore" without naming B-rep faces.
"""
import os, tempfile
from dataclasses import dataclass
import numpy as np


@dataclass
class FeaResult:
    load_N: tuple
    tets: int
    vm_max_MPa: float        # largest element value: includes the stress singularity at the fixed edge
    vm_p999_MPa: float       # 99.9th percentile of element maxima: the value used for checks
    vm_at_mm: tuple
    max_disp_mm: float


def mesh_shape(shape, h):
    import gmsh
    from skfem import MeshTet
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "part.brep")
        from build123d import export_brep
        export_brep(shape, path)
        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.occ.importShapes(path); gmsh.model.occ.synchronize()
            gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h / 3)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
            gmsh.model.mesh.generate(3)
            _, xyz, _ = gmsh.model.mesh.getNodes()
            _, _, conn = gmsh.model.mesh.getElements(3)
        finally:
            gmsh.finalize()
    X = xyz.reshape(-1, 3)
    T = np.array(conn[0], dtype=np.int64).reshape(-1, 4) - 1
    used = np.unique(T); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
    return MeshTet(X[used].T, remap[T].T)


def solve(mesh, material, fixed, loads):
    """fixed(xyz) -> bool mask over facet centres (3, n); loads: [(selector, force_vec_N)].
    Each force is spread as a uniform traction over its selected facets."""
    from skfem import Basis, FacetBasis, ElementTetP1, ElementTetP2, ElementVector, asm, condense, solve as fsolve, \
        LinearForm, Functional
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from skfem.helpers import sym_grad
    e = ElementVector(ElementTetP2())
    ib = Basis(mesh, e)
    lam, mu = lame_parameters(material.E_MPa, material.nu)
    K = asm(linear_elasticity(lam, mu), ib)
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    fix = bf[fixed(fc)]
    if len(fix) == 0:
        raise ValueError("no facets selected for the fixed support")
    f = np.zeros(K.shape[0])
    for sel, F in loads:
        lf = bf[sel(fc)]
        if len(lf) == 0:
            raise ValueError("no facets selected for a load")
        area = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), FacetBasis(mesh, ElementTetP1(), facets=lf))
        t = np.asarray(F, float) / area

        @LinearForm
        def trac(v, w):
            return v[0] * t[0] + v[1] * t[1] + v[2] * t[2]
        f += asm(trac, FacetBasis(mesh, e, facets=lf))
    u = fsolve(*condense(K, f, D=ib.get_dofs(fix).all()))
    eps = sym_grad(ib.interpolate(u))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    s = 2 * mu * eps + lam * np.einsum("ij,...->ij...", np.eye(3), tr)
    vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                 + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))
    vm_el = vm.max(axis=1)
    xq = ib.mapping.F(ib.X)
    i = int(np.argmax(vm_el)); q = int(np.argmax(vm[i]))
    umag = np.sqrt(sum(u[ib.nodal_dofs[k]] ** 2 for k in range(3)))
    total = tuple(round(float(sum(np.asarray(F)[k] for _, F in loads)), 1) for k in range(3))
    return FeaResult(total, int(mesh.t.shape[1]), round(float(vm_el.max()), 1),
                     round(float(np.percentile(vm_el, 99.9)), 1),
                     tuple(round(float(c), 1) for c in xq[:, i, q]), round(float(umag.max()), 4))
