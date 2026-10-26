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


def mesh_shape(shape, h, curvature=12):
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
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature)
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
    return solve_cases(mesh, material, fixed, {"_": loads})["_"]


def solve_cases(mesh, material, fixed, cases, solver="direct", fields=False):
    """Several load cases on one mesh and one stiffness matrix: {name: FeaResult}.
    solver: "direct" (one sparse LU for all cases) or "amg" (smoothed-aggregation AMG
    preconditioned CG with the rigid-body modes, for meshes too large for the LU).
    fields: also return per-element von Mises maxima and centroids under r.fields."""
    from skfem.models.elasticity import lame_parameters
    from skfem.helpers import sym_grad
    ib, K, D, rhs = _assemble(mesh, material, fixed)
    lam, mu = lame_parameters(material.E_MPa, material.nu)
    lin = linear_solver(ib, K, D, solver)
    xq = ib.mapping.F(ib.X)
    out = {}
    for name, loads in cases.items():
        u = lin(sum(rhs(sel, F) for sel, F in loads))
        eps = sym_grad(ib.interpolate(u))
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        s = 2 * mu * eps + lam * np.einsum("ij,...->ij...", np.eye(3), tr)
        vm = np.sqrt(0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2)
                     + 3 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2))
        vm_el = vm.max(axis=1)
        i = int(np.argmax(vm_el)); q = int(np.argmax(vm[i]))
        umag = np.sqrt(sum(u[ib.nodal_dofs[k]] ** 2 for k in range(3)))
        total = tuple(round(float(sum(np.asarray(F)[k] for _, F in loads)), 1) for k in range(3))
        r = FeaResult(total, int(mesh.t.shape[1]), round(float(vm_el.max()), 1),
                      round(float(np.percentile(vm_el, 99.9)), 1),
                      tuple(round(float(c), 1) for c in xq[:, i, q]), round(float(umag.max()), 4))
        if fields:
            r.fields = dict(vm_el=vm_el, centroids=mesh.p[:, mesh.t].mean(axis=1), u=u, basis=ib)
        out[name] = r
    return out


def _assemble(mesh, material, fixed):
    """Stiffness matrix, fixed dofs and a traction-load builder rhs(selector, F)."""
    from skfem import Basis, FacetBasis, ElementTetP1, ElementTetP2, ElementVector, asm, LinearForm, Functional
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    e = ElementVector(ElementTetP2())
    ib = Basis(mesh, e)
    lam, mu = lame_parameters(material.E_MPa, material.nu)
    K = asm(linear_elasticity(lam, mu), ib)
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    fix = bf[fixed(fc)]
    if len(fix) == 0:
        raise ValueError("no facets selected for the fixed support")

    def rhs(sel, F):
        lf = bf[sel(fc)]
        if len(lf) == 0:
            raise ValueError("no facets selected for a load")
        area = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), FacetBasis(mesh, ElementTetP1(), facets=lf))
        t = np.asarray(F, float) / area

        @LinearForm
        def trac(v, w):
            return v[0] * t[0] + v[1] * t[1] + v[2] * t[2]
        return asm(trac, FacetBasis(mesh, e, facets=lf))
    return ib, K, ib.get_dofs(fix).all(), rhs


def rigid_body_modes(ib):
    """The six rigid-body displacement fields of a vector P1/P2 basis, (ndofs, 6)."""
    X = ib.doflocs
    comp = np.empty(ib.N, int)
    for k in range(3):
        comp[ib.nodal_dofs[k]] = k
        if hasattr(ib, "edge_dofs") and ib.edge_dofs.size:
            comp[ib.edge_dofs[k]] = k
    B = np.zeros((ib.N, 6))
    for k in range(3):
        B[comp == k, k] = 1.0
    x, y, z = X
    for j, (a, b_) in enumerate(((0, 1), (1, 2), (2, 0))):     # rotations about z, x, y
        ca, cb = [x, y, z][a], [x, y, z][b_]
        B[comp == a, 3 + j] = -cb[comp == a]
        B[comp == b_, 3 + j] = ca[comp == b_]
    return B


def linear_solver(ib, K, D, kind="direct", tol=1e-9):
    """u = lin(f) with the dofs D held at zero."""
    I = ib.complement_dofs(D)
    KI = K[I][:, I].tocsr()
    if kind == "direct":
        from scipy.sparse.linalg import factorized
        lu = factorized(KI.tocsc())

        def lin(f):
            u = np.zeros(K.shape[0]); u[I] = lu(f[I]); return u
        return lin
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(KI, B=rigid_body_modes(ib)[I], strength=("symmetric", {"theta": 0.0}),
                                           smooth="energy", max_coarse=2000)

    def lin(f):
        res = []
        x = ml.solve(f[I], tol=tol, accel="cg", maxiter=2000, residuals=res)
        if res[-1] > tol * res[0] * 10:
            raise RuntimeError(f"AMG-CG did not converge: {res[-1] / res[0]:.1e}")
        u = np.zeros(K.shape[0]); u[I] = x; return u
    return lin


def displacement(mesh, material, fixed, loads):
    """Displacement field of the linear static problem of solve(): (basis, u)."""
    ib, K, D, rhs = _assemble(mesh, material, fixed)
    return ib, linear_solver(ib, K, D)(sum(rhs(sel, F) for sel, F in loads))
