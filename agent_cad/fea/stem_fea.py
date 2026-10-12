"""Linear static FEA of the stem body: steerer bore fixed, vertical load on bar bore."""
import sys, json, math, numpy as np, gmsh, meshio

from agent_cad.parts import stem
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot

step, F, h = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
p = stem.StemParams()
d, n, S, B = stem.frame(p)
E, nu, Sy = 68.9e3, 0.33, 276.0          # A6061-T6, MPa
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.occ.importShapes(step); gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h / 3)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
gmsh.model.mesh.generate(3)
_, xyz, _ = gmsh.model.mesh.getNodes()
X = xyz.reshape(-1, 3)
et, tags, conn = gmsh.model.mesh.getElements(3)
T = np.array(conn[0], dtype=np.int64).reshape(-1, 4) - 1
gmsh.finalize()
used = np.unique(T); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
mesh = MeshTet(X[used].T, remap[T].T)
print("nodes", mesh.p.shape[1], "tets", mesh.t.shape[1], file=sys.stderr)
e = ElementVector(ElementTetP2())
ib = Basis(mesh, e)
lam, mu = lame_parameters(E, nu)
K = asm(linear_elasticity(lam, mu), ib)
fp = mesh.p[:, mesh.facets[:, mesh.boundary_facets()]].mean(axis=1)
bf = mesh.boundary_facets()
# steerer bore: radius ~ steerer_d/2
r_s = np.hypot(fp[0], fp[1])
fix = bf[np.abs(r_s - p.steerer_d / 2) < 0.3]
# bar bore: distance from bar axis (Y through B)
rel = fp - np.array([[B.X], [B.Y], [B.Z]])
r_b = np.hypot(rel[0], rel[2])
load = bf[np.abs(r_b - p.bar_d / 2) < 0.3]
fb = FacetBasis(mesh, e, facets=load)
area = asm(Functional(lambda w: 1.0 + 0 * w.x[0]), FacetBasis(mesh, ElementTetP1(), facets=load))
t = np.array([0, 0, -F / area])
@LinearForm
def trac(v, w): return v[0] * t[0] + v[1] * t[1] + v[2] * t[2]
f = asm(trac, fb)
D = ib.get_dofs(fix).all()
u = solve(*condense(K, f, D=D))
# von Mises at quadrature points
sb = ib.with_element(ElementVector(ElementTetP1()))  # placeholder not used
from skfem.helpers import sym_grad
uh = ib.interpolate(u)
eps = sym_grad(uh)
tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
sig = 2 * mu * eps + lam * np.einsum('ij,...->ij...', np.eye(3), tr)
s = sig
vm = np.sqrt(0.5 * ((s[0,0]-s[1,1])**2 + (s[1,1]-s[2,2])**2 + (s[2,2]-s[0,0])**2) + 3 * (s[0,1]**2 + s[1,2]**2 + s[2,0]**2))
vm_el = vm.max(axis=1)
xq = ib.mapping.F(ib.X)  # (3, nel, nq)
iel = np.argmax(vm_el); qi = np.argmax(vm[iel])
loc = xq[:, iel, qi]
disp = u[ib.nodal_dofs].T if False else None
umag = np.sqrt(sum(u[ib.nodal_dofs[i]] ** 2 for i in range(3)))
res = dict(load_N=F, mesh_h=h, tets=int(mesh.t.shape[1]), dofs=int(len(u)),
           vm_max_MPa=round(float(vm_el.max()), 1), vm_p999_MPa=round(float(np.percentile(vm_el, 99.9)), 1),
           vm_max_location_mm=[round(float(c), 1) for c in loc], max_disp_mm=round(float(umag.max()), 3),
           yield_MPa=Sy, safety_factor_vs_yield=round(Sy / float(vm_el.max()), 2))
print(json.dumps(res))
# save field for plotting
np.savez("field.npz", p=mesh.p, t=mesh.t, vm_el=vm_el, u=np.array([u[ib.nodal_dofs[i]] for i in range(3)]))
