import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skfem import MeshTet
z = np.load("field.npz"); m = MeshTet(z["p"], z["t"]); vm = z["vm_el"]
bf = m.boundary_facets(); tri = m.facets[:, bf].T
el = m.f2t[0, bf]
P = m.p.T; V = P[tri]
vals = vm[el]
fig = plt.figure(figsize=(13, 6), dpi=120)
norm = matplotlib.colors.Normalize(0, 80); cmap = matplotlib.colormaps["turbo"]
for i, (elv, az, ttl) in enumerate([(25, -60, "Front iso"), (20, 120, "Rear iso")]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d", proj_type="ortho")
    ax.add_collection3d(Poly3DCollection(V, facecolors=cmap(norm(vals)), edgecolors="none"))
    c = (P.max(0) + P.min(0)) / 2; r = (P.max(0) - P.min(0)).max() / 2 * 0.62
    for k, f in enumerate("xyz"): getattr(ax, f"set_{f}lim")(c[k] - r, c[k] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elv, az); ax.set_axis_off(); ax.set_title(ttl)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="von Mises stress [MPa]  (1000 N vertical on bar clamp)")
plt.savefig("stress.png", facecolor="white")
