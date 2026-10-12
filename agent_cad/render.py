import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def tris(shape, tol=0.08):
    v, t = shape.tessellate(tol, 0.2)
    V = np.array([(p.X, p.Y, p.Z) for p in v]); return V[np.array(t)]

def render(parts, path, views=((25, -60), (20, 120), (90, -90), (0, -90)), titles=None):
    fig = plt.figure(figsize=(12, 10), dpi=120)
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    allv = np.concatenate([tris(s).reshape(-1, 3) for s, _ in parts])
    c = (allv.max(0) + allv.min(0)) / 2; r = (allv.max(0) - allv.min(0)).max() / 2 * 0.62
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d", proj_type="ortho")
        for s, col in parts:
            T = tris(s)
            nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            sh = 0.35 + 0.65 * np.abs(nrm @ light)
            base = np.array(matplotlib.colors.to_rgb(col))
            pc = Poly3DCollection(T, facecolors=np.clip(base * sh[:, None], 0, 1), edgecolors="none")
            ax.add_collection3d(pc)
        for k, f in enumerate("xyz"):
            getattr(ax, f"set_{f}lim")(c[k] - r, c[k] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.set_axis_off()
        if titles: ax.set_title(titles[i], fontsize=10)
    plt.tight_layout(); plt.savefig(path, facecolor="white"); plt.close(fig)
