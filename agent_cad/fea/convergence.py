"""Mesh quality and mesh convergence of the solid FEA.

    q = mesh_quality(mesh)                          # shape quality of every tetrahedron
    rows = [run(h, curv) for h, curv in levels]     # the same problem on finer and finer meshes
    ext = richardson([r["h_eff"] for r in rows], [r["value"] for r in rows])

The convergence of a result is judged by Richardson extrapolation on the three finest
meshes with the Grid Convergence Index of Roache as computed by Celik et al. (2008,
J. Fluids Eng. 130, 078001; ASME V&V 20): observed order p, the extrapolated value and
GCI_fine = 1.25 |e21| / (r21^p - 1), the error band of the finest mesh. A result whose
changes do not shrink (oscillating, or growing like a stress singularity) gets no
extrapolation and is reported as such: a rigid support's re-entrant edge has an
unbounded peak stress, so a peak there is a property of the mesh, not of the part.
A result that changes by less than SMALL_CHANGE on the finest step is taken as converged
(its order cannot be resolved from changes that small).
"""
import math
import numpy as np

SMALL_CHANGE = 0.01          # a change below 1 % on the finest step counts as converged


def tet_volumes(mesh):
    P = mesh.p[:, mesh.t]                         # (3, 4, n)
    a, b, c = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]
    return np.einsum("ij,ij->j", a, np.cross(b, c, axis=0)) / 6


def mesh_quality(mesh):
    """Mean-ratio shape quality q = 12 (3 V)^(2/3) / sum(l^2) (1 for a regular tetrahedron,
    0 for a flat one), the minimum dihedral angle and the edge-length aspect ratio of every
    element, summarised: minimum, 1st percentile, median, and the share of poor elements
    (q < 0.2: P2 stress accuracy degrades; q < 0.1: unusable)."""
    P = mesh.p[:, mesh.t]
    V = np.abs(tet_volumes(mesh))
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    L = np.array([np.linalg.norm(P[:, i] - P[:, j], axis=0) for i, j in pairs])
    q = 12 * (3 * V) ** (2 / 3) / (L ** 2).sum(axis=0)
    # dihedral angles from the face normals
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    N = []
    for i, j, k in faces:
        nrm = np.cross(P[:, j] - P[:, i], P[:, k] - P[:, i], axis=0)
        N.append(nrm / np.linalg.norm(nrm, axis=0))
    dih = []
    for a in range(4):
        for b in range(a + 1, 4):
            dih.append(180 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->j", N[a], N[b]), -1, 1))))
    dmin = np.min(dih, axis=0)
    ar = L.max(axis=0) / L.min(axis=0)
    return {"tets": int(mesh.t.shape[1]), "nodes": int(mesh.p.shape[1]),
            "volume_mm3": round(float(V.sum()), 1), "h_eff_mm": round(float((V.sum() / len(V)) ** (1 / 3)), 3),
            "quality_min": round(float(q.min()), 3), "quality_p01": round(float(np.percentile(q, 1)), 3),
            "quality_median": round(float(np.median(q)), 3),
            "share_q_below_0.2_pct": round(float(100 * np.mean(q < 0.2)), 3),
            "share_q_below_0.1_pct": round(float(100 * np.mean(q < 0.1)), 4),
            "dihedral_min_deg": round(float(dmin.min()), 2), "dihedral_p01_deg": round(float(np.percentile(dmin, 1)), 2),
            "aspect_max": round(float(ar.max()), 2), "aspect_p99": round(float(np.percentile(ar, 99)), 2)}


def richardson(h, f, Fs=1.25):
    """Observed order, extrapolated value and GCI of the finest of the last three meshes
    (h: representative element sizes, coarse to fine; f: the result on each)."""
    h1, h2, h3 = h[-1], h[-2], h[-3]               # 1 = finest
    f1, f2, f3 = f[-1], f[-2], f[-3]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    out = {"f_fine": f1, "r21": round(r21, 3), "r32": round(r32, 3), "change_fine_pct": None}
    if f1:
        out["change_fine_pct"] = round(100 * abs(e21 / f1), 2)
    if abs(e21) <= SMALL_CHANGE * abs(f1):
        # below the scatter of the result's own rounding and of the remeshing: converged, with
        # the last change as the error band
        return {**out, "status": "converged", "p": None, "f_extrap": f1,
                "gci_fine_pct": round(Fs * abs(e21 / f1) * 100, 2) if f1 else 0.0}
    if abs(e32) < 1e-12 * max(abs(f2), 1e-30) or e32 * e21 < 0:
        return {**out, "status": "oscillating", "p": None, "f_extrap": None, "gci_fine_pct": None}
    s = 1.0 if e32 / e21 > 0 else -1.0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(100):                           # Celik eq. (3)-(4): fixed point for unequal ratios
        q = math.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    if abs(e21) > abs(e32) * 0.98:
        # the change does not shrink: no asymptotic range (singular peak, or too coarse)
        return {**out, "status": "diverging", "p": round(p, 2), "f_extrap": None, "gci_fine_pct": None}
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    gci = Fs * abs(e21 / f1) / (r21 ** p - 1) * 100
    return {**out, "status": "converging", "p": round(p, 2), "f_extrap": round(f_ext, 4),
            "gci_fine_pct": round(gci, 2)}
