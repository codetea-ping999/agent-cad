"""Linear 3D frame (Euler-Bernoulli beam) analysis of a tubular centreline.

Used for the handlebar: the bar is a slender thin-walled tube, so a beam model along
the centreline gives section forces everywhere in milliseconds, which is what a
verification loop over parameters needs. The solid FEA (fea/static.py) cross-checks it.

Curved tube bends are softer and more highly stressed than a straight beam of the
same section, because the section ovalises. Bend elements carry the flexibility factor
k = 1.65 / h and the stress intensification factors i_in = 0.9 / h^(2/3) (in-plane),
i_out = 0.75 / h^(2/3) (out-of-plane), h = t R / r_m^2 (ASME B31.3 Appendix D, both >= 1).

    nodes, elems = discretise(segments, section_of)
    u, forces = solve(nodes, elems, fixed_nodes, [(node, F, M)], E, nu)
"""
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class Section:
    ro: float
    ri: float

    @property
    def A(self):
        return math.pi * (self.ro ** 2 - self.ri ** 2)

    @property
    def I(self):
        return math.pi / 4 * (self.ro ** 4 - self.ri ** 4)

    @property
    def J(self):
        return 2 * self.I

    @property
    def Z(self):
        return self.I / self.ro


@dataclass
class Elem:
    n0: int
    n1: int
    sec: Section
    e2: np.ndarray            # local axis 2 (unit, normal to the element): toward the centre of curvature on bends
    seg: str                  # segment name (L0, B1, ...)
    R: float = 0.0            # bend radius (0: straight)
    k_flex: float = 1.0
    i_in: float = 1.0
    i_out: float = 1.0


def bend_factors(sec, R):
    t, rm = sec.ro - sec.ri, (sec.ro + sec.ri) / 2
    h = t * R / rm ** 2
    return max(1.65 / h, 1.0), max(0.9 / h ** (2 / 3), 1.0), max(0.75 / h ** (2 / 3), 1.0)


def discretise(segs, section_of, h=4.0, bend_effects=True):
    """segs: objects with kind, length, R, name, point(s) -> (xyz, tangent), and n0 (arcs).
    section_of(seg, s_mid_point_xyz) -> Section. Returns node coordinates (n, 3) and elements."""
    nodes, elems, index = [], [], {}

    def node(x):                       # segments that meet share their end node
        key = tuple(np.round(x, 5) + 0.0)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(np.asarray(x, float))
        return index[key]
    for s in segs:
        n = max(1, int(math.ceil(s.length / h)))
        us = np.linspace(0, s.length, n + 1)
        prev = node(s.point(0)[0])
        for a, b in zip(us[:-1], us[1:]):
            xb = s.point(b)[0]
            nb = node(xb)
            xm, tm = s.point((a + b) / 2)
            sec = section_of(s, xm)
            if s.kind == "arc":
                c = np.array(s.p0) + s.R * np.array(s.n0)
                e2 = c - xm
            else:
                e2 = np.cross(tm, [0, 0, 1.0]) if abs(tm[2]) < 0.9 else np.cross(tm, [1.0, 0, 0])
            e = Elem(prev, nb, sec, e2 / np.linalg.norm(e2), s.name, s.R if s.kind == "arc" else 0.0)
            if s.kind == "arc" and bend_effects:
                e.k_flex, e.i_in, e.i_out = bend_factors(sec, s.R)
            elems.append(e)
            prev = nb
    return np.array(nodes), elems


def _frame(X, e):
    d = X[e.n1] - X[e.n0]
    L = float(np.linalg.norm(d))
    e1 = d / L
    e2 = e.e2 - np.dot(e.e2, e1) * e1
    e2 /= np.linalg.norm(e2)
    return L, np.vstack([e1, e2, np.cross(e1, e2)])


def _k_local(L, E, G, A, I2, I3, J):
    """12 x 12 stiffness, dofs (u1 u2 u3 r1 r2 r3) at both ends; I2 about local 2, I3 about local 3."""
    k = np.zeros((12, 12))
    a = E * A / L
    k[0, 0] = k[6, 6] = a; k[0, 6] = k[6, 0] = -a
    t = G * J / L
    k[3, 3] = k[9, 9] = t; k[3, 9] = k[9, 3] = -t
    # bending in the 1-2 plane (about 3): v = u2, theta = r3
    c = E * I3 / L ** 3
    for (i, j), v in {(1, 1): 12, (1, 5): 6 * L, (1, 7): -12, (1, 11): 6 * L, (5, 5): 4 * L * L, (5, 7): -6 * L,
                      (5, 11): 2 * L * L, (7, 7): 12, (7, 11): -6 * L, (11, 11): 4 * L * L}.items():
        k[i, j] = k[j, i] = c * v
    # bending in the 1-3 plane (about 2): w = u3, theta = r2 (sign flips)
    c = E * I2 / L ** 3
    for (i, j), v in {(2, 2): 12, (2, 4): -6 * L, (2, 8): -12, (2, 10): -6 * L, (4, 4): 4 * L * L, (4, 8): 6 * L,
                      (4, 10): 2 * L * L, (8, 8): 12, (8, 10): 6 * L, (10, 10): 4 * L * L}.items():
        k[i, j] = k[j, i] = c * v
    return k


def solve(X, elems, fixed, loads, E, nu):
    """fixed: node indices (all 6 dofs held); loads: [(node, F(3), M(3))].
    Returns displacements (n, 6) and per-element end forces in local axes
    [(N, V2, V3, T, M2, M3) at end 0, same at end 1] (internal forces acting on the element)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    G = E / (2 * (1 + nu))
    n = len(X)
    rows, cols, vals, cache = [], [], [], []
    for e in elems:
        L, Rm = _frame(X, e)
        s = e.sec
        # bends: in-plane bending is about local 3 (axis of the bend), out-of-plane about local 2
        kl = _k_local(L, E, G, s.A, s.I / e.k_flex, s.I / e.k_flex, s.J)
        T = np.kron(np.eye(4), Rm)
        kg = T.T @ kl @ T
        dofs = np.r_[6 * e.n0:6 * e.n0 + 6, 6 * e.n1:6 * e.n1 + 6]
        rows.append(np.repeat(dofs, 12)); cols.append(np.tile(dofs, 12)); vals.append(kg.ravel())
        cache.append((kl, T, dofs))
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(6 * n, 6 * n)).tocsr()
    f = np.zeros(6 * n)
    for i, F, M in loads:
        f[6 * i:6 * i + 3] += F
        f[6 * i + 3:6 * i + 6] += M
    held = np.zeros(6 * n, bool)
    for i in fixed:
        held[6 * i:6 * i + 6] = True
    free = ~held
    u = np.zeros(6 * n)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    forces = []
    for kl, T, dofs in cache:
        fl = kl @ (T @ u[dofs])
        forces.append((-fl[:6], fl[6:]))       # internal force at end 0 (sign flipped) and end 1
    return u.reshape(n, 6), forces


def stresses(elems, forces):
    """Peak von Mises on the outer fibre at both element ends: bending with the bend's
    SIFs, axial, torsion and shear neglected (slender tube). Returns (n_elem, 2)."""
    out = np.zeros((len(elems), 2))
    for k, (e, ends) in enumerate(zip(elems, forces)):
        s = e.sec
        for j, fe in enumerate(ends):
            N, _, _, Tq, M2, M3 = fe
            sb = math.hypot(e.i_out * M2, e.i_in * M3) / s.Z
            sa = abs(N) / s.A
            tau = abs(Tq) * s.ro / s.J
            out[k, j] = math.sqrt((sb + sa) ** 2 + 3 * tau ** 2)
    return out
