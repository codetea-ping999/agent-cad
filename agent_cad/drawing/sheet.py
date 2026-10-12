"""Drawing sheet model: orthographic/section views by hidden-line removal and
a display list of annotation primitives, rendered to PDF (matplotlib) and DXF
(ezdxf) from the same data.

Conventions follow JIS B 0001 (mechanical drawing) / JIS Z 8311 (sheet):
third-angle projection, mm, line types outline 0.5 / thin 0.25,
hidden lines omitted where holes are described by notes.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
from build123d import Compound, Vector, Plane, Box, Shape, GeomType, Align

A3 = (420.0, 297.0)
# style -> (lineweight mm, linetype)
STYLES = {
    "outline": (0.5, "solid"), "hidden": (0.25, "dashed"), "center": (0.18, "dashdot"),
    "thin": (0.18, "solid"), "cut": (0.5, "dashdot"), "frame": (0.7, "solid"), "hatch": (0.13, "solid"),
}


def text_width(s, size):
    """Approximate rendered width (mm): full-width glyphs ~1.0 em, others ~0.55 em."""
    import unicodedata
    return sum(size * (1.25 if unicodedata.east_asian_width(ch) in "WFA" else 0.64) for ch in s)


def text_box(xy, s, size, rot=0.0, ha="center", va="bottom"):
    """Axis-aligned bounding box (min, max) of a text item on the sheet."""
    w, h = text_width(s, size), 1.25 * size
    ox = {"left": 0, "center": -w / 2, "right": -w}[ha]
    oy = {"bottom": 0, "center": -h / 2, "top": -h}[va]
    c = np.array([[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]])
    r = math.radians(rot)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    pts = c @ R.T + np.asarray(xy, float)
    return pts.min(0), pts.max(0)


def boxes_overlap(a, b, shrink=0.3):
    (a0, a1), (b0, b1) = a, b
    return (min(a1[0], b1[0]) - max(a0[0], b0[0]) > shrink) and (min(a1[1], b1[1]) - max(a0[1], b0[1]) > shrink)


def _vec(v):
    return np.array([v.X, v.Y, v.Z]) if hasattr(v, "X") else np.asarray(v, float)


def _discretize(edge, tol=0.05):
    L = edge.length
    if edge.geom_type == GeomType.LINE:
        n = 1
    else:
        n = max(4, int(L / 0.6))
    return [edge.position_at(i / n) for i in range(n + 1)]


@dataclass
class View:
    name: str
    shape: Shape
    direction: tuple        # viewing direction: from the eye toward the part
    xdir: tuple             # sheet-right direction in model space
    origin: tuple           # sheet position (mm) of the model origin
    scale: float = 1.0
    section: tuple | None = None   # (plane origin, plane normal): keep the side the normal points to
    label: str = ""
    vis: list = field(default_factory=list)
    cut_faces: list = field(default_factory=list)

    @property
    def ydir(self):
        d, x = -_vec(Vector(*self.direction)), _vec(Vector(*self.xdir))
        return np.cross(d, x)

    def to_sheet(self, p):
        p = _vec(p)
        x, y = p @ _vec(Vector(*self.xdir)), p @ self.ydir
        return np.array(self.origin) + self.scale * np.array([x, y])

    def build(self):
        shape = self.shape
        if self.section:
            o, n = Vector(*self.section[0]), Vector(*self.section[1])
            pl = Plane(origin=o, z_dir=n)
            keep = pl * Box(2000, 2000, 1000, align=(Align.CENTER, Align.CENTER, Align.MIN))
            cut = shape & keep
            shape = cut
            # faces lying on the cutting plane get hatched
            for f in cut.faces():
                if f.geom_type == GeomType.PLANE and abs(abs(f.normal_at().dot(n)) - 1) < 1e-6 \
                        and abs((f.center() - o).dot(n)) < 1e-6:
                    loops = [[self.to_sheet(p) for e in w.edges() for p in _discretize(e)] for w in f.wires()]
                    self.cut_faces.append(loops)
        eye = -_vec(Vector(*self.direction))
        a = HLRBRep_Algo(); a.Add(shape.wrapped)
        a.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*eye), gp_Dir(*self.xdir))))
        a.Update(); a.Hide()
        h = HLRBRep_HLRToShape(a)
        for c in (h.VCompound(), h.OutLineVCompound()):
            if c.IsNull():
                continue
            for e in Compound(c).edges():
                pts = [np.array(self.origin) + self.scale * np.array([p.X, p.Y]) for p in _discretize(e)]
                self.vis.append(np.array(pts))
        return self

    def bbox(self):
        pts = np.vstack(self.vis)
        return pts.min(0), pts.max(0)

    def shift(self, dx, dy):
        o = np.array([dx, dy])
        self.origin = tuple(np.array(self.origin) + o)
        self.vis = [v + o for v in self.vis]
        self.cut_faces = [[np.asarray(l) + o for l in loops] for loops in self.cut_faces]


A2 = (594.0, 420.0)


class Sheet:
    """Display list: ('line', pts, style) | ('fill', pts) | ('text', x, y, s, size, rot, ha, va)
    | ('hatch', loops)."""

    def __init__(self, size=A3):
        self.size = size
        self.items = []
        self.views = []
        self.meta = {}

    # ---- primitives
    def line(self, pts, style="thin"):
        self.items.append(("line", np.asarray(pts, float), style))

    def fill(self, pts):
        self.items.append(("fill", np.asarray(pts, float)))

    def text(self, xy, s, size=3.5, rot=0.0, ha="center", va="bottom"):
        self.items.append(("text", float(xy[0]), float(xy[1]), s, size, rot, ha, va))

    def outline_points(self, step=0.5):
        pts = []
        for it in self.items:
            if it[0] == "line" and it[2] == "outline":
                P = it[1]
                for a, b in zip(P[:-1], P[1:]):
                    k = max(1, int(np.linalg.norm(b - a) / step))
                    pts.append(a + np.outer(np.linspace(0, 1, k + 1), b - a))
        return np.vstack(pts) if pts else np.zeros((0, 2))

    def is_free(self, box, pad=0.3):
        """True if the box clears part outlines, existing texts and the sheet margin."""
        b0, b1 = box
        W, H = self.size
        if b0[0] < 12 or b0[1] < 12 or b1[0] > W - 12 or b1[1] > H - 12:
            return False
        if getattr(self, "_opts", None) is None or len(self._opts) == 0:
            self._opts = self.outline_points()
        P = self._opts
        if ((P[:, 0] > b0[0] - pad) & (P[:, 0] < b1[0] + pad) & (P[:, 1] > b0[1] - pad) & (P[:, 1] < b1[1] + pad)).any():
            return False
        for it in self.items:
            if it[0] == "text" and boxes_overlap(box, text_box((it[1], it[2]), it[3], it[4], it[5], it[6], it[7]), -pad):
                return False
        return True

    def add_view(self, v: View, built=False):
        if not built:
            v.build()
        self.views.append(v)
        for pts in v.vis:
            self.line(pts, "outline")
        for loops in v.cut_faces:
            self.items.append(("hatch", [np.array(l) for l in loops]))
        return v

    def views_region(self, margin=10.0):
        W, H = self.size
        return (margin + 6, 115.0, W - margin - 6, H - margin - 6)

    def place_third_angle(self, front, top=None, left=None, right=None, gap=30.0, region=None, edge=18.0, gap_right=None):
        """Lay out views in third-angle arrangement: top above front, left view to
        the left, right/auxiliary view to the right; `gap` leaves room for
        dimensions. Views must be built at origin (0, 0). Returns False if they
        do not fit the region (caller then picks a larger sheet or scale)."""
        x0, y0, x1, y1 = region or self.views_region()
        bb = {k: v.bbox() for k, v in dict(front=front, top=top, left=left, right=right).items() if v is not None}
        size = {k: b[1] - b[0] for k, b in bb.items()}
        wl = size["left"][0] + gap if left else 0
        gap_right = gap if gap_right is None else gap_right
        wr = size["right"][0] + gap_right if right else 0
        W = wl + size["front"][0] + wr
        H = size["front"][1] + (size["top"][1] + gap if top else 0)
        hl = max(size[k][1] for k in ("left", "right") if k in size) if (left or right) else 0
        H = max(H, hl)
        if W + 2 * edge > x1 - x0 or H + 2 * edge > y1 - y0:
            return False
        fx = x0 + ((x1 - x0) - W) / 2 + wl - bb["front"][0][0]
        fy = y0 + ((y1 - y0) - H) / 2 - bb["front"][0][1]
        front.shift(fx, fy)
        fb = front.bbox()
        if top:
            top.shift(front.origin[0] - top.origin[0], fb[1][1] + gap - bb["top"][0][1])
        cy = (fb[0][1] + fb[1][1]) / 2
        if left:   # same height datum as the front view (third-angle projection)
            left.shift(fb[0][0] - gap - bb["left"][1][0], front.origin[1] - left.origin[1])
        if right:  # auxiliary view: centred on the front view
            right.shift(fb[1][0] + gap_right - bb["right"][0][0], cy - (bb["right"][0][1] + bb["right"][1][1]) / 2)
        for v in (front, top, left, right):
            if v is not None:
                self.add_view(v, built=True)
        return True

    # ---- annotation helpers
    def arrow(self, tip, direction, length=3.0, width=1.0):
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        tip = np.asarray(tip, float)
        self.fill([tip, tip - d * length + n * width / 2, tip - d * length - n * width / 2])

    def centerline(self, a, b, ext=3.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (b - a) / np.linalg.norm(b - a)
        self.line([a - u * ext, b + u * ext], "center")

    def center_mark(self, c, r, ext=3.0):
        c = np.asarray(c, float)
        self.line([c - (r + ext, 0), c + (r + ext, 0)], "center")
        self.line([c - (0, r + ext), c + (0, r + ext)], "center")

    def dim(self, p1, p2, offset, text, u=None, tol=None, gap=1.0, ext=2.0):
        """Linear dimension between sheet points p1, p2 measured along unit
        vector u (default: along p1->p2), dimension line shifted by `offset`
        mm to the left of u. tol: (upper, lower) strings stacked after text."""
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        u = (p2 - p1) if u is None else np.asarray(u, float)
        u = u / np.linalg.norm(u)
        n = np.array([-u[1], u[0]])
        base = p1 + n * offset
        L = (p2 - p1) @ u
        a, b = base, base + u * L
        for p, q in ((p1, a), (p2, b)):
            s = np.sign((q - p) @ n) or 1.0
            self.line([p + n * s * gap, q + n * s * ext], "thin")
        self.line([a, b], "thin")
        if abs(L) > 8:
            self.arrow(a, a - b); self.arrow(b, b - a)
        else:  # arrows outside for short dimensions
            self.line([a - u * 6, b + u * 6], "thin")
            self.arrow(a, b - a); self.arrow(b, a - b)
        self._dim_text((a + b) / 2 + n * 0.8, u, text, tol)
        return L

    def _dim_text(self, pos, u, text, tol=None, size=3.5):
        ang = math.degrees(math.atan2(u[1], u[0]))
        if ang > 90.01 or ang <= -90:  # JIS: readable from the bottom or the right
            ang -= 180 * np.sign(ang)
        if tol:
            r = math.radians(ang)
            t = np.array([math.cos(r), math.sin(r)]); nn = np.array([-t[1], t[0]])
            w, wt = text_width(text, size), text_width(tol[0], size * 0.7)
            c0 = pos - t * (wt + 0.8) / 2
            self.text(c0, text, size, ang, "center", "bottom")
            p = c0 + t * (w / 2 + 0.8)
            self.text(p + nn * 3.3, tol[0], size * 0.7, ang, "left", "bottom")
            self.text(p, tol[1], size * 0.7, ang, "left", "bottom")
        else:
            self.text(pos, text, size, ang, "center", "bottom")

    def _leader_layout(self, tip, knee, lines, size, right):
        right = knee[0] >= tip[0] if right is None else right
        w = max(text_width(s, size) for s in lines) + 2
        end = knee + (w if right else -w, 0)
        x0 = min(knee[0], end[0]) + 0.5
        pos = []
        for i, s in enumerate(lines):
            y = knee[1] + 0.8 if i == 0 else knee[1] - 1.2 - (i - 1) * (1.25 * size + 1.0)
            pos.append(((x0, y), s, "bottom" if i == 0 else "top"))
        return end, pos

    def leader(self, tip, knee, text, size=3.5, dot=False, lines=None, right=None):
        """Leader with text on a horizontal shoulder. `knee` may be a list of
        candidate knees: the first whose text lands on free space is used."""
        tip = np.asarray(tip, float)
        lines = lines or [text]
        knees = [np.asarray(k, float) for k in knee] if np.ndim(knee) == 2 else [np.asarray(knee, float)]
        choice = None
        for k in knees:
            for r in ([right] if right is not None else [k[0] >= tip[0], k[0] < tip[0]]):
                end, pos = self._leader_layout(tip, k, lines, size, r)
                if all(self.is_free(text_box(xy, s, size, 0, "left", va)) for xy, s, va in pos):
                    choice = (k, end, pos); break
            if choice:
                break
        if choice is None:
            k = knees[0]; end, pos = self._leader_layout(tip, k, lines, size, right)
            choice = (k, end, pos)
        knee, end, pos = choice
        self.line([tip, knee, end], "thin")
        if dot:
            self.fill([tip + (0.5 * math.cos(t), 0.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12)])
        else:
            self.arrow(tip, tip - knee)
        for xy, s, va in pos:
            self.text(xy, s, size, 0, "left", va)

    def angle_dim(self, vertex, a1_deg, a2_deg, radius, text):
        c = np.asarray(vertex, float)
        ts = np.radians(np.linspace(a1_deg, a2_deg, 30))
        pts = c + radius * np.c_[np.cos(ts), np.sin(ts)]
        self.line(pts, "thin")
        for t, s in ((ts[0], -1), (ts[-1], 1)):
            tan = np.array([-math.sin(t), math.cos(t)]) * s * np.sign(a2_deg - a1_deg)
            self.arrow(c + radius * np.array([math.cos(t), math.sin(t)]), tan)
        sgn = np.sign(a2_deg - a1_deg) or 1
        for am in ((a1_deg + a2_deg) / 2, a1_deg - 5 * sgn, a2_deg + 5 * sgn):
            p = c + (radius + 1.5) * np.array([math.cos(math.radians(am)), math.sin(math.radians(am))])
            if self.is_free(text_box(p, text, 3.5, 0, "left", "center")):
                break
        else:
            am = (a1_deg + a2_deg) / 2
            p = c + (radius + 1.5) * np.array([math.cos(math.radians(am)), math.sin(math.radians(am))])
        self.text(p, text, 3.5, 0, "left", "center")

    def diameter_dim(self, center, r, ang_deg, text, tol=None, outside=10.0, below=0.0):
        """Dimension line through the centre with arrows on the circle, carried out
        to a horizontal shoulder holding the (horizontal) text. `ang_deg` may be a
        list of candidate directions: the first whose text (and `below` mm of
        space under it, e.g. for feature control frames) is free is used.
        Returns the shoulder text origin."""
        c = np.asarray(center, float)
        w = text_width(text, 3.5) + (text_width(tol[0], 2.5) + 1.5 if tol else 0) + 2
        cands = list(np.atleast_1d(ang_deg))
        for a in cands + [cands[0]]:
            u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            knee = c + u * (r + outside)
            right = u[0] >= 0
            end = knee + ((w if right else -w), 0)
            x0 = min(knee[0], end[0]) + 1
            box = (np.array([x0, knee[1] - below]), np.array([x0 + w, knee[1] + 6.0]))
            if self.is_free(box):
                break
        p1, p2 = c - u * r, c + u * r
        self.line([p1, knee, end], "thin")
        self.arrow(p1, -u); self.arrow(p2, u)
        self.text((x0, knee[1] + 0.8), text, 3.5, 0, "left", "bottom")
        if tol:
            xt = x0 + text_width(text, 3.5) + 0.8
            self.text((xt, knee[1] + 4.2), tol[0], 2.5, 0, "left", "bottom")
            self.text((xt, knee[1] + 0.8), tol[1], 2.5, 0, "left", "bottom")
        return np.array([x0, knee[1]])

    def fcf(self, pos, cells, size=3.5, attach=None):
        """Feature control frame (JIS B 0021). First cell is a characteristic
        name ('flatness', 'cylindricity', 'perpendicularity', 'parallelism',
        'position') drawn as geometry, then tolerance and datum cells."""
        x, y = pos; h = size + 3
        widths = [h] + [text_width(c, size) + 3 for c in cells[1:]]
        xx = x
        for i, (c, w) in enumerate(zip(cells, widths)):
            self.line([(xx, y), (xx + w, y), (xx + w, y + h), (xx, y + h), (xx, y)], "thin")
            if i == 0:
                self._gdt_symbol((xx + w / 2, y + h / 2), c, h * 0.62)
            else:
                self.text((xx + w / 2, y + h / 2), c, size, 0, "center", "center")
            xx += w
        if attach is not None:
            self.line([(x, y + h / 2), attach], "thin")
            self.arrow(attach, np.asarray(attach) - (x, y + h / 2))
        self.meta.setdefault("gdt", []).append(cells)
        return xx - x, h

    def _gdt_symbol(self, c, name, s):
        c = np.asarray(c, float); h = s / 2
        if name == "flatness":
            self.line([c + (-h, -h * 0.5), c + (h * 0.5, -h * 0.5), c + (h, h * 0.5), c + (-h * 0.5, h * 0.5), c + (-h, -h * 0.5)], "thin")
        elif name == "cylindricity":
            t = np.linspace(0, 2 * math.pi, 30); r = h * 0.55
            self.line(np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)], "thin")
            for sgn in (-1, 1):
                self.line([c + (sgn * r - h * 0.35, -h), c + (sgn * r + h * 0.35, h)], "thin")
        elif name == "perpendicularity":
            self.line([c + (-h, -h * 0.8), c + (h, -h * 0.8)], "thin"); self.line([c + (0, -h * 0.8), c + (0, h)], "thin")
        elif name == "parallelism":
            for dx in (-0.3, 0.3):
                self.line([c + ((dx - 0.3) * h, -h), c + ((dx + 0.3) * h, h)], "thin")
        elif name == "position":
            t = np.linspace(0, 2 * math.pi, 30); r = h * 0.6
            self.line(np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)], "thin")
            self.line([c + (-h, 0), c + (h, 0)], "thin"); self.line([c + (0, -h), c + (0, h)], "thin")
        else:
            raise ValueError(name)

    def datum(self, foot, direction, letter, length=7.0, size=3.5):
        """Datum feature symbol: filled triangle on the feature, square frame with the letter."""
        f = np.asarray(foot, float); d = np.asarray(direction, float); d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        self.fill([f + n * 1.5, f - n * 1.5, f + d * 2.6])
        top = f + d * length
        self.line([f + d * 2.6, top], "thin")
        h = size + 3
        c = top + d * h / 2
        sq = [c + (-h / 2, -h / 2), c + (h / 2, -h / 2), c + (h / 2, h / 2), c + (-h / 2, h / 2), c + (-h / 2, -h / 2)]
        self.line(sq, "thin")
        self.text(c, letter, size, 0, "center", "center")

    def roughness_leader(self, tip, knee, ra, size=3.5):
        """Surface texture symbol on the end of a leader pointing at the surface (JIS B 0031)."""
        tip, knee = np.asarray(tip, float), np.asarray(knee, float)
        self.line([tip, knee], "thin"); self.arrow(tip, tip - knee)
        self.roughness(knee, ra, size)

    def roughness(self, pos, ra, size=3.5, rot=0.0):
        """JIS B 0031 symbol: removal of material required, with Ra value."""
        x, y = pos; h = size * 1.4
        r = math.radians(rot)
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        pts = np.array([(-h * 0.58, h), (0, 0), (h * 1.15, h * 2), (h * 1.15 + text_width(f'Ra {ra}', size) + 1.5, h * 2)])
        self.line(pts @ R.T + (x, y), "thin")
        self.line(np.array([(-h * 0.58, h), (h * 0.58, h)]) @ R.T + (x, y), "thin")
        self.text(np.array([h * 1.3, h * 2 + 0.6]) @ R.T + (x, y), f"Ra {ra}", size, rot, "left", "bottom")

    def section_line(self, a, b, label, look):
        """Cutting-plane line (thin chain, thick ends) with arrows showing the
        viewing direction `look` (sheet vector) of the section view."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (b - a) / np.linalg.norm(b - a)
        lk = np.asarray(look, float); lk /= np.linalg.norm(lk)
        self.line([a, b], "center")
        for p, s in ((a, 1), (b, -1)):
            self.line([p, p + u * 5 * s], "outline")
            self.line([p, p + lk * 6], "thin"); self.arrow(p + lk * 6, lk)
            self.text(p - u * 4 * s + lk * 3, label, 5, 0, "center", "center")

    # ---- sheet furniture
    def frame_and_title(self, title_rows, margin=10.0):
        W, H = self.size
        self.line([(margin, margin), (W - margin, margin), (W - margin, H - margin), (margin, H - margin), (margin, margin)], "frame")
        # centring marks (JIS Z 8311)
        for p, q in (((W / 2, 0 + 5), (W / 2, margin)), ((W / 2, H - margin), (W / 2, H - 5)),
                     ((5, H / 2), (margin, H / 2)), ((W - margin, H / 2), (W - 5, H / 2))):
            self.line([p, q], "frame")
        # title block: bottom right, 170 wide
        x0, y0, w, rh = W - margin - 170, margin, 170, 7.0
        n = len(title_rows)
        for i in range(n + 1):
            self.line([(x0, y0 + i * rh), (x0 + w, y0 + i * rh)], "frame" if i in (0, n) else "thin")
        self.line([(x0, y0), (x0, y0 + n * rh)], "frame")
        self.line([(x0 + 40, y0), (x0 + 40, y0 + n * rh)], "thin")
        for i, (k, v) in enumerate(reversed(title_rows)):
            self.text((x0 + 2, y0 + i * rh + rh / 2), k, 3.0, 0, "left", "center")
            self.text((x0 + 42, y0 + i * rh + rh / 2), v, 3.5, 0, "left", "center")
        self.meta["title_block"] = dict(title_rows)
        return x0, y0 + n * rh

    def projection_symbol(self, pos, size=10.0):
        """Third-angle projection symbol (JIS B 0001)."""
        x, y = pos; s = size
        self.line([(x, y + s * 0.3), (x + s * 0.9, y), (x + s * 0.9, y + s), (x, y + s * 0.7), (x, y + s * 0.3)], "thin")
        cx = x + s * 1.6
        for r in (s * 0.5, s * 0.25):
            t = np.linspace(0, 2 * math.pi, 40)
            self.line(np.c_[cx + r * np.cos(t), y + s / 2 + r * np.sin(t)], "thin")
        self.line([(x - 1, y + s / 2), (cx + s * 0.7, y + s / 2)], "center")
