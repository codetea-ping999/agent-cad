"""JIS drawing (A3, third-angle) of the injection-moulded housing.

The part is drawn opening up (model -Y up on the sheet): the front view looks at the
window wall, the view above it looks into the box (ribs, bosses, holes) and the
right view is section A-A through the right-hand bosses. Every dimension value is
measured from the B-rep (sections, faces) and recorded in sheet.meta["dims"] next
to the design intent from the parameters, so check_sheet() catches any model /
drawing disagreement. Sizes are given where the draft starts (rim, boss tips,
window at the outer face), as the notes say.

General tolerances: JIS B 0405 class c (coarse), the usual choice for moulded
parts on Japanese drawings; JIS K 7109 is the basis for tighter, resin-specific
tolerances to agree with the moulder.
"""
import math, datetime
import shapely
from shapely.geometry import Polygon as SPolygon, LineString
from build123d import Plane, Vector, section
from ..drawing.sheet import Sheet, View, A2, A3
from ..standards import jis
from ..process import molding as mo
from . import housing

TOL_CLASS = "c"
MATERIAL_SPEC = {  # material -> (drawing designation, colour / finish)
    "PA66-GF30": ("PA66-GF30 (ISO 1874-1, 熱安定化グレード)", "黒、成形のまま"),
    "PA6-GF30": ("PA6-GF30 (ISO 1874-1, 熱安定化グレード)", "黒、成形のまま"),
    "PBT-GF30": ("PBT-GF30 (ISO 7792-1)", "黒、成形のまま"),
    "PP": ("PP (ISO 19069-1, ホモポリマー)", "黒、成形のまま"),
}
TRIES = [(A3, 1.0), (A2, 1.0), (A3, 0.5), (A2, 0.5)]


def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


class _Dims:
    def __init__(self, sheet):
        self.sheet = sheet; sheet.meta["dims"] = []

    def rec(self, name, measured, intent, text):
        self.sheet.meta["dims"].append(dict(name=name, measured=round(float(measured), 4),
                                            intent=float(intent), text=text, tol=None))

    def need(self, name, found, expected):
        self.sheet.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))


# ------------------------------------------------------------------ measuring
def section_local(shape, pl):
    """Section of `shape` by plane `pl` as a shapely geometry in the plane's (x, y)."""
    tris = []
    for f in section(shape, pl).faces():
        vs, ts = f.tessellate(0.005, 0.1)
        P = [tuple(pl.to_local_coords(v))[:2] for v in vs]
        tris += [SPolygon([P[i] for i in t]) for t in ts]
    return shapely.union_all(tris).buffer(0) if tris else SPolygon()


def _polys(g):
    return [x for x in getattr(g, "geoms", [g]) if not x.is_empty]


def _y_section(shape, y):
    """Section at Y = y in (x, z)."""
    return section_local(shape, Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0)))


def _flat_y(shape, sign):
    """{Y: [faces]} of planar faces whose normal is sign * Y."""
    out = {}
    for f in shape.faces():
        if f.geom_type.name == "PLANE" and f.normal_at().Y * sign > 1 - 1e-9:
            out.setdefault(round(f.center().Y, 6), []).append(f)
    return out


def measure(p, part):
    m = {}
    bb = part.bounding_box()
    m["height"] = bb.max.Y - bb.min.Y
    rim = _y_section(part, 1e-4)
    outer = max(_polys(rim), key=lambda g: g.area)
    x0, z0, x1, z1 = outer.exterior.bounds          # local y = -Z: only the extents matter here
    m["length"], m["width"] = x1 - x0, z1 - z0
    inner = max(outer.interiors, key=lambda r: SPolygon(r).area)
    ix0, iz0, ix1, iz1 = SPolygon(inner).bounds
    m["wall_x"], m["wall_z"] = ((x1 - x0) - (ix1 - ix0)) / 2, ((z1 - z0) - (iz1 - iz0)) / 2
    down = _flat_y(part, -1)
    fl = max(down, key=lambda y: sum(f.area for f in down[y]))       # floor underside: the largest
    m["floor"] = bb.max.Y - fl
    # ribs: root thickness just under the floor, across the X rib at x = L/4 and the Z rib at z = W/4
    root = _y_section(part, fl - 1e-3)
    m["rib_t_x"] = root.intersection(LineString([(p.length / 4, -6), (p.length / 4, 6)])).length
    m["rib_t_z"] = root.intersection(LineString([(-6, p.width / 4), (6, p.width / 4)])).length
    # rib tip: the downward face under the X rib at x = L/4
    def under(f, x, z):
        b = f.bounding_box()
        return b.min.X <= x <= b.max.X and b.min.Z <= z <= b.max.Z
    tips = [y for y in down if 1e-3 < y < fl - 1e-6 and any(under(f, p.length / 4, 0.0) for f in down[y])]
    m["rib_h"] = fl - min(tips) if tips else float("nan")
    # bosses: annuli just above their tips
    yt = min((y for y in down if y > 1e-3), default=float("nan"))
    m["boss_tip_y"] = yt
    ring = _y_section(part, yt + 1e-4)
    bosses = [g for g in _polys(ring) if g.interiors and g.area < 200]
    m["bosses"] = []
    for g in bosses:
        c = SPolygon(g.interiors[0])
        ex0, ey0, ex1, ey1 = g.exterior.bounds
        hx0, hy0, hx1, hy1 = c.bounds
        m["bosses"].append(dict(x=c.centroid.x, z=-c.centroid.y, d=ex1 - ex0, hole=hx1 - hx0))
    # hole bottoms: downward planes inside the holes
    m["hole_depth"] = []
    for b in m["bosses"]:
        ys = [y for y, fs in down.items() if any(math.hypot(f.center().X - b["x"], f.center().Z - b["z"]) < 1e-3 for f in fs)]
        m["hole_depth"].append(max(ys) - yt if ys else float("nan"))
    # window: section by the front outer face (tilted by the draft), just inside it
    t = math.tan(math.radians(p.draft_deg))
    nrm = Vector(0, t, -1).normalized()                         # outward: the face leans in going up
    zf = housing.window_face_z(p)
    pl = Plane(origin=Vector(0, p.window_y, zf) - nrm * 1e-4, x_dir=(1, 0, 0), z_dir=nrm)
    wall = section_local(part, pl)
    holes = [SPolygon(r) for g in _polys(wall) for r in g.interiors]
    m["window"] = None
    if holes:
        h = max(holes, key=lambda g: g.area)
        a0, b0, a1, b1 = h.bounds
        w = h.intersection(LineString([(a0 - 1, 0.0), (a1 + 1, 0.0)])).length    # at the window centre height
        dy = pl.y_dir.Y                                         # in-plane height -> Y
        ys = sorted((pl.origin.Y + b0 * dy, pl.origin.Y + b1 * dy))
        m["window"] = dict(w=w, h=ys[1] - ys[0], x=(a0 + a1) / 2, y0=ys[0])
    return m


# ------------------------------------------------------------------ sheet
def _notes(sh, lines, pos):
    x, y = pos
    sh.text((x, y), "注記", 4.0, 0, "left", "top")
    for i, s in enumerate(lines):
        sh.text((x + 2, y - 7 - i * 5.2), f"{i + 1}. {s}", 3.0, 0, "left", "top")
    sh.meta["notes"] = lines


def _general_tol_table(sh, pos):
    x, y = pos
    rows = jis.b0405_table(TOL_CLASS)
    sh.text((x, y), f"普通許容差 JIS B 0405-{TOL_CLASS} (粗級)", 3.2, 0, "left", "bottom")
    cw, rh = 17.0, 4.6
    y -= 1.5
    cols = [(f"{_fmt(a)}以上" if i == 0 else f"{_fmt(a)}を超え", f"{_fmt(b)}以下", f"±{_fmt(t)}")
            for i, (a, b, t) in enumerate(rows)]
    sh.meta["general_tol_table"] = [dict(over=a, incl_lower=i == 0, upto=b, tol=t) for i, (a, b, t) in enumerate(rows)]
    n = len(cols) + 1
    for j in (0, 2, 3):
        sh.line([(x, y - j * rh), (x + cw * n, y - j * rh)], "thin")
    sh.line([(x + cw, y - rh), (x + cw * n, y - rh)], "thin")
    sh.text((x + cw / 2, y - rh), "寸法区分", 2.5, 0, "center", "center")
    sh.text((x + cw / 2, y - 2.5 * rh), "許容差", 2.5, 0, "center", "center")
    for i, col in enumerate(cols, start=1):
        for j, t in enumerate(col):
            sh.text((x + i * cw + cw / 2, y - (j + 0.5) * rh), t, 2.5, 0, "center", "center")
    for i in range(n + 1):
        sh.line([(x + i * cw, y), (x + i * cw, y - 3 * rh)], "thin")


def _scale_text(k):
    return f"{k:g}:1" if k >= 1 else f"1:{1 / k:g}"


def part_sheet(p: housing.HousingParams, part, dwg_no):
    """A3 at 1:1; if the views or their annotations do not fit, A2, then 1:2."""
    from .stem_drawing import _fits
    sh = None
    for size, k in TRIES:
        cand = _sheet(p, part, dwg_no, size, k)
        if cand is None:
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError("housing views do not fit A2 at 1:2")
    return sh


def _sheet(p, part, dwg_no, size, k):
    sh = Sheet(size); D = _Dims(sh)
    m = measure(p, part)
    rs = mo.resin(p)
    D.need("bosses", len(m["bosses"]), 4)
    D.need("window", m["window"] is not None, 1)
    bx = max(b["x"] for b in m["bosses"]) if m["bosses"] else p.boss_pitch_x / 2

    front = View("front", part, (0, 0, 1), (1, 0, 0), (0, 0), k).build()
    top = View("top", part, (0, 1, 0), (1, 0, 0), (0, 0), k).build()
    right = View("A-A", part, (-1, 0, 0), (0, 0, 1), (0, 0), k, section=((bx, 0, 0), (-1, 0, 0)), label="A-A").build()
    if not sh.place_third_angle(front, top, right=right, gap=30, edge=14):
        return None

    mass = part.volume / 1000 * housing.density(p)
    mat, finish = MATERIAL_SPEC.get(p.material, (p.material, ""))
    rows = [("図名", "電装ハウジング (HOUSING)"), ("図番", dwg_no), ("材料", mat), ("質量 (計算)", f"{mass:.1f} g"),
            ("尺度", _scale_text(k)), ("投影法", "第三角法"), ("単位", "mm"),
            ("普通公差", f"JIS B 0405-{TOL_CLASS}"), ("作成", f"agent-cad  {datetime.date.today().isoformat()}")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 120, ytop + 4))
    sh.text((x0 + 4, ytop + 6), "表面: 成形のまま (金型面 #800 磨き相当)", 3.2, 0, "left", "bottom")
    sh.meta["roughness_general"] = "成形のまま"
    sh.meta["finish"] = finish
    sh.meta["part"] = "HOUSING"
    _notes(sh, [f"材料 {p.material}、色 {finish.split('、')[0] or '未指定'}。材質表示 {rs.iso_mark} (JIS K 6899-1) を内面に浮き出し。",
                f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS} による (樹脂の公差は JIS K 7109 を参考に成形者と協議)。",
                f"指示なき抜き勾配: 外面・内面 {_fmt(p.draft_deg)}°、リブ・ボス・穴 {_fmt(p.rib_draft_deg)}° (片側)、窓 {_fmt(p.window_draft_deg)}°。",
                "寸法は勾配の始まる側 (開口端・ボス先端・窓は外面) で示す。",
                "PL: 開口端面。PL段差・バリ 0.1 以下。ゲート: +X 側面下端 (エッジゲート)、残り 0.3 以下。",
                "突出しピン跡は内面のみ、凹凸 0.1 以下。外観面にヒケ・ウエルドラインなきこと。",
                f"指示なき肉厚 {_fmt(m['floor'])}。ねじ穴は呼び径4 樹脂用タッピンねじ用下穴 (先端径を示す)。",
                "図示なき形状は3Dモデル (STEP AP242、同図番) による。"], (16, 108))
    _general_tol_table(sh, (16, 36))

    fs, ts, rsec = front.to_sheet, top.to_sheet, right.to_sheet
    tb, rb = top.bbox(), right.bbox()
    L, W, H = m["length"], m["width"], m["height"]
    # ---- front view: rim at the sheet top (opening up)
    sh.centerline(fs((0, -2, -W / 2)), fs((0, H + 2, -W / 2)), ext=2)
    sh.dim(fs((-p.length / 2, 0, -W / 2)), fs((p.length / 2, 0, -W / 2)), 20, _fmt(L))
    D.rec("length", L, p.length, _fmt(L))
    sh.dim(fs((-p.length / 2, H, -W / 2)), fs((-p.length / 2, 0, -W / 2)), 12, _fmt(H), u=(0, 1))
    D.rec("height", H, p.height, _fmt(H))
    win = m["window"]
    if win:
        wx0, wx1 = win["x"] - win["w"] / 2, win["x"] + win["w"] / 2
        yb, yt_ = win["y0"], win["y0"] + win["h"]
        sh.dim(fs((wx0, yb, -W / 2)), fs((wx1, yb, -W / 2)), yb * k + 10, _fmt(win["w"]))
        D.rec("window_w", win["w"], p.window_w, _fmt(win["w"]))
        sh.dim(fs((wx1, yt_, -W / 2)), fs((wx1, yb, -W / 2)), -8, _fmt(win["h"]), u=(0, 1))
        D.rec("window_h", win["h"], p.window_h, _fmt(win["h"]))
        sh.dim(fs((win["x"], H, -W / 2)), fs((0, H, -W / 2)), -10, _fmt(abs(win["x"])))
        sh.centerline(fs((win["x"], H + 2, -W / 2)), fs((win["x"], yb - 2, -W / 2)), ext=0)
        D.rec("window_x", win["x"], p.window_x, _fmt(abs(win["x"])))
        sh.dim(fs((wx0, yb, -W / 2)), fs((wx0, 0, -W / 2)), 8, _fmt(yb), u=(0, 1))
        D.rec("window_bottom", yb, p.window_y - p.window_h / 2, _fmt(yb))
    # parting line and gate marks
    pl0 = fs((p.length / 2, 0, -W / 2))
    sh.leader(pl0, [pl0 + (12, 10), pl0 + (12, -12)], "PL")
    g0 = fs((p.length / 2, 0.75, -W / 2))
    sh.leader(g0, [g0 + (8, -(H * k) - 14), g0 + (8, -(H * k) - 24)], "ゲート (G)")

    # ---- view into the box: bosses, holes, ribs
    sh.centerline(ts((-L / 2 - 3, 0, 0)), ts((L / 2 + 3, 0, 0)), ext=0)
    sh.centerline(ts((0, 0, -W / 2 - 3)), ts((0, 0, W / 2 + 3)), ext=0)
    for b in m["bosses"]:
        sh.center_mark(ts((b["x"], 0, b["z"])), k * b["d"] / 2, ext=2)
    sh.dim(ts((p.length / 2, 0, W / 2)), ts((p.length / 2, 0, -W / 2)), -12, _fmt(W), u=(0, 1))
    D.rec("width", W, p.width, _fmt(W))
    if len(m["bosses"]) == 4:
        xs = sorted({round(b["x"], 4) for b in m["bosses"]}); zs = sorted({round(b["z"], 4) for b in m["bosses"]})
        px, pz = xs[-1] - xs[0], zs[-1] - zs[0]
        sh.dim(ts((xs[0], 0, zs[-1])), ts((xs[-1], 0, zs[-1])), (W / 2 - zs[-1]) * k + 10, _fmt(px))
        D.rec("boss_pitch_x", px, p.boss_pitch_x, _fmt(px))
        sh.dim(ts((xs[0], 0, zs[-1])), ts((xs[0], 0, zs[0])), (xs[0] + L / 2) * k + 10, _fmt(pz), u=(0, 1))
        D.rec("boss_pitch_z", pz, p.boss_pitch_z, _fmt(pz))
        bd = m["bosses"][0]["d"]; hd = m["bosses"][0]["hole"]
        for i, b in enumerate(m["bosses"]):
            D.rec("boss_d" if i == 0 else f"boss_d#{i + 1}", b["d"], p.boss_d, f"φ{_fmt(bd)}")
            D.rec("hole_d" if i == 0 else f"hole_d#{i + 1}", b["hole"], p.hole_d, f"φ{_fmt(hd)}")
        depth = m["hole_depth"][0]
        for i, dpt in enumerate(m["hole_depth"]):
            D.rec("hole_depth" if i == 0 else f"hole_depth#{i + 1}", dpt, p.hole_depth, _fmt(depth))
        bt = max(m["bosses"], key=lambda b: (b["x"], b["z"]))
        tip = ts((bt["x"] + bd / 2 * 0.7071, 0, bt["z"] + bd / 2 * 0.7071))
        sh.leader(tip, [tip + (22, 16), tip + (22, -18)], "", lines=[f"4×φ{_fmt(bd)} ボス", f"穴 φ{_fmt(hd)} 深さ {_fmt(depth)}"])
    tip = ts((-p.length / 4, 0, -m["rib_t_x"] / 2))
    out = tb[0][0] - tip[0] - 26                                # knee left of the view, clear of the 46
    sh.leader(tip, [tip + (out, -18), tip + (out, -26), tip + (out, 10)], f"リブ厚 {_fmt(m['rib_t_x'])} (根元)", right=False)
    D.rec("rib_t", m["rib_t_x"], p.rib_t, _fmt(m["rib_t_x"]))
    D.rec("rib_t_z", m["rib_t_z"], p.rib_t, _fmt(m["rib_t_x"]))

    # ---- section A-A through the right-hand bosses
    ty = ts((bx, 0, 0))[0]
    sh.section_line((ty, tb[0][1] - 6), (ty, tb[1][1] + 6), "A", look=(1, 0))
    sh.text(((rb[0][0] + rb[1][0]) / 2, rb[1][1] + 12), "A-A", 5, 0, "center", "bottom")
    fy = m["height"] - m["floor"]
    sh.dim(rsec((bx, H, W / 2)), rsec((bx, fy, W / 2)), -14, _fmt(m["floor"]), u=(0, 1))
    D.rec("floor", m["floor"], p.wall, _fmt(m["floor"]))
    D.rec("wall_x", m["wall_x"], p.wall, _fmt(m["floor"]))
    D.rec("wall_z", m["wall_z"], p.wall, _fmt(m["floor"]))
    sh.dim(rsec((bx, fy, 0)), rsec((bx, fy - m["rib_h"], 0)), -(W / 2 * k) - 24, _fmt(m["rib_h"]), u=(0, 1))
    D.rec("rib_h", m["rib_h"], p.rib_h, _fmt(m["rib_h"]))
    sh.dim(rsec((bx, m["boss_tip_y"], -W / 2)), rsec((bx, 0, -W / 2)), 10, _fmt(m["boss_tip_y"]), u=(0, 1))
    D.rec("boss_recess", m["boss_tip_y"], p.boss_recess, _fmt(m["boss_tip_y"]))
    return sh
