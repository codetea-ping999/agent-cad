"""JIS drawing (third-angle) of the drop handlebar.

Views: the front view (from ahead of the bike: width and drop), section A-A on the
horizontal plane through the bar axis above it (the tops cut open: the swaged taper and
the internal butting), and the right side view (reach and drop). The bend geometry is
given the way a tube shop works with it: the vertex (XYZ) table of the tangent-line
intersections and the bending table (feed L, rotation R, angle A, radius); the wall
thickness by zone as a table. Every value is measured from the B-rep (end faces, cones,
tori, sections) and recorded next to its design intent, so check_sheet() catches a
drawing that does not match the model.

Clamp diameter h10 and grip diameter h11 (JIS B 0401), general tolerances JIS B 0405-m;
the bent shape is checked on the inspection gauge to the tolerances in the notes.
"""
import math
import numpy as np
from shapely.geometry import LineString
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Torus, GeomAbs_Plane
from build123d import Plane
from ..drawing.sheet import Sheet, View, A2, A3
from ..drawing.tool import table
from ..drawing.measure import cylinders
from ..standards import jis
from ..process import tube_bending as tb
from . import handlebar as hb
from .housing_drawing import section_local, _polys

TOL_CLASS = "m"
CLAMP_FIT, GRIP_FIT = "h10", "h11"
RA_GENERAL, RA_CLAMP = 3.2, 1.6
TRIES = [(A3, 0.5), (A2, 0.5), (A2, 0.4)]
MATERIAL_SPEC = {
    "A7075-T6": ("A7075-T6 (JIS H 4080 継目無引抜管 A7075TD-O より成形)", "陽極酸化 JIS H 8601 AA10 黒"),
    "A6061-T6": ("A6061-T6 (JIS H 4080 継目無引抜管 A6061TD-O より成形)", "陽極酸化 JIS H 8601 AA10 黒"),
}
SHAPE_TOL = "曲げ形状 (幅・リーチ・ドロップ) ±1.0、曲げ角 ±0.5°、左右対称度 1.0 (検査ゲージによる)"


def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _scale_text(k):
    return f"{k:g}:1" if k >= 1 else f"1:{1 / k:g}"


class _Dims:
    def __init__(self, sheet):
        self.sheet = sheet
        sheet.meta["dims"] = []

    def rec(self, name, measured, intent, text, tol=None):
        self.sheet.meta["dims"].append(dict(name=name, measured=round(float(measured), 4), intent=float(intent),
                                            text=text, tol=tol))

    def need(self, name, found, expected):
        self.sheet.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))

    def ra(self, name, value, required=None):
        self.sheet.meta.setdefault("roughness_local", []).append(
            dict(name=name, drawn=value, required=value if required is None else required))


# ------------------------------------------------------------------ measuring

def _surf(f):
    return BRepAdaptor_Surface(f.wrapped)


def measure(p, bar):
    """Read the bar back from the B-rep."""
    m = {}
    ends = []
    for f in bar.faces():
        if _surf(f).GetType() == GeomAbs_Plane:
            c = f.center()
            ends.append(np.array([c.X, c.Y, c.Z]))
    m["ends"] = sorted(ends, key=lambda q: q[1])
    if len(ends) == 2:
        L, R = m["ends"]
        m["width"] = R[1] - L[1]
        m["drop"] = -(L[2] + R[2]) / 2
        m["end_x"] = (L[0] + R[0]) / 2
    bb = bar.bounding_box()
    m["reach"] = bb.max.X - p.grip_d / 2
    m["bbox"] = bb
    # outer cylinders: clamp and grip diameters, clamp zone from the cylinder along Y
    cyl = [c for c in cylinders(bar) if not c.inner]
    clamp = [c for c in cyl if abs(c.r - p.clamp_d / 2) < 0.5]
    grip = [c for c in cyl if abs(c.r - p.grip_d / 2) < 0.5]
    m["clamp_d"] = 2 * clamp[0].r if clamp else float("nan")
    m["grip_d"] = 2 * grip[0].r if grip else float("nan")
    # outside cones (swaged taper): where each cone meets the clamp and the grip radius
    cones = []
    for f in bar.faces():
        sf = _surf(f)
        if sf.GetType() != GeomAbs_Cone:
            continue
        cn = sf.Cone(); ax = cn.Axis()
        d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        if abs(abs(d[1]) - 1) > 1e-6:
            continue
        q = f.position_at(0.5, 0.5)
        if abs(math.hypot(q.X, q.Z) - hb.radii_at(p, q.Y)[0]) > 0.3:
            continue                                   # the bore
        ya, a = cn.Apex().Y(), abs(cn.SemiAngle())
        sgn = 1 if q.Y > ya else -1
        cones.append(dict(y_clamp=ya + sgn * p.clamp_d / 2 / math.tan(a),
                          y_grip=ya + sgn * p.grip_d / 2 / math.tan(a), side=float(np.sign(q.Y))))
    m["cones"] = cones
    right = [c for c in cones if c["side"] > 0]
    left = [c for c in cones if c["side"] < 0]
    if right and left:
        m["clamp_zone"] = right[0]["y_clamp"] - left[0]["y_clamp"]
        m["taper_len"] = abs(right[0]["y_grip"] - right[0]["y_clamp"])
    # bend tori (axis not along Y): major radii
    tori = []
    for f in bar.faces():
        s = _surf(f)
        if s.GetType() != GeomAbs_Torus:
            continue
        t = s.Torus(); ax = t.Axis(); d, o = ax.Direction(), ax.Location()
        if abs(abs(d.Y()) - 1) < 1e-6 and math.hypot(o.X(), o.Z()) < 1e-6:
            continue                                   # a blend of the taper (revolved about the bar axis)
        tori.append(t.MajorRadius())
    m["tori"] = tori
    # walls: sections normal to the centreline at stations
    sol = hb.solve(p)
    a, b, n, c, d = hb.zones(p)
    segs = {g.name: g for g in hb.path(p, 1, sol)}
    st = {"clamp": ("L0", 0.0), "taper_end": ("L0", taper_end_station(p)), "butt": ("L0", (n + c) / 2),
          "grip": ("L0", (d + sol.L0) / 2), "bend_B2": ("B2", segs["B2"].length / 2),
          "drops": ("L3", sol.L3 / 2)}
    m["walls"] = {}
    for k, (nm, s_) in st.items():
        g = segs[nm]
        q, t = g.point(s_)
        x = (0, 0, 1) if abs(t[2]) < 0.9 else (1, 0, 0)
        sec = section_local(bar, Plane(origin=tuple(q), x_dir=x, z_dir=tuple(t)))
        polys = [g_ for g_ in _polys(sec) if g_.area > 1]
        from shapely.geometry import Point
        ring = min(polys, key=lambda g_: g_.centroid.distance(Point(0, 0))) if polys else None
        if ring is None or not ring.interiors:
            m["walls"][k] = float("nan")
            continue
        ro = math.sqrt(abs(_area(ring.exterior)) / math.pi)
        ri = math.sqrt(abs(_area(ring.interiors[0])) / math.pi)
        m["walls"][k] = ro - ri
    return m


def _area(ring):
    from shapely.geometry import Polygon
    return Polygon(ring).area


def taper_end_station(p):
    """|y| just past the outside cone's end blend (the outside is back on grip_d there)."""
    a, b, n, c, d = hb.zones(p)
    turn = math.atan((p.clamp_d - p.grip_d) / 2 / p.taper_len)
    return b + p.taper_blend_r * math.tan(turn / 2) + 0.5


def wall_intents(p):
    yt = taper_end_station(p)
    return {"clamp": p.wall_clamp, "taper_end": p.grip_d / 2 - hb.radii_at(p, yt)[1], "butt": p.wall_butt,
            "grip": p.wall_grip, "bend_B2": p.wall_grip, "drops": p.wall_grip}


WALL_LABEL = {"clamp": "クランプ部", "taper_end": "テーパ終端", "butt": "バテッド部",
              "grip": "トップ外側", "bend_B2": "曲げ部 (公称)", "drops": "ドロップ部"}


# ------------------------------------------------------------------ sheet

def part_sheet(p: hb.HandlebarParams, bar, dwg_no):
    from .stem_drawing import _fits
    sh = None
    for size, k in TRIES:
        cand = _sheet(p, bar, dwg_no, size, k)
        if cand is None:
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError("handlebar views do not fit A2 at 1:2.5")
    return sh


def _notes(sh, lines, pos):
    x, y = pos
    sh.text((x, y), "注記", 4.0, 0, "left", "top")
    for i, s in enumerate(lines):
        sh.text((x + 2, y - 7 - i * 5.2), f"{i + 1}. {s}", 3.0, 0, "left", "top")
    sh.meta["notes"] = lines
    return y - 7 - len(lines) * 5.2


def _sheet(p, bar, dwg_no, size, k):
    sh = Sheet(size); D = _Dims(sh)
    m = measure(p, bar)
    sol = hb.solve(p)
    D.need("drop_ends", len(m["ends"]), 2)
    D.need("taper_cones", len(m["cones"]), 2)       # one outside cone per side
    D.need("bend_tori", len(m["tori"]), 12)

    front = View("front", bar, (-1, 0, 0), (0, 1, 0), (0, 0), k).build()
    top = View("A-A", bar, (0, 0, -1), (0, 1, 0), (0, 0), k, section=((0, 0, 0), (0, 0, -1)), label="A-A").build()
    right = View("right", bar, (0, -1, 0), (-1, 0, 0), (0, 0), k).build()
    if not sh.place_third_angle(front, top, right=right, gap=34, edge=14):
        return None
    # removed section B-B of the clamp zone at the centre, 1:1, above the right view
    from build123d import Box, Pos
    ring = bar & Pos(0, -0.25, 0) * Box(80, 0.5, 80)
    kb = 1.0
    bview = View("B-B", ring, (0, -1, 0), (-1, 0, 0), (0, 0), kb).build()
    rbx, tbx = right.bbox(), top.bbox()
    bbx = bview.bbox()
    cx, cy = (rbx[0][0] + rbx[1][0]) / 2, (tbx[0][1] + tbx[1][1]) / 2
    bview.shift(cx - (bbx[0][0] + bbx[1][0]) / 2, cy - (bbx[0][1] + bbx[1][1]) / 2)
    if bview.bbox()[1][0] > size[0] - 30 or bview.bbox()[0][0] < top.bbox()[1][0] + 20:
        return None
    sh.add_view(bview, built=True)

    mass = bar.volume / 1000 * hb.density(p)
    mat, finish = MATERIAL_SPEC.get(p.material, (p.material, ""))
    rows = [("図名", "ハンドルバー (ドロップ)"), ("図番", dwg_no), ("材料", mat), ("質量 (計算)", f"{mass:.1f} g"),
            ("尺度", _scale_text(k)), ("投影法", "第三角法"), ("単位", "mm"), ("普通公差", f"JIS B 0405-{TOL_CLASS}")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 140, ytop + 4))
    sh.roughness((x0 + 6, ytop + 4), RA_GENERAL)
    sh.text((x0 + 40, ytop + 5.5), f"( Ra {RA_CLAMP} クランプ部 )", 3.2, 0, "left", "bottom")
    sh.meta["roughness_general"] = f"Ra {RA_GENERAL}"
    sh.meta["finish"] = finish
    sh.meta["part"] = "HANDLEBAR"

    fs, ts, rs = front.to_sheet, top.to_sheet, right.to_sheet
    Rc, Rg = p.clamp_d / 2, p.grip_d / 2
    yE = sol.L0 + p.r_shoulder                       # drop-end centre |y| (flare 0)
    if len(m["ends"]) == 2:
        yE = m["ends"][1][1]
    zE = -m.get("drop", p.drop)
    # ---- front view: width, drop, clamp and grip diameters, clamp zone and taper
    sh.centerline(fs((0, 0, Rc + 6)), fs((0, 0, zE - Rg - 6)), ext=0)
    if "width" in m:
        sh.dim(fs((0, -yE, zE)), fs((0, yE, zE)), -(Rg * k + 14), _fmt(m["width"]))
        D.rec("width", m["width"], p.width, _fmt(m["width"]))
        sh.dim(fs((0, yE, 0)), fs((0, yE, zE)), -(Rg * k + 16), _fmt(m["drop"]), u=(0, -1))
        D.rec("drop", m["drop"], p.drop, _fmt(m["drop"]))
        for q in m["ends"]:
            sh.center_mark(fs(q), Rg * k, ext=2)
    if "clamp_zone" in m:
        ya = m["clamp_zone"] / 2
        sh.dim(fs((0, -ya, Rc)), fs((0, ya, Rc)), 14, _fmt(m["clamp_zone"]))
        D.rec("clamp_zone", m["clamp_zone"], p.clamp_zone_w, _fmt(m["clamp_zone"]))
        yb = ya + m["taper_len"]
        sh.dim(fs((0, ya, Rc)), fs((0, yb, Rg)), 14 + (Rc - Rg) * k, _fmt(m["taper_len"]), u=(1, 0))
        D.rec("taper_len", m["taper_len"], p.taper_len, _fmt(m["taper_len"]))
    # local roughness on the clamp zone
    tip = fs((0, p.clamp_zone_w / 4, Rc))
    sh.roughness_leader(tip, tip + (14, 22), RA_CLAMP)
    D.ra("clamp_zone", RA_CLAMP)
    # section line A-A on the bar axis
    fb = front.bbox()
    ya_ = fs((0, 0, 0))[1]
    sh.section_line((fb[0][0] - 14, ya_), (fb[1][0] + 14, ya_), "A", look=(0, 1))

    # ---- section A-A (seen from above): butting and taper, shoulder bend radius
    tb_ = top.bbox()
    sh.text(((tb_[0][0] + tb_[1][0]) / 2, tb_[1][1] + 8), "A-A", 5, 0, "center", "bottom")
    sh.centerline(ts((Rc + 8, 0, 0)), ts((-p.reach - Rg - 8, 0, 0)), ext=0)
    r1 = [r for r in m["tori"] if abs(r - p.r_shoulder) < 1e-3]
    if r1:
        g = hb.path(p, -1, sol)[1]                    # left B1, in the section's kept half (z <= 0)
        c = np.array(g.p0) + g.R * np.array(g.n0)
        q = c + (np.array(g.point(g.length / 2)[0]) - c) * (1 + Rg / g.R)
        tip = ts(q)
        sh.leader(tip, [tip + (-16, 14), tip + (16, 14)], f"R{_fmt(r1[0])} (管中心)")
        D.rec("r_shoulder", r1[0], p.r_shoulder, f"R{_fmt(r1[0])}")
    tip = ts((Rc - p.wall_clamp / 2, 4.0, 0))
    sh.leader(tip, [tip + (10, 20), tip + (-10, 20)], "肉厚は肉厚表による")

    # ---- right side view: reach, drop-end length, hood and drop bend radii
    rb = right.bbox()
    segs = {g.name: g for g in hb.path(p, 1, sol)}
    sh.centerline(rs((Rc + 6, yE, 0)), rs((-Rc - 6, yE, 0)), ext=0)
    sh.centerline(rs((0, yE, Rc + 6)), rs((0, yE, -Rc - 6)), ext=0)
    sh.dim(rs((0, yE, 0)), rs((m["reach"], yE, 0)), -(rb[1][1] - rs((0, yE, 0))[1]) - 12, _fmt(m["reach"]), u=(-1, 0))
    D.rec("reach", m["reach"], p.reach, _fmt(m["reach"]))
    b3 = segs["B3"]
    xb = b3.end()[0][0]
    if len(m["ends"]) == 2:
        L3 = xb - m["end_x"]
        sh.dim(rs((xb, yE, zE)), rs((m["end_x"], yE, zE)), -(Rg * k + 10), _fmt(L3), u=(-1, 0))
        D.rec("drop_len", L3, p.drop_len, _fmt(L3))
    for nm, intent, ang in (("B2", p.r_hood, 45), ("B3", p.r_drop, -45)):
        g = segs[nm]
        rr = [r for r in m["tori"] if abs(r - intent) < 1e-3]
        if not rr:
            continue
        c = np.array(g.p0) + g.R * np.array(g.n0)
        q = c + (np.array(g.point(g.length / 2)[0]) - c) * (1 + Rg / g.R)
        tip = rs(q)
        dy = 12 if ang > 0 else -12
        sh.leader(tip, [tip + (-18, dy), tip + (18, dy), tip + (-18, 2.5 * dy), tip + (18, 2.5 * dy)],
                  f"R{_fmt(rr[0])} (管中心)")
        D.rec(f"r_{'hood' if nm == 'B2' else 'drop'}", rr[0], intent, f"R{_fmt(rr[0])}")

    # grip diameter on the right drop end, seen end-on
    if len(m["ends"]) == 2:
        gup, glo = (jis.fmt_dev(x) for x in jis.fit_deviations(round(m["grip_d"], 3), GRIP_FIT))
        gtxt = f"φ{_fmt(m['grip_d'])} {GRIP_FIT}"
        sh.diameter_dim(fs(m["ends"][1]), Rg * k, [-140, -110, -40, 40], gtxt, tol=(gup, glo), outside=14)
        D.rec("grip_d", m["grip_d"], p.grip_d, gtxt, tol=(gup, glo))
    # ---- section B-B: clamp diameter and wall
    bs = bview.to_sheet
    bb_ = bview.bbox()
    sh.text(((bb_[0][0] + bb_[1][0]) / 2, bb_[1][1] + 6), "B-B", 5, 0, "center", "bottom")
    up, lo = (jis.fmt_dev(x) for x in jis.fit_deviations(round(m["clamp_d"], 3), CLAMP_FIT))
    txt = f"φ{_fmt(m['clamp_d'])} {CLAMP_FIT}"
    sh.diameter_dim(bs((0, -0.25, 0)), Rc * kb, [30, 150, -30], txt, tol=(up, lo), outside=12)
    D.rec("clamp_d", m["clamp_d"], p.clamp_d, txt, tol=(up, lo))
    tip = bs((-(Rc - p.wall_clamp / 2) * 0.7071, -0.25, -(Rc - p.wall_clamp / 2) * 0.7071))
    sh.leader(tip, [tip + (-14, -14), tip + (14, -14)], f"肉厚 {_fmt(m['walls']['clamp'])}")
    xb_ = fs((0, 0, 0))[0]
    sh.section_line((xb_, fs((0, 0, -Rc))[1] - 5), (xb_, fs((0, 0, Rc))[1] + 5), "B", look=(1, 0))

    # ---- notes, tables
    sw = tb.swaging(p)
    notes = [f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS} による。{SHAPE_TOL}。",
             f"材料 {p.material}: O材の引抜管を両端スエージ ({sw['passes']} パス) 後に曲げ、溶体化・焼入れ・時効 (T6)。",
             f"表面処理 {finish}。処理前ショットブラスト。クランプ部は処理後 Ra {RA_CLAMP}。",
             "管内面はスエージ・引抜きのまま (加工しない)。肉厚は肉厚表および3Dモデルによる。",
             "両端面は直角に切断、内外面バリ取り C0.3。端部の曲げつかみ代は切除する。",
             "クランプ部中央にセンターマーク、両側に ±10° の角度目盛 (2° ピッチ) をレーザー刻印。",
             "曲げ部の外側肉厚は 0.85 以上、偏平率 (長径-短径)/外径 5 % 以下。",
             "強度は ISO 4210-5 (競技用) のハンドルバー試験に合格すること。",
             "図示なき形状は3Dモデル (STEP AP242、同図番) による。"]
    W, H = sh.size
    ybot = _notes(sh, notes, (16, 108))
    # wall table (measured)
    wi = wall_intents(p)
    wrows = []
    for key, lab in WALL_LABEL.items():
        v = m["walls"].get(key, float("nan"))
        wrows.append([lab, _fmt(v)])
        D.rec(f"wall.{key}", v, wi[key], _fmt(v))
    sh.text((16, ybot - 4), "肉厚表 (公称)", 3.2, 0, "left", "bottom")
    table(sh, (16, ybot - 5), [("部位", 46), ("肉厚", 16)], wrows)
    sh.meta["wall_table"] = wrows
    # vertex (XYZ) and bending tables, right half (the left half mirrors in Y)
    rows_lra, verts = tb.lra_table(p, 1)
    xt = 16 + 46 + 16 + 10
    vrows = [[f"P{i}", _fmt(v[0]), _fmt(v[1]), _fmt(v[2])] for i, v in enumerate(verts)]
    sh.text((xt, ybot - 4), "交点座標表 (右半分、左は Y 対称)", 3.2, 0, "left", "bottom")
    table(sh, (xt, ybot - 5), [("点", 10), ("X", 16), ("Y", 16), ("Z", 16)], vrows)
    sh.meta["vertex_table"] = vrows
    xt2 = xt + 58 + 10
    brows = [[r["bend"], _fmt(r["L"]), _fmt(r.get("R_rot", 0)), _fmt(r.get("A", 0)), _fmt(r.get("radius", 0))]
             for r in rows_lra if r["bend"] != "end"]
    brows.append(["端部", _fmt(rows_lra[-1]["L"]), "", "", ""])
    sh.text((xt2, ybot - 4), "曲げ表 (中心から、右半分)", 3.2, 0, "left", "bottom")
    table(sh, (xt2, ybot - 5), [("曲げ", 12), ("L", 14), ("R°", 12), ("A°", 12), ("半径", 14)], brows)
    sh.meta["bend_table"] = brows
    return sh
