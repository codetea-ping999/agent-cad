"""JIS drawing (third-angle) of the finished die-cast bracket.

Views: the plan seen from the boss side (top), section A-A through the hub (front)
and section B-B through the right-hand bosses (right). Every value is measured from the
finished B-rep and recorded next to the design intent from BracketParams, so
check_sheet() catches any disagreement. Drafted walls are dimensioned at their root,
on the plate's top face (bosses, hub, ribs) or on the parting face (plate outline,
cored holes), as the notes say; the intents follow the draft from the parameters.

General tolerances JIS B 0405-m for the machined features, JIS B 0403 CT6 for the
as-cast ones (agreed with the foundry); bore H7 (JIS B 0401), perpendicular to the
back face (datum A).
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon as SPolygon, LineString, Point
from ..drawing.sheet import Sheet, View, A2, A3
from ..drawing.measure import find_cyl
from ..standards import jis
from .housing_drawing import _polys, _flat_y, _y_section
from . import bracket

TOL_CLASS = "m"
RA_FACED, RA_BORE = 6.3, 1.6
BORE_FIT = "H7"
TRIES = [(A3, 1.0), (A2, 1.0), (A3, 0.5), (A2, 0.5)]
MATERIAL_SPEC = {"ADC12": ("ADC12 (JIS H 5302 アルミニウム合金ダイカスト)", "鋳放し、加工面は素地")}


def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _scale_text(k):
    return f"{k:g}:1" if k >= 1 else f"1:{1 / k:g}"


def at_root(d, y, deg):
    """Size of a wall drafted inward going up (`d` at Y = 0) at height y."""
    return d - 2 * y * math.tan(math.radians(deg))


def _ray(poly, c, u, far=200.0):
    """Distance from c along u to the first boundary of poly (c inside)."""
    g = poly.boundary.intersection(LineString([c, (c[0] + u[0] * far, c[1] + u[1] * far)]))
    pts = [np.array(q.coords[0]) for q in getattr(g, "geoms", [g]) if not q.is_empty]
    return min(np.linalg.norm(q - np.array(c)) for q in pts) if pts else float("nan")


def _corner_r(poly, L, W):
    """Least-squares circle through the outline points of the +X+Z corner arc."""
    P = np.array(poly.exterior.coords)
    x0, z0 = poly.bounds[2], poly.bounds[3]
    m = (P[:, 0] > x0 - 25) & (P[:, 1] > z0 - 25) & (P[:, 0] < x0 - 0.05) & (P[:, 1] < z0 - 0.05)
    Q = P[m]
    if len(Q) < 3:
        return float("nan")
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)[0]
    return float(math.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))


def measure(p, fin):
    m = {}
    base = _y_section(fin, 1e-3)
    plate = max(_polys(base), key=lambda g: g.area)
    x0, z0, x1, z1 = plate.bounds
    m["L"], m["W"] = x1 - x0, z1 - z0
    m["R"] = _corner_r(plate, m["L"], m["W"])
    holes = [SPolygon(r) for r in plate.interiors]
    m["holes"] = sorted([(h.centroid.x, h.centroid.y, (h.bounds[2] - h.bounds[0])) for h in holes if h.area < 150])
    root = _y_section(fin, p.plate_t + 1e-3)
    body = max(_polys(root), key=lambda g: g.area)
    m["boss_d"] = []
    for x, z, _ in m["holes"]:
        u = (1.0 if x > 0 else -1.0, 0.0)
        m["boss_d"].append(2 * _ray(body, (x + u[0] * 6, z), u) + 2 * 6)
    m["hub_d"] = 2 * _ray(body, (p.bore_d / 2 + 1, 0.0), (1.0, 0.0)) + 2 * (p.bore_d / 2 + 1)
    # rib root width across the middle of the +X+Z rib
    bx, bz = p.boss_pitch_x / 2, p.boss_pitch_z / 2
    mid, n = np.array([bx / 2, bz / 2]), np.array([-bz, bx]) / math.hypot(bx, bz)
    m["rib_t"] = body.intersection(LineString([mid - 6 * n, mid + 6 * n])).length
    up = _flat_y(fin, +1)
    ys = sorted(up)
    m["plate_t"] = min(ys)
    m["tops"] = ys
    def top_at(x, z):
        return max((y for y, fs in up.items() for f in fs
                    if f.bounding_box().min.X - 1e-6 <= x <= f.bounding_box().max.X + 1e-6
                    and f.bounding_box().min.Z - 1e-6 <= z <= f.bounding_box().max.Z + 1e-6), default=float("nan"))
    m["hub_h"] = top_at(p.bore_d / 2 + 2, 0.0)
    m["boss_h"] = [top_at(x + np.sign(x) * (d / 2 + 2), z) for x, z, d in m["holes"]]
    m["rib_h"] = top_at(*mid)
    bore = find_cyl(fin, (0, 1, 0), r=p.bore_d / 2, inner=True, rtol=0.2)
    m["bore"] = 2 * bore[0].r if bore else float("nan")
    return m


class _D:
    def __init__(self, sh):
        self.sh = sh; sh.meta["dims"] = []

    def rec(self, name, measured, intent, text, tol=None):
        self.sh.meta["dims"].append(dict(name=name, measured=round(float(measured), 4), intent=float(intent), text=text, tol=tol))

    def need(self, name, found, expected):
        self.sh.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))

    def ra(self, name, value):
        self.sh.meta.setdefault("roughness_local", []).append(dict(name=name, drawn=value, required=value))


def _notes(sh, lines, pos):
    x, y = pos
    sh.text((x, y), "注記", 4.0, 0, "left", "top")
    for i, s in enumerate(lines):
        sh.text((x + 2, y - 7 - i * 5.2), f"{i + 1}. {s}", 3.0, 0, "left", "top")
    sh.meta["notes"] = lines


def part_sheet(p: bracket.BracketParams, fin, dwg_no):
    from .stem_drawing import _fits
    sh = None
    for size, k in TRIES:
        cand = _sheet(p, fin, dwg_no, size, k)
        if cand is None:
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError("bracket views do not fit A2 at 1:2")
    return sh


def _sheet(p, fin, dwg_no, size, k):
    sh = Sheet(size); D = _D(sh)
    m = measure(p, fin)
    D.need("bolt_holes", len(m["holes"]), 4)
    D.need("bore", m["bore"] == m["bore"], 1)
    bx = p.boss_pitch_x / 2
    sm = dict(smooth_deg=6.0)      # the drafted bosses, hub and holes are fine polygon prisms
    front = View("A-A", fin, (0, 0, -1), (1, 0, 0), (0, 0), k, section=((0, 0, 0), (0, 0, -1)), label="A-A", **sm).build()
    top = View("top", fin, (0, -1, 0), (1, 0, 0), (0, 0), k, **sm).build()
    right = View("B-B", fin, (-1, 0, 0), (0, 0, -1), (0, 0), k, section=((bx, 0, 0), (-1, 0, 0)), label="B-B", **sm).build()
    if not sh.place_third_angle(front, top, right=right, gap=28, edge=12):
        return None
    mass = fin.volume / 1000 * bracket.density(p)
    mat, finish = MATERIAL_SPEC.get(p.material, (p.material, ""))
    rows = [("図名", "ブラケット (BRACKET)"), ("図番", dwg_no), ("材料", mat), ("質量 (計算)", f"{mass:.1f} g"),
            ("尺度", _scale_text(k)), ("投影法", "第三角法"), ("単位", "mm"),
            ("普通公差", f"JIS B 0405-{TOL_CLASS} / 鋳放し JIS B 0403 CT6")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 140, ytop + 4))
    sh.text((x0 + 4, ytop + 6), f"指示なき面: 鋳放し  ( √Ra {RA_FACED}  √Ra {RA_BORE} )", 3.2, 0, "left", "bottom")
    sh.meta.update(roughness_general="鋳放し", finish=finish, part="BRACKET")
    s = p.stock_mm
    _notes(sh, [f"材料 {p.material}、高圧ダイカスト。湯口・オーバーフロー・バリは除去し、残り 0.3 以下。",
                f"指示なき寸法許容差は、加工部 JIS B 0405-{TOL_CLASS}、鋳放し部 JIS B 0403 CT6 による。",
                f"指示なき抜き勾配 {_fmt(p.draft_deg)}° (外面・リブ)、鋳抜き穴 {_fmt(p.core_draft_deg)}°。",
                "ボス・ハブ・リブの寸法は根元 (板上面)、板外形と鋳抜き穴は裏面 (見切り面) で示す。",
                "PL: 裏面。押出しピン跡は裏面以外の平面、凹凸 0.2 以下。",
                f"ボス・ハブ上面は面削り、ハブ穴はリーマ仕上げ (鋳造時の取り代 {_fmt(s)})。",
                "鋳巣: 加工面に φ0.5 を超える巣なきこと。X線検査は協議による。",
                "図示なき形状は3Dモデル (STEP AP242、同図番) による。"], (16, 112))
    _tol_table(sh, (16, 40))

    fs, ts, rs = front.to_sheet, top.to_sheet, right.to_sheet
    L, W = m["L"], m["W"]
    # ---- plan
    sh.centerline(ts((-L / 2 - 3, 0, 0)), ts((L / 2 + 3, 0, 0)), ext=0)
    sh.centerline(ts((0, 0, -W / 2 - 3)), ts((0, 0, W / 2 + 3)), ext=0)
    yt = ts((0, 0, -W / 2))[1]
    sh.dim(ts((-L / 2, 0, -W / 2)), ts((L / 2, 0, -W / 2)), 13, _fmt(L))
    D.rec("plate_l", L, p.plate_l, _fmt(L))
    sh.dim(ts((L / 2, 0, W / 2)), ts((L / 2, 0, -W / 2)), -12, _fmt(W), u=(0, 1))
    D.rec("plate_w", W, p.plate_w, _fmt(W))
    hx = sorted({round(x, 4) for x, _, _ in m["holes"]}); hz = sorted({round(z, 4) for _, z, _ in m["holes"]})
    if len(m["holes"]) == 4:
        px, pz = hx[-1] - hx[0], hz[-1] - hz[0]
        zt = -pz / 2 if ts((0, 0, -pz / 2))[1] > ts((0, 0, pz / 2))[1] else pz / 2   # the boss row nearer the sheet top
        sh.dim(ts((hx[0], 0, zt)), ts((hx[-1], 0, zt)), yt - ts((0, 0, zt))[1] + 5, _fmt(px))
        D.rec("boss_pitch_x", px, p.boss_pitch_x, _fmt(px))
        a, b = sorted((ts((hx[0], 0, hz[0])), ts((hx[0], 0, hz[-1]))), key=lambda q: q[1])
        sh.dim(a, b, (hx[0] + L / 2) * k + 10, _fmt(pz), u=(0, 1))
        D.rec("boss_pitch_z", pz, p.boss_pitch_z, _fmt(pz))
        for x, z, _ in m["holes"]:
            sh.center_mark(ts((x, 0, z)), k * p.boss_d / 2, ext=2)
        hd = m["holes"][0][2]
        hole0 = p.hole_d + 2 * math.tan(math.radians(p.core_draft_deg))   # the core starts 1 mm below the back face
        for i, (_, _, d) in enumerate(m["holes"]):
            D.rec("hole_d" if i == 0 else f"hole_d#{i + 1}", d, hole0, f"φ{_fmt(hd)}")
        bd = m["boss_d"][0]
        for i, d in enumerate(m["boss_d"]):
            D.rec("boss_d" if i == 0 else f"boss_d#{i + 1}", d, at_root(p.boss_d, p.plate_t, p.draft_deg), f"φ{_fmt(bd)}")
        x, z = hx[-1], hz[-1] if ts((0, 0, hz[-1]))[1] < ts((0, 0, hz[0]))[1] else hz[0]   # lower right boss
        tip = ts((x + p.hole_d / 2 * 0.7071, 0, z)) + np.array([0, -k * p.hole_d / 2 * 0.7071])
        sh.leader(tip, [tip + (26, -16), tip + (26, -26), tip + (30, 10)], "",
                  lines=[f"4×φ{_fmt(hd)} 鋳抜き (裏面)", f"ボス 4×φ{_fmt(bd)} (根元)"])
    R = m["R"]
    ca = ts((L / 2 - R + R * 0.7071, 0, -(W / 2 - R + R * 0.7071)))
    sh.leader(ca, [ca + (14, 12), ca + (14, -12)], f"4×R{_fmt(R)}")
    D.rec("corner_r", R, p.corner_r, f"R{_fmt(R)}")
    hub = m["hub_d"]
    D.rec("hub_d", hub, at_root(p.hub_d, p.plate_t, p.draft_deg), f"φ{_fmt(hub)}")
    hc = ts((-hub / 2, 0, 0))                     # in the wedge between the two left ribs
    sh.leader(hc, [hc + (-6, 4), hc + (-6, -4)], f"φ{_fmt(hub)} (根元)", right=False)
    mid = np.array([bx / 2, p.boss_pitch_z / 4])
    rt = ts((mid[0], 0, mid[1]))
    knee = ts((bx * 0.62, 0, 0))                  # in the wedge between the two right ribs
    sh.leader(rt, [knee + (0, 3), knee + (0, -3)], f"リブ厚 {_fmt(m['rib_t'])}", right=True)
    D.rec("rib_t", m["rib_t"], at_root(p.rib_t, p.plate_t, p.draft_deg), _fmt(m["rib_t"]))

    # ---- section A-A through the hub
    xs = ts((0, 0, 0))[0]
    tb = top.bbox()
    sh.section_line((tb[0][0] - 30, ts((0, 0, 0))[1]), (tb[1][0] + 26, ts((0, 0, 0))[1]), "A", look=(0, -1))
    fb = front.bbox()
    sh.text((fb[1][0] - 12, fb[0][1] - 8), "A-A", 5, 0, "center", "top")
    sh.dim(fs((-L / 2, m["plate_t"], 0)), fs((-L / 2, 0, 0)), 10, _fmt(m["plate_t"]), u=(0, 1))
    D.rec("plate_t", m["plate_t"], p.plate_t, _fmt(m["plate_t"]))
    rx = -(bx / 2)
    sh.dim(fs((rx, m["rib_h"], 0)), fs((rx, 0, 0)), (L / 2 + rx) * k + 22, _fmt(m["rib_h"]), u=(0, 1))
    D.rec("rib_h", m["rib_h"], p.rib_h, _fmt(m["rib_h"]))
    hr = hub / 2
    sh.dim(fs((L / 2, m["hub_h"], 0)), fs((L / 2, 0, 0)), -24, _fmt(m["hub_h"]), u=(0, 1))
    sh.line([fs((hr, m["hub_h"], 0)) + (1, 0), fs((L / 2, m["hub_h"], 0)) + (26, 0)], "thin")
    D.rec("hub_h", m["hub_h"], p.hub_h, _fmt(m["hub_h"]))
    up, lo = jis.fit_deviations(round(m["bore"], 3), BORE_FIT)
    tol = (f"{up:+.3f}", "0" if lo == 0 else f"{lo:+.3f}")
    b = m["bore"]
    sh.dim(fs((-b / 2, m["hub_h"], 0)), fs((b / 2, m["hub_h"], 0)), 12, f"φ{_fmt(b)} {BORE_FIT}", tol=tol)
    D.rec("bore_d", b, p.bore_d, f"φ{_fmt(b)} {BORE_FIT}", tol=tol)
    sh.datum(fs((-L / 4, 0, 0)), (0, -1), "A", length=6)
    sh.fcf(fs((hr, m["hub_h"], 0)) + (12, 6), ["perpendicularity", "φ0.05", "A"], attach=fs((b / 2, m["hub_h"] * 0.6, 0)))
    sh.roughness_leader(fs((-b / 2, m["hub_h"] * 0.5, 0)), fs((-b / 2, m["hub_h"] * 0.5, 0)) + (-46, 6), RA_BORE)
    D.ra("bore", RA_BORE)
    sh.roughness_leader(fs((-hr * 0.75, m["hub_h"], 0)), fs((-hr * 0.75, m["hub_h"], 0)) + (-24, 10), RA_FACED)
    D.ra("hub_top", RA_FACED)

    # ---- section B-B through the right-hand bosses
    ty = ts((bx, 0, 0))[0]
    sh.section_line((ty, tb[0][1] - 6), (ty, tb[1][1] + 6), "B", look=(1, 0))
    rb = right.bbox()
    sh.text(((rb[0][0] + rb[1][0]) / 2, rb[0][1] - 8), "B-B", 5, 0, "center", "top")
    bh = m["boss_h"][0]
    for i, h in enumerate(m["boss_h"]):
        D.rec("boss_h" if i == 0 else f"boss_h#{i + 1}", h, p.boss_h, _fmt(bh))
    zr = p.boss_pitch_z / 2
    zz = zr if rs((bx, bh, zr))[0] > rs((bx, bh, -zr))[0] else -zr
    sh.dim(rs((bx, bh, zz)), rs((bx, 0, zz)), -(W / 2 - zr) * k - 12, _fmt(bh), u=(0, 1))
    tip = rs((bx, bh, -zz))
    sh.roughness_leader(tip, tip + (-18, 12), RA_FACED)
    D.ra("boss_top", RA_FACED)
    return sh


def _tol_table(sh, pos):
    x, y = pos
    rows = jis.b0405_table(TOL_CLASS)
    sh.text((x, y), f"普通許容差 JIS B 0405-{TOL_CLASS} (加工部)", 3.2, 0, "left", "bottom")
    cw, rh = 17.0, 4.6
    y -= 1.5
    cols = [(f"{_fmt(a)}以上" if i == 0 else f"{_fmt(a)}を超え", f"{_fmt(b)}以下", f"±{_fmt(t)}")
            for i, (a, b, t) in enumerate(rows)]
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
