"""JIS drawings (A3, third-angle, 1:1) of the stem body and faceplate.

Every dimension value is measured from the B-rep (agent_cad.drawing.measure)
and recorded in sheet.meta["dims"] next to the design intent from the
parameters, so verify_drawing() can catch model/drawing disagreement.
"""
import math, datetime, hashlib, json
import numpy as np
from build123d import Vector
from ..drawing.sheet import Sheet, View, A2, A3, text_box
from ..drawing.measure import find_cyl, planes
from ..standards import jis
from . import stem

MATERIAL_SPEC = {  # material -> (drawing designation, finish)
    "A6061-T6": ("A6061FD-T6 (JIS H 4140 鍛造品)", "硫酸アルマイト 黒 (JIS H 8601 AA10)"),
    "A7075-T6": ("A7075FD-T6 (JIS H 4140 鍛造品)", "硫酸アルマイト 黒 (JIS H 8601 AA10)"),
    "A2014-T6": ("A2014FD-T6 (JIS H 4140 鍛造品)", "硫酸アルマイト 黒 (JIS H 8601 AA10)"),
    "Ti-6Al-4V": ("TAB6400H (JIS H 4650)", "なし (バフ仕上げ)"),
}
BORE_FIT = "F8"          # steerer and bar bores: clearance for split clamps (JIS B 0401)
RA_BORE, RA_FACE, RA_GENERAL = "1.6", "1.6", "3.2"
TOL_CLASS, GEO_CLASS = "m", "K"


def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _fit_text(d, code):
    up, lo = jis.fit_deviations(d, code)
    return f"φ{_fmt(d)} {code}", (f"{up:+.3f}", f"{lo:+.3f}" if lo else "0")


class _Dims:
    def __init__(self, sheet):
        self.sheet = sheet; sheet.meta["dims"] = []

    def rec(self, name, measured, intent, text, tol=None):
        self.sheet.meta["dims"].append(dict(name=name, measured=round(float(measured), 4),
                                            intent=float(intent), text=text, tol=tol))

    def need(self, name, found, expected):
        """Required feature: check_sheet fails if the model does not have it."""
        self.sheet.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))


def _notes(sh, lines, pos):
    x, y = pos
    sh.text((x, y), "注記", 4.0, 0, "left", "top")
    for i, s in enumerate(lines):
        sh.text((x + 2, y - 7 - i * 5.6), f"{i + 1}. {s}", 3.2, 0, "left", "top")
    sh.meta["notes"] = lines


def _general_tol_table(sh, pos):
    x, y = pos
    rows = jis.b0405_table(TOL_CLASS)
    sh.text((x, y), f"普通許容差 JIS B 0405-{TOL_CLASS} (中級)", 3.2, 0, "left", "bottom")
    cw, rh = 17.0, 5.5
    y -= 1.5
    cols = [("寸法区分", "許容差")] + [(f"{_fmt(a)}〜{_fmt(b)}", f"±{_fmt(t)}") for a, b, t in rows]
    for j in range(3):
        sh.line([(x, y - j * rh), (x + cw * len(cols), y - j * rh)], "thin")
    for i, (a, b) in enumerate(cols):
        sh.line([(x + i * cw, y), (x + i * cw, y - 2 * rh)], "thin")
        sh.text((x + i * cw + cw / 2, y - rh / 2), a, 2.5, 0, "center", "center")
        sh.text((x + i * cw + cw / 2, y - 1.5 * rh), b, 2.5, 0, "center", "center")
    sh.line([(x + len(cols) * cw, y), (x + len(cols) * cw, y - 2 * rh)], "thin")


def _title(sh, p, part_name, dwg_no, mass_g, scale="1:1"):
    mat, _ = MATERIAL_SPEC.get(p.material, (p.material, ""))
    rows = [("図名", part_name), ("図番", dwg_no), ("材料", mat), ("質量 (計算)", f"{mass_g:.1f} g"),
            ("尺度", scale), ("投影法", "第三角法"), ("単位", "mm"),
            ("普通公差", f"JIS B 0405-{TOL_CLASS} / JIS B 0419-{GEO_CLASS}"),
            ("作成", f"agent-cad  {datetime.date.today().isoformat()}")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 120, ytop + 4))
    # general surface texture (JIS B 0031): general value with the exceptions in parentheses
    sh.roughness((x0 + 6, ytop + 6), RA_GENERAL)
    sh.text((x0 + 42, ytop + 7.5), f"( √Ra {RA_BORE} )", 3.5, 0, "left", "bottom")
    sh.meta["roughness_general"] = RA_GENERAL


def doc_no(p):
    """Drawing-number stem unique to a parameter set: variants that differ in any
    parameter (size, angle, material...) never share a document identifier."""
    h = hashlib.sha1(json.dumps(stem.asdict(p), sort_keys=True, default=str).encode()).hexdigest()[:6].upper()
    return f"AC-STEM-{h}"


def _common_notes(p, extra):
    _, finish = MATERIAL_SPEC.get(p.material, ("", ""))
    return [f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS}、幾何公差は JIS B 0419-{GEO_CLASS} による。",
            "指示なきかどは糸面取り C0.2〜0.5、バリなきこと。",
            f"表面処理: {finish}。寸法は表面処理前に適用する。",
            "ねじは JIS B 0205 メートル並目、公差域クラス 6H (JIS B 0209)。",
            "クランプ内径部は傷・打痕なきこと。"] + extra


# ======================================================================= body
def body_sheet(p: stem.StemParams, body, faceplate=None, dwg_no=None):
    d, n, S, B = stem.frame(p)
    dv, nv = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z])
    sh = Sheet(); D = _Dims(sh)
    rho = stem.density(p)

    # ---- measurements
    st_bore = max(find_cyl(body, (0, 0, 1), near=(0, 0, 0), inner=True), key=lambda c: c.r)
    st_out = max(find_cyl(body, (0, 0, 1), near=(0, 0, 0), inner=False), key=lambda c: c.r)
    bar_bore = [c for c in find_cyl(body, (0, 1, 0), inner=True) if c.r > 10][0]
    Bm = bar_bore.loc - (bar_bore.loc @ np.array([0, 1, 0])) * np.array([0, 1, 0])
    zs = [z for z, f, _ in planes(body, (0, 0, 1), near_center=np.zeros(3), radius=1e9)
          if np.hypot(f.center().X, f.center().Y) < st_out.r + 12]
    z0, z1 = min(zs), max(zs)
    Sm = np.array([0, 0, (z0 + z1) / 2])
    L = np.linalg.norm(Bm - Sm); ang = math.degrees(math.atan2(Bm[2] - Sm[2], Bm[0] - Sm[0]))
    y0, y1 = bar_bore.axial_extent()
    light = find_cyl(body, dv, r=p.light_d / 2, inner=True)[0]
    pinch = sorted(find_cyl(body, (0, 1, 0), r=p.bolt_clear_d / 2, inner=True), key=lambda c: c.loc[2])
    taps = find_cyl(body, dv, r=p.tap_drill_d / 2, inner=True)
    cbores = find_cyl(body, (0, 1, 0), r=p.cbore_d / 2, inner=True)
    slot = sorted(y for y, f, _ in planes(body, (0, 1, 0)) if f.center().X < -st_bore.r and abs(f.center().Y) < 5)
    slot_w = slot[-1] - slot[0] if len(slot) >= 2 else float("nan")
    D.need("pinch_holes", len(pinch), 2); D.need("pinch_cbores", len(cbores), 2)
    D.need("fp_taps", len(taps), 4); D.need("slot", 0 if math.isnan(slot_w) else 1, 1)

    # ---- views (third angle): left | front section A-A | aux B ; top above front
    def mk():
        return (View("A-A", body, (0, 1, 0), (1, 0, 0), (0, 0), section=((0, 0, 0), (0, 1, 0)), label="A-A").build(),
                View("top", body, (0, 0, -1), (1, 0, 0), (0, 0)).build(),
                View("left", body, (1, 0, 0), (0, -1, 0), (0, 0)).build(),
                View("B", body, tuple(-dv), (0, 1, 0), (0, 0)).build())
    front, top, left, aux = mk()
    if not sh.place_third_angle(front, top, left, aux, gap_right=60):
        sh.size = A2
        sh.place_third_angle(front, top, left, aux, gap_right=60)
    # sheet furniture first so annotations can avoid it
    mass = body.volume / 1000 * rho
    _title(sh, p, "ステム 本体 (STEM BODY)", f"{dwg_no or doc_no(p)}-01", mass)
    _notes(sh, _common_notes(p, [f"※ φ{_fmt(2 * bar_bore.r)} はフェイスプレートと組み付けた状態で仕上げる (共加工)。",
                                 "ピンチボルト締結トルク 5〜6 N·m、フェイスプレート 4〜5 N·m (参考)。",
                                 "図示なき外形形状は3Dモデル (STEP AP242、同図番) による。"]), (16, 106))
    _general_tol_table(sh, (16, 36))
    fb, tb, ab = front.bbox(), top.bbox(), aux.bbox()
    sh.text(((ab[0][0] + ab[1][0]) / 2, ab[1][1] + 14), "B 矢視", 5, 0, "center", "bottom")
    sh.section_line((tb[0][0] - 12, top.to_sheet((0, 0, 0))[1]), (tb[1][0] + 34, top.to_sheet((0, 0, 0))[1]), "A", look=(0, 1))
    # view arrow B, looking against d at the boss face
    tipB = np.array([fb[1][0] + 6, front.to_sheet(Bm)[1]])
    sh.line([tipB + (9, 0), tipB], "thin"); sh.arrow(tipB, (-1, 0)); sh.text(tipB + (11, 0), "B", 5, 0, "left", "center")

    # ---- centre lines
    sh.centerline(front.to_sheet((0, 0, z0)), front.to_sheet((0, 0, z1)))
    sh.centerline(front.to_sheet(Sm), front.to_sheet(Bm), ext=6)
    sh.center_mark(front.to_sheet(Bm), bar_bore.r)
    sh.center_mark(top.to_sheet((0, 0, 0)), st_out.r)
    sh.centerline(top.to_sheet((0, 0, 0)), top.to_sheet(Bm), ext=6)
    for c in pinch:
        sh.center_mark(front.to_sheet(c.loc), c.r)
    for c in taps:
        sh.center_mark(aux.to_sheet(c.loc), c.r, ext=2)

    # ---- cutting planes of the tube sections near both clamps (before dims, which avoid them)
    hmax = max(p.body_h, p.body_h_end or p.body_h)
    cuts = _section_cuts(sh, p, front, Sm, dv, nv,
                         [("D", st_out.r + 0.5 * hmax * abs(math.sin(math.radians(ang))) + p.junction_r + 3),
                          ("E", L - bar_bore.r - p.bar_wall - p.junction_r - 3)])

    # ---- dimensions: front section
    fs = front.to_sheet
    t = _fit_text(2 * st_bore.r, BORE_FIT)
    sh.dim(fs((-st_bore.r, 0, z0)), fs((st_bore.r, 0, z0)), -15, t[0], tol=t[1])
    D.rec("steerer_bore_d", 2 * st_bore.r, p.steerer_d, t[0], t[1])
    sh.dim(fs((-st_out.r, 0, z0)), fs((-st_out.r, 0, z1)), 22, _fmt(z1 - z0))
    D.rec("stack_h", z1 - z0, p.stack_h, _fmt(z1 - z0))
    sh.dim(fs(Sm), fs(Bm), 38 if ang >= 0 else -38, _fmt(L), u=(fs(Bm) - fs(Sm)))
    D.rec("length", L, p.length, _fmt(L))
    # angle: arc beyond the bar clamp, between the horizontal and the stem axis
    ra = L + bar_bore.r + p.bar_wall + 14
    sh.line([fs(Sm), fs(Sm) + (ra + 6, 0)], "thin")
    sh.angle_dim(fs(Sm), 0, ang, ra, f"{_fmt(ang)}°")
    D.rec("angle_deg", ang, p.angle_deg, f"{_fmt(ang)}°")
    t = _fit_text(2 * bar_bore.r, BORE_FIT)
    k = sh.diameter_dim(fs(Bm), bar_bore.r, [-50, 50, -130, 130, -30, 30], t[0] + " ※", tol=t[1], outside=14, below=16)
    D.rec("bar_bore_d", 2 * bar_bore.r, p.bar_d, t[0], t[1])
    sh.fcf(k + (0, -7.5), ["cylindricity", "0.02"])
    sh.fcf(k + (0, -14), ["perpendicularity", "φ0.05", "A"])
    a_r = math.radians(-120 if ang >= 0 else 120)
    sh.roughness_leader(fs(Bm) + bar_bore.r * np.array([math.cos(a_r), math.sin(a_r)]),
                        fs(Bm) + (bar_bore.r + 8) * np.array([math.cos(a_r), math.sin(a_r)]) + (-6, 0), RA_BORE)
    # lightening bore: diameter across the hollow, in the section
    lp = Sm + dv * (0.45 * L)
    nsh = fs(lp + nv) - fs(lp)
    sh.dim(fs(lp - nv * light.r), fs(lp + nv * light.r), 0.0, f"φ{_fmt(2 * light.r)}", u=nsh)
    D.rec("light_d", 2 * light.r, p.light_d, f"φ{_fmt(2 * light.r)}")
    if len(pinch) == 2:
        pz = pinch[1].loc[2] - pinch[0].loc[2]
        bx = pinch[0].loc[0]
        sh.dim(fs((bx, 0, pinch[0].loc[2])), fs((bx, 0, pinch[1].loc[2])), 12, _fmt(pz))
        D.rec("pinch_pitch", pz, p.pinch_pitch, _fmt(pz))
        sh.dim(fs((bx, 0, pinch[0].loc[2])), fs((0, 0, pinch[0].loc[2])), -(pinch[0].loc[2] - z0) - 5, _fmt(-bx))
        D.rec("pinch_bolt_x", -bx, p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2, _fmt(-bx))
        cb_lo, cb_hi = cbores[0].axial_extent() if cbores else (0.0, 0.0)
        D.rec("pinch_cbore_depth", cb_hi - cb_lo, p.cbore_depth, _fmt(cb_hi - cb_lo))
        D.rec("pinch_clear_d", 2 * pinch[0].r, p.bolt_clear_d, f"φ{_fmt(2 * pinch[0].r)}")
        if cbores:
            D.rec("pinch_cbore_d", 2 * cbores[0].r, p.cbore_d, f"φ{_fmt(2 * cbores[0].r)}")
        lb = left.bbox()
        sh.leader(fs(pinch[0].loc - (0, 0, pinch[0].r * 0.7)),
                  [lb[0] + (-4, -12), lb[0] + (-4, -22), (lb[0][0] - 4, lb[1][1] + 26)], "", right=True,
                  lines=[f"2×φ{_fmt(2 * pinch[0].r)} キリ", f"  φ{_fmt(2 * cbores[0].r) if cbores else '-'} 深ザグリ 深さ{_fmt(cb_hi - cb_lo)} (+Y側)",
                         f"  反対側 M5×0.8 深さ{_fmt(math.floor((p.ear_half_w - p.slot_w / 2 - 1) * 2) / 2)} (下穴φ{_fmt(p.tap_drill_d)})"])
    # GD&T on the steerer bore, datum A = steerer bore
    sh.datum(fs((st_bore.r, 0, z0)) + (0, -15), (0, -1), "A", length=5)
    sh.fcf(fs((-st_bore.r, 0, z0)) + (0, -24), ["cylindricity", "0.02"])
    sh.roughness_leader(fs((-st_bore.r, 0, z1 - 6)), fs((-st_bore.r - 16, 0, z1 + 8)), RA_BORE)

    # ---- top view
    ts = top.to_sheet
    sh.dim(ts((Bm[0], y0, 0)), ts((Bm[0], y1, 0)), -22, _fmt(y1 - y0), u=(0, 1))
    D.rec("bar_clamp_w", y1 - y0, p.bar_clamp_w, _fmt(y1 - y0))
    sh.diameter_dim(ts((0, 0, 0)), st_out.r, 150, f"φ{_fmt(2 * st_out.r)}", outside=8)
    D.rec("steerer_clamp_od", 2 * st_out.r, p.steerer_d + 2 * p.steerer_wall, f"φ{_fmt(2 * st_out.r)}")

    # ---- left view: slot
    ls = left.to_sheet
    if not math.isnan(slot_w):
        sh.dim(ls((0, slot[0], z1)), ls((0, slot[1], z1)), 14, _fmt(slot_w), u=(1, 0))
        D.rec("slot_w", slot_w, p.slot_w, _fmt(slot_w))

    # ---- aux view B: faceplate bolt pattern
    if len(taps) == 4:
        q = np.array([[c.loc @ np.array([0, 1, 0]), (c.loc - Bm) @ nv] for c in taps])
        wy, hn = q[:, 0].max() - q[:, 0].min(), q[:, 1].max() - q[:, 1].min()
        a_ = taps[int(np.argmin(q[:, 0] + q[:, 1]))].loc
        lo = (ab[0][1] - aux.to_sheet(a_)[1])
        sh.dim(aux.to_sheet(a_), aux.to_sheet(a_ + np.array([0, wy, 0])), lo - 8, _fmt(wy))
        sh.dim(aux.to_sheet(a_), aux.to_sheet(a_ + nv * hn), (aux.to_sheet(a_)[0] - ab[0][0]) + 8, _fmt(hn))
        D.rec("fp_bolt_pitch_y", wy, 2 * p.fp_bolt_w, _fmt(wy)); D.rec("fp_bolt_pitch_n", hn, 2 * p.fp_bolt_off, _fmt(hn))
        tr = taps[int(np.argmax(q[:, 0] + q[:, 1]))].loc
        e0, e1 = taps[0].axial_extent()      # opens on the mating face, so the extent is the drill depth
        D.rec("fp_tap_drill_depth", e1 - e0, p.tap_depth, _fmt(e1 - e0))
        D.rec("fp_tap_drill_d", 2 * taps[0].r, p.tap_drill_d, f"φ{_fmt(2 * taps[0].r)}")
        sh.leader(aux.to_sheet(tr) + (1.2, 1.2), [(ab[1][0] + 4, ab[1][1] + 4), (ab[0][0] - 4, ab[1][1] + 4)], "",
                  lines=[f"4×M5×0.8 深さ{_fmt(e1 - e0 - 2)}", f"  下穴φ{_fmt(2 * taps[0].r)} 深さ{_fmt(e1 - e0)}"])
        w_, h_ = sh.fcf((ab[0][0], ab[0][1] - 26), ["flatness", "0.05"])
        sh.text((ab[0][0], ab[0][1] - 32), "合わせ面 (図示面)", 3, 0, "left", "bottom")
        sh.roughness((ab[0][0] + w_ + 6, ab[0][1] - 26), RA_FACE)

    # ---- removed sections D-D, E-E (views placed last, in the space left over)
    _body_sections(sh, D, p, body, Sm, dv, cuts)

    # section label under everything drawn below the front view
    ys = [it[2] for it in sh.items if it[0] == "text" and fb[0][0] - 20 < it[1] < fb[1][0] + 20 and 112 < it[2] < fb[0][1]]
    ys += [pt[1] for it in sh.items if it[0] == "line" for pt in it[1] if fb[0][0] < pt[0] < fb[1][0] and 112 < pt[1] < fb[0][1]]
    sh.text(((fb[0][0] + fb[1][0]) / 2, min(ys + [fb[0][1]]) - 4), "A-A", 5, 0, "center", "top")
    sh.meta["part"] = "STEM_BODY"
    return sh


def _section_cuts(sh, p, front, Sm, dv, nv, stations):
    """Cutting planes for the removed tube sections, drawn before the dimensions so
    those avoid them: a station near the nominal one whose line and labels clear text."""
    he = p.body_h_end or p.body_h
    fs = front.to_sheet
    look = fs(Sm + dv) - fs(Sm); lk = look / np.linalg.norm(look)
    sh._opts = None
    texts = [text_box((it[1], it[2]), it[3], it[4], it[5], it[6], it[7]) for it in sh.items if it[0] == "text"]
    cuts = []
    for lab, t0 in stations:
        best = None
        for dt in (0, 3, 6, 9, 12) if lab == "D" else (0, -3, -6, -9, -12):   # move away from the clamp only
            for k in (7, 12, 4):
                t = t0 + dt
                hi = p.body_h + (he - p.body_h) * t / p.length
                a, b = fs(Sm + dv * t + nv * (0.5 * hi + k)), fs(Sm + dv * t - nv * (0.5 * hi + k))
                u = (b - a) / np.linalg.norm(b - a)
                seg = np.vstack([a - u * 5 + np.outer(np.linspace(0, 1, 60), (b - a) + u * 10),
                                 a + np.outer(np.linspace(0, 1, 8), lk * 7), b + np.outer(np.linspace(0, 1, 8), lk * 7)])
                hit = any(((seg[:, 0] > t_[0][0] - 0.5) & (seg[:, 0] < t_[1][0] + 0.5) &
                           (seg[:, 1] > t_[0][1] - 0.5) & (seg[:, 1] < t_[1][1] + 0.5)).any() for t_ in texts)
                if not hit and all(sh.is_free(text_box(q, lab, 5, 0, "center", "center"), pad=1)
                                   for q in (a - u * 4 + lk * 3, b + u * 4 + lk * 3)):
                    best = (t, a, b); break
            if best:
                break
        t, a, b = best or (t0, fs(Sm + dv * t0 + nv * 20), fs(Sm + dv * t0 - nv * 20))
        sh.section_line(a, b, lab, look=look)
        cuts.append((lab, t))
    sh._opts = None
    return cuts


def _body_sections(sh, D, p, body, Sm, dv, cuts):
    """Removed sections of the lofted tube: width/height measured on the solid and
    checked against the loft (linear between the end sections)."""
    from build123d import Plane, Vector
    we, he = (p.body_w_end or p.body_w), (p.body_h_end or p.body_h)
    placed = []
    for lab, t in reversed(cuts):      # placed right to left, so D ends up left of E
        c = Sm + dv * t
        sec = body.intersect(Plane(origin=Vector(*c), x_dir=Vector(0, 1, 0), z_dir=Vector(*dv)))
        faces = sec.faces() if sec is not None else []
        D.need(f"section_{lab}", len(faces), 1)
        if len(faces) != 1:
            continue
        v = View(f"{lab}-{lab}", faces[0], tuple(dv), (0, -1, 0), (0, 0)).build()
        v.cut_faces = [[np.array([v.to_sheet(q) for e in w.edges() for q in _pts(e)]) for w in faces[0].wires()]]
        b0, b1 = v.bbox()
        w_, h_ = b1 - b0
        # free spot for the view plus its dimensions and label
        need = (w_ + 22, h_ + 26)
        W, H = sh.size
        spot = None
        for y in np.arange(H - 20 - need[1], 112, -4.0):
            for x in np.arange(W - 20 - need[0], 20, -4.0):
                box = (np.array([x, y]), np.array([x + need[0], y + need[1]]))
                if sh.is_free(box, pad=2) and not any(_boxes_hit(box, q) for q in placed):
                    spot = box; break
            if spot:
                break
        D.need(f"section_{lab}_placed", spot is not None, 1)
        if spot is None:
            continue
        placed.append(spot)
        v.shift(*(spot[0] + (12, 12) - b0))
        sh.add_view(v, built=True); sh._opts = None
        b0, b1 = v.bbox()
        sh.text(((b0[0] + b1[0]) / 2, b1[1] + 5), f"{lab}-{lab}", 5, 0, "center", "bottom")
        ww, hh = (b1 - b0)
        wi, hi = p.body_w + (we - p.body_w) * t / p.length, p.body_h + (he - p.body_h) * t / p.length
        sh.dim((b0[0], b0[1]), (b1[0], b0[1]), -7, _fmt(ww))
        sh.dim((b1[0], b0[1]), (b1[0], b1[1]), -7, _fmt(hh), u=(0, 1))
        D.rec(f"section_{lab}_w", ww, wi, _fmt(ww)); D.rec(f"section_{lab}_h", hh, hi, _fmt(hh))


def _pts(e, n=24):
    return [np.array(tuple(e.position_at(k / n))) for k in range(n + 1)]


def _boxes_hit(a, b):
    return not (a[1][0] < b[0][0] or b[1][0] < a[0][0] or a[1][1] < b[0][1] or b[1][1] < a[0][1])


# ======================================================================= faceplate
def _window(fp, Bm, dv, nv, p):
    """Through window at the faceplate centre, measured on the solid: a line along d
    misses the faceplate inside the window. Returns (width, height, corner radius) or None."""
    from build123d import Edge
    yv = np.array([0.0, 1.0, 0.0])

    def open_at(y, h):
        c = Bm + yv * y + nv * h
        return fp.distance_to(Edge.make_line(tuple(c - dv * 100), tuple(c + dv * 100))) > 1e-6

    if not open_at(0, 0):
        return None

    def edge(u, step=0.5):          # walk out to the first wall along u (y, n), then bisect
        u = np.array(u, float); lo = 0.0
        while open_at(*(u * (lo + step))):
            lo += step
            if lo > 200:
                return float("inf")
        hi = lo + step
        for _ in range(30):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if open_at(*(u * m)) else (lo, m)
        return lo
    ww, wh = edge((1, 0)) + edge((-1, 0)), edge((0, 1)) + edge((0, -1))
    rs = [c.r for c in find_cyl(fp, dv, inner=True)
          if abs((c.loc - Bm) @ yv) < ww / 2 and abs((c.loc - Bm) @ nv) < wh / 2]
    return ww, wh, (min(rs) if rs else 0.0)


def _fits(sh):
    from ..drawing.check import check_sheet
    bad = {c["check"] for c in check_sheet(sh)["checks"] if not c["pass"]}
    return not bad & {"layout.inside_frame", "layout.text_overlap", "layout.text_on_geometry"}


def faceplate_sheet(p: stem.StemParams, fp, scale=2.0, dwg_no=None):
    """A3 at `scale`; if the layout does not fit, smaller scales, then A2."""
    tries = [(A3, s) for s in dict.fromkeys((scale, 1.5, 1.0)) if s <= scale] + [(A2, scale), (A2, 1.0)]
    for size, sc in tries:
        sh = _faceplate_sheet(p, fp, sc, size, dwg_no)
        if _fits(sh):
            return sh
    return sh


def _faceplate_sheet(p, fp, scale, size, dwg_no):
    d, n, S, B = stem.frame(p)
    dv, nv = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z])
    Bm = np.array([B.X, B.Y, B.Z])
    sh = Sheet(size); D = _Dims(sh)
    kx, ky = size[0] / A3[0], size[1] / A3[1]
    bore = [c for c in find_cyl(fp, (0, 1, 0), inner=True) if c.r > 10][0]
    holes = find_cyl(fp, dv, r=p.bolt_clear_d / 2, inner=True)
    cb = find_cyl(fp, dv, r=p.cbore_d / 2, inner=True)
    y0, y1 = bore.axial_extent()
    fronts = planes(fp, dv)
    # boss face (outward +d) and mating face (outward -d): the largest planar face of each
    t_front = max((f.area, z) for z, f, nn in fronts if nn @ dv > 0)[1] - Bm @ dv
    t_back = max((f.area, z) for z, f, nn in fronts if nn @ dv < 0)[1] - Bm @ dv
    ext_n = [((np.array([v.X, v.Y, v.Z]) - Bm) @ nv) for v in fp.vertices()]
    hn_all = max(ext_n) - min(ext_n)

    def place(name, direction, xdir, at, **kw):
        v0 = View(name, fp, direction, xdir, (0, 0), scale)
        return sh.add_view(View(name, fp, direction, xdir, tuple(np.array(at) - v0.to_sheet(Bm)), scale, **kw))
    # front view = looking at the faceplate from the rider side (along -d); section C-C along the stem plane
    front = place("F", tuple(-dv), (0, 1, 0), (125 * kx, 190 * ky))
    side = place("C-C", (0, 1, 0), tuple(dv), (265 * kx, 190 * ky), section=(tuple(Bm), (0, 1, 0)))
    fs, ss = front.to_sheet, side.to_sheet
    sh.text(ss(Bm) + (0, scale * hn_all / 2 + 20), "C-C", 5, 0, "center", "bottom")
    sh.section_line(fs(Bm) + (0, scale * hn_all / 2 + 20), fs(Bm) - (0, scale * hn_all / 2 + 20), "C", look=(1, 0))
    sh.center_mark(ss(Bm), scale * bore.r)
    sh.centerline(fs(Bm - np.array([0, (y1 - y0) / 2, 0])), fs(Bm + np.array([0, (y1 - y0) / 2, 0])))
    # section: bore, thickness, height
    t = _fit_text(2 * bore.r, BORE_FIT)
    k = sh.diameter_dim(ss(Bm), scale * bore.r, [-35, 35, -50], t[0] + " ※", tol=t[1], outside=22, below=9)
    sh.fcf(k + (0, -7.5), ["cylindricity", "0.02"])
    D.rec("bar_bore_d", 2 * bore.r, p.bar_d, t[0], t[1])
    # bore roughness: symbol inside the bore when it fits, else on a leader to free space
    for a_deg, k in ((-90, 0.0), (-120, 1.0), (-60, 1.0), (-150, 1.0), (-30, 1.0)):
        u = np.array([math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))])
        tip = ss(Bm) + u * scale * bore.r
        pos = tip + (-4, 0) if k == 0 else tip + u * 18 + (-8, 0)
        if sh.is_free(text_box(pos + (5.4, 7.6), f"Ra {RA_BORE}", 3.5, 0, "left", "bottom")) and \
                sh.is_free(((pos[0] - 3, pos[1]), (pos[0] + 6, pos[1] + 7))):
            break
    if k == 0:
        sh.roughness(pos, RA_BORE)
    else:
        sh.roughness_leader(tip, pos, RA_BORE)
    top_n = max(ext_n)
    sh.dim(ss(Bm + nv * top_n), ss(Bm + nv * top_n + dv * t_front), 8, _fmt(t_front))
    D.rec("front_thickness", t_front, p.fp_front, _fmt(t_front))
    sh.dim(ss(Bm + dv * t_back + nv * min(ext_n)), ss(Bm + dv * t_back + nv * top_n), 12, _fmt(hn_all), u=(0, 1))
    D.rec("height", hn_all, 2 * (p.fp_bolt_off + p.boss_margin), _fmt(hn_all))
    sh.dim(ss(Bm + nv * min(ext_n)), ss(Bm + nv * min(ext_n) + dv * t_back), -8, _fmt(t_back))
    D.rec("gap_half", t_back, p.fp_gap / 2, _fmt(t_back))
    sh.fcf(ss(Bm + dv * t_back + nv * min(ext_n)) + (-26, -16), ["flatness", "0.05"])
    sh.text(ss(Bm + dv * t_back + nv * min(ext_n)) + (-26, -21), "合わせ面", 3, 0, "left", "bottom")
    # front: width, bolt pattern, window
    lo_n = min(ext_n)
    sh.dim(fs(Bm + np.array([0, y0, 0]) + nv * lo_n), fs(Bm + np.array([0, y1, 0]) + nv * lo_n), -10, _fmt(y1 - y0))
    D.rec("width", y1 - y0, p.bar_clamp_w, _fmt(y1 - y0))
    for c in holes:
        sh.center_mark(fs(c.loc), scale * c.r, 2)
    D.need("bolt_holes", len(holes), 4); D.need("cbores", len(cb), 4)
    if holes:
        D.rec("bolt_clear_d", 2 * holes[0].r, p.bolt_clear_d, f"φ{_fmt(2 * holes[0].r)}")
    if cb:
        D.rec("cbore_d", 2 * cb[0].r, p.cbore_d, f"φ{_fmt(2 * cb[0].r)}")
    if len(holes) == 4:
        q = np.array([[c.loc[1], (c.loc - Bm) @ nv] for c in holes])
        wy, hn = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        tl = holes[int(np.argmax(-q[:, 0] + q[:, 1]))].loc      # top-left hole
        tr = tl + np.array([0, wy, 0])
        sh.dim(fs(tl), fs(tr), (top_n - (tl - Bm) @ nv) * scale + 8, _fmt(wy))
        sh.dim(fs(tr - nv * hn), fs(tr), -((y1 - tr[1]) * scale + 10), _fmt(hn), u=(0, 1))
        D.rec("bolt_pitch_y", wy, 2 * p.fp_bolt_w, _fmt(wy)); D.rec("bolt_pitch_n", hn, 2 * p.fp_bolt_off, _fmt(hn))
        cb_depth = t_front - (cb[0].axial_extent()[0] - Bm @ dv) if cb else 0.0
        D.rec("cbore_depth", cb_depth, 2.5, _fmt(cb_depth))
        fb_ = front.bbox()
        sh.leader(fs(tl) + np.array([-1, 1]) * scale * holes[0].r * 0.7,
                  [fs(Bm) + (-75, 62), (fb_[0][0] - 10, fb_[1][1] + 10), (fb_[0][0] - 30, fb_[1][1] + 18),
                   (fb_[0][0] + 10, fb_[1][1] + 22)], "", right=True,
                  lines=[f"4×φ{_fmt(2 * holes[0].r)} キリ", f"φ{_fmt(2 * cb[0].r) if cb else '-'} 深ザグリ 深さ{_fmt(cb_depth)}"])
    win = _window(fp, Bm, dv, nv, p)
    want_win = p.fp_window_w > 0 and p.fp_window_h > 0     # stem.build() cuts a window only then
    if win or want_win:
        ww, wh, wr = win or (0.0, 0.0, 0.0)
        if win:
            fb_ = front.bbox()
            sh.leader(fs(Bm - np.array([0, ww / 2, 0])),
                      [fs(Bm) + (-60, -48), (fb_[0][0] - 8, fb_[0][1] - 10), (fb_[0][0] - 20, fb_[0][1] - 20)], "", right=False,
                      lines=[f"窓 {_fmt(ww)}×{_fmt(wh)}", f"R{_fmt(wr)} 貫通"])
        D.rec("window_w", ww, p.fp_window_w if want_win else 0.0, _fmt(ww))
        D.rec("window_h", wh, p.fp_window_h if want_win else 0.0, _fmt(wh))
    mass = fp.volume / 1000 * stem.density(p)
    _title(sh, p, "ステム フェイスプレート (FACEPLATE)", f"{dwg_no or doc_no(p)}-02", mass, scale=f"{scale:g}:1")
    _notes(sh, _common_notes(p, [f"※ φ{_fmt(2 * bore.r)} は本体と組み付けた状態で仕上げる (共加工)。",
                                 f"本体との合わせ面のすき間 {_fmt(2 * t_back)} (締結前、参考)。"]), (16, 106))
    _general_tol_table(sh, (16, 36))
    sh.meta["part"] = "STEM_FACEPLATE"
    return sh
