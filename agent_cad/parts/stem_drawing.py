"""JIS drawings (A3, third-angle, 1:1) of the stem body and faceplate.

Every dimension value is measured from the B-rep (agent_cad.drawing.measure)
and recorded in sheet.meta["dims"] next to the design intent from the
parameters, so verify_drawing() can catch model/drawing disagreement.
"""
import math, datetime, hashlib, json
import numpy as np
from build123d import Vector
from ..drawing.sheet import Sheet, View, A2, A3, text_box, text_width
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
    d = float(_fmt(d))           # deviations belong to the nominal as printed (size band boundaries)
    up, lo = jis.fit_deviations(d, code)
    return f"φ{_fmt(d)} {code}", (f"{up:+.3f}", f"{lo:+.3f}" if lo else "0")


class _Dims:
    def __init__(self, sheet):
        self.sheet = sheet; sheet.meta["dims"] = []

    def rec(self, name, measured, intent, text, tol=None):
        self.sheet.meta["dims"].append(dict(name=name, measured=round(float(measured), 4),
                                            intent=float(intent), text=text, tol=tol))

    def ra(self, name, value, required=None):
        """Local surface texture: `required` is the value the design asks for,
        `value` what the sheet shows (None: not drawn); check_sheet compares them."""
        self.sheet.meta.setdefault("roughness_local", []).append(
            dict(name=name, drawn=value, required=value if required is None else required))

    def each(self, name, values, intent, text):
        """Record one dimension per feature an N× callout covers; the first keeps the plain name."""
        for i, v in enumerate(values):
            self.rec(name if i == 0 else f"{name}#{i + 1}", v, intent, text(v))

    def need(self, name, found, expected):
        """Required feature: check_sheet fails if the model does not have it."""
        self.sheet.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))


def _unrecognised(p, size, part, part_name, dwg, found):
    """A model lacking a primary feature cannot be drawn: a sheet that only records
    the missing features (check_sheet fails it) instead of an exception."""
    sh = Sheet(size); D = _Dims(sh)
    for k, v in found.items():
        D.need(k, v, 1)
    _title(sh, p, part_name, dwg, 0.0)
    sh.meta["part"] = part
    return sh


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
    cw, rh = 17.0, 4.6
    y -= 1.5
    # JIS B 0405 band limits: the first band includes its lower limit, the others start just above it
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


def _scale_text(k):
    return f"{k:g}:1" if k >= 1 else f"1:{1 / k:g}"


def doc_no(p):
    """Drawing-number stem unique to a parameter set: variants that differ in any
    parameter (size, angle, material...) never share a document identifier."""
    h = hashlib.sha1(json.dumps(stem.asdict(p), sort_keys=True, default=str).encode()).hexdigest()[:12].upper()
    return f"AC-STEM-{h}"


def _common_notes(sh, p, extra):
    # a material outside MATERIAL_SPEC has no finish: the note says so and check_sheet fails
    _, finish = MATERIAL_SPEC.get(p.material, ("", ""))
    sh.meta["finish"] = finish
    return [f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS}、幾何公差は JIS B 0419-{GEO_CLASS} による。",
            "指示なきかどは糸面取り C0.2〜0.5、バリなきこと。",
            f"表面処理: {finish or '未指定 (MATERIAL_SPEC に材料を登録すること)'}。寸法は表面処理前に適用する。",
            "ねじは JIS B 0205 メートル並目、公差域クラス 6H (JIS B 0209)。",
            "クランプ内径部は傷・打痕なきこと。"] + extra


# ======================================================================= body
BODY_TRIES = [(A3, 1.0), (A2, 1.0), (A3, 0.5), (A2, 0.5), (A2, 0.2)]   # JIS Z 8314 preferred scales


def body_sheet(p: stem.StemParams, body, faceplate=None, dwg_no=None):
    """A3 at 1:1; if the views or their annotations do not fit, A2, then 1:2 and 1:5."""
    sh = None
    for size, sc in BODY_TRIES:
        cand = _body_sheet(p, body, dwg_no, size, sc)
        if cand is None:            # the views alone do not fit: never annotate unplaced views
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError(f"stem body views do not fit A2 at 1:{1 / BODY_TRIES[-1][1]:g}")
    return sh


def _body_sheet(p, body, dwg_no, size, k):
    d, n, S, B = stem.frame(p)
    dv, nv = np.array([d.X, d.Y, d.Z]), np.array([n.X, n.Y, n.Z])
    sh = Sheet(size); D = _Dims(sh)
    rho = stem.density(p)

    # ---- measurements
    st_bore = max(find_cyl(body, (0, 0, 1), near=(0, 0, 0), inner=True), key=lambda c: c.r, default=None)
    st_out = max(find_cyl(body, (0, 0, 1), near=(0, 0, 0), inner=False), key=lambda c: c.r, default=None)
    # the bar bore: the largest Y hole whose axis passes the bar axis (any size; pinch holes lie elsewhere)
    bar_bore = max(find_cyl(body, (0, 1, 0), inner=True, near=(B.X, 0, B.Z)), key=lambda c: c.r, default=None)
    if None in (st_bore, st_out, bar_bore):
        return _unrecognised(p, size, "STEM_BODY", "ステム 本体 (STEM BODY)", f"{dwg_no or doc_no(p)}-01",
                             dict(steerer_bore=st_bore is not None, steerer_clamp=st_out is not None,
                                  bar_bore=bar_bore is not None))
    Bm = bar_bore.loc - (bar_bore.loc @ np.array([0, 1, 0])) * np.array([0, 1, 0])
    # the views put the steerer clamp on the Z axis and draw the bore concentric with it
    D.rec("steerer_clamp_axis_offset", math.hypot(*st_out.loc[:2]), 0.0, "")
    D.rec("steerer_bore_axis_offset", math.hypot(*(st_bore.loc[:2] - st_out.loc[:2])), 0.0, "")
    z0, z1 = st_out.axial_extent()       # the clamp cylinder itself: pinch bosses may stand proud of it
    Sm = np.array([0, 0, (z0 + z1) / 2])
    L = np.linalg.norm(Bm - Sm); ang = math.degrees(math.atan2(Bm[2] - Sm[2], Bm[0] - Sm[0]))
    y0, y1 = bar_bore.axial_extent()
    # lightening bore: the internal cylinder along the stem axis whose axis lies inside the body
    # section (faceplate taps are far off it), whatever its size; of several, the longest
    # (a sliver left where the bore meets another is not it)
    def off_axis(c):
        rel = c.loc - Sm
        return np.linalg.norm(rel - (rel @ dv) * dv)
    light = min((c for c in find_cyl(body, dv, inner=True) if off_axis(c) < min(p.body_w, p.body_h) / 2 - c.r),
                key=lambda c: (-round(np.subtract(*c.axial_extent()[::-1]), 3), off_axis(c)), default=None)
    D.need("light_bore", light is not None, 1)
    pinch = sorted(find_cyl(body, (0, 1, 0), r=p.bolt_clear_d / 2, inner=True), key=lambda c: c.loc[2])
    taps = find_cyl(body, dv, r=p.tap_drill_d / 2, inner=True)
    cbores = find_cyl(body, (0, 1, 0), r=p.cbore_d / 2, inner=True)
    slot_faces = [(y, f) for y, f, _ in planes(body, (0, 1, 0)) if f.center().X < -st_bore.r and abs(f.center().Y) < 5]
    slot = sorted({round(y, 6) for y, _ in slot_faces})
    slot_w = slot[-1] - slot[0] if len(slot) >= 2 else float("nan")
    if len(slot) >= 2:                   # the split is symmetric about the stem plane (Y = 0)
        D.rec("slot_center_y", (slot[0] + slot[-1]) / 2, 0.0, "")
        # the split runs through the whole clamp: across the clamp wall, each side face
        # covers z0..z1 (sampled every ~1 mm); record the length it leaves uncovered
        from build123d import Compound, Vector
        xm = -(st_bore.r + st_out.r) / 2
        zs = np.linspace(z0 + 0.25, z1 - 0.25, max(9, int(z1 - z0) + 1))
        for side, ys in (("lo", slot[0]), ("hi", slot[-1])):
            fs_ = Compound([f for y, f in slot_faces if abs(y - ys) < 1e-6])
            miss = sum(fs_.distance_to(Vector(xm, ys, z)) > 1e-3 for z in zs)
            D.rec(f"slot_{side}_uncovered", miss / len(zs) * (z1 - z0), 0.0, "")
    D.need("pinch_holes", len(pinch), 2); D.need("pinch_cbores", len(cbores), 2)
    D.need("fp_taps", len(taps), 4); D.need("slot", 0 if math.isnan(slot_w) else 1, 1)
    ptaps = find_cyl(body, (0, 1, 0), r=p.tap_drill_d / 2, inner=True)     # pinch threads, -Y ear
    D.need("pinch_taps", len(ptaps), 2)
    D.need("fp_tap_pattern", _corners([((c.loc - Bm) @ np.array([0, 1, 0]), (c.loc - Bm) @ nv) for c in taps],
                                      p.fp_bolt_w, p.fp_bolt_off), 4)

    # ---- views (third angle): left | front section A-A | aux B ; top above front
    def mk():
        return (View("A-A", body, (0, 1, 0), (1, 0, 0), (0, 0), k, section=((0, 0, 0), (0, 1, 0)), label="A-A").build(),
                View("top", body, (0, 0, -1), (1, 0, 0), (0, 0), k).build(),
                View("left", body, (1, 0, 0), (0, -1, 0), (0, 0), k).build(),
                View("B", body, tuple(-dv), (0, 1, 0), (0, 0), k).build())
    front, top, left, aux = mk()
    if not sh.place_third_angle(front, top, left, aux, gap_right=60):
        return None
    # sheet furniture first so annotations can avoid it
    mass = body.volume / 1000 * rho
    _title(sh, p, "ステム 本体 (STEM BODY)", f"{dwg_no or doc_no(p)}-01", mass, scale=_scale_text(k))
    _notes(sh, _common_notes(sh, p, [f"※ φ{_fmt(2 * bar_bore.r)} はフェイスプレートと組み付けた状態で仕上げる (共加工)。",
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
    sh.center_mark(front.to_sheet(Bm), k * bar_bore.r)
    sh.center_mark(top.to_sheet((0, 0, 0)), k * st_out.r)
    sh.centerline(top.to_sheet((0, 0, 0)), top.to_sheet(Bm), ext=6)
    for c in pinch:
        sh.center_mark(front.to_sheet(c.loc), k * c.r)
    for c in taps:
        sh.center_mark(aux.to_sheet(c.loc), k * c.r, ext=2)

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
    ra = k * (L + bar_bore.r + p.bar_wall) + 14
    sh.line([fs(Sm), fs(Sm) + (ra + 6, 0)], "thin")
    sh.angle_dim(fs(Sm), 0, ang, ra, f"{_fmt(ang)}°")
    D.rec("angle_deg", ang, p.angle_deg, f"{_fmt(ang)}°")
    t = _fit_text(2 * bar_bore.r, BORE_FIT)
    kp = sh.diameter_dim(fs(Bm), k * bar_bore.r, [-50, 50, -130, 130, -30, 30], t[0] + " ※", tol=t[1], outside=14, below=16)
    D.rec("bar_bore_d", 2 * bar_bore.r, p.bar_d, t[0], t[1])
    sh.fcf(kp + (0, -7.5), ["cylindricity", "0.02"])
    sh.fcf(kp + (0, -14), ["perpendicularity", "φ0.05", "A"])
    a_r = math.radians(-120 if ang >= 0 else 120)
    sh.roughness_leader(fs(Bm) + k * bar_bore.r * np.array([math.cos(a_r), math.sin(a_r)]),
                        fs(Bm) + (k * bar_bore.r + 8) * np.array([math.cos(a_r), math.sin(a_r)]) + (-6, 0), RA_BORE)
    D.ra("bar_bore", RA_BORE)
    # lightening bore: diameter across the hollow, in the section
    lp = Sm + dv * (0.45 * L)
    nsh = fs(lp + nv) - fs(lp)
    if light is not None and k * 2 * light.r >= text_width(f"φ{_fmt(2 * light.r)}", 3.5) + 6:
        sh.dim(fs(lp - nv * light.r), fs(lp + nv * light.r), 0.0, f"φ{_fmt(2 * light.r)}", u=nsh)
    elif light is not None:     # reduced scale: the text no longer fits inside the bore, use a leader
        tip = fs(lp + nv * light.r)
        sh.leader(tip, [tip + (14, 22), tip + (-14, 22), tip + (14, -22), tip + (-14, -22)], f"φ{_fmt(2 * light.r)}")
    if light is not None:
        D.rec("light_d", 2 * light.r, p.light_d, f"φ{_fmt(2 * light.r)}")
        # drawn on the stem centreline Sm-Bm: the axis must lie on it
        rel = light.loc - Sm
        D.rec("light_axis_offset", np.linalg.norm(rel - (rel @ dv) * dv), 0.0, "")
    if len(pinch) == 2:
        pz = pinch[1].loc[2] - pinch[0].loc[2]
        bx = pinch[0].loc[0]
        sh.dim(fs((bx, 0, pinch[0].loc[2])), fs((bx, 0, pinch[1].loc[2])), 12, _fmt(pz))
        D.rec("pinch_pitch", pz, p.pinch_pitch, _fmt(pz))
        sh.dim(fs((bx, 0, pinch[0].loc[2])), fs((0, 0, pinch[0].loc[2])), -k * (pinch[0].loc[2] - z0) - 5, _fmt(-bx))
        D.rec("pinch_bolt_x", -bx, p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2, _fmt(-bx))
        # the 2x callout covers both holes: each axis at the drawn (x, z), cbore and tap on it
        D.each("pinch_hole_x", [-c.loc[0] for c in pinch], p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2, _fmt)
        D.rec("pinch_center_z", (pinch[0].loc[2] + pinch[1].loc[2]) / 2 - (z0 + z1) / 2, 0.0, "")

        def xz_off(c, others):
            return min((math.hypot(c.loc[0] - o.loc[0], c.loc[2] - o.loc[2]) for o in others), default=float("inf"))
        D.each("pinch_cbore_coaxial", [xz_off(c, cbores) for c in pinch], 0.0, lambda v: "")
        D.each("pinch_tap_coaxial", [xz_off(c, ptaps) for c in pinch], 0.0, lambda v: "")
        cb_lo, cb_hi = cbores[0].axial_extent() if cbores else (0.0, 0.0)
        # every hole the 2x callouts cover, not just the first
        D.each("pinch_cbore_depth", [np.subtract(*c.axial_extent()[::-1]) for c in cbores] or [0.0], p.cbore_depth, _fmt)
        D.each("pinch_clear_d", [2 * c.r for c in pinch], p.bolt_clear_d, lambda v: f"φ{_fmt(v)}")
        D.each("pinch_cbore_d", [2 * c.r for c in cbores], p.cbore_d, lambda v: f"φ{_fmt(v)}")
        pt_lo, pt_hi = ptaps[0].axial_extent() if ptaps else (0.0, 0.0)
        D.each("pinch_tap_drill_d", [2 * c.r for c in ptaps], p.tap_drill_d, lambda v: f"φ{_fmt(v)}")
        D.each("pinch_tap_length", [np.subtract(*c.axial_extent()[::-1]) for c in ptaps], p.ear_half_w - p.slot_w / 2, _fmt)
        pthr = stem.usable_thread(pt_hi - pt_lo, through=True)
        D.rec("pinch_thread_depth", pthr, stem.thread_depths(p)["pinch"], _fmt(pthr))
        lb = left.bbox()
        sh.leader(fs(pinch[0].loc - (0, 0, pinch[0].r * 0.7)),
                  [lb[0] + (-4, -12), lb[0] + (-4, -22), (lb[0][0] - 4, lb[1][1] + 26)], "", right=True,
                  lines=[f"2×φ{_fmt(2 * pinch[0].r)} キリ", f"  φ{_fmt(2 * cbores[0].r) if cbores else '-'} 深ザグリ 深さ{_fmt(cb_hi - cb_lo)} (+Y側)",
                         f"  反対側 M5×0.8 深さ{_fmt(pthr)} (下穴φ{_fmt(2 * ptaps[0].r) if ptaps else '-'})"])
    # GD&T on the steerer bore, datum A = steerer bore
    sh.datum(fs((st_bore.r, 0, z0)) + (0, -15), (0, -1), "A", length=5)
    w_fcf = 6.5 + text_width("0.02", 3.5) + 3       # keep the frame clear of the datum A square
    x_fcf = min(fs((-st_bore.r, 0, z0))[0], fs((st_bore.r, 0, z0))[0] - 3.25 - 1.5 - w_fcf)
    sh.fcf((x_fcf, fs((0, 0, z0))[1] - 24), ["cylindricity", "0.02"])
    sh.roughness_leader(fs((-st_bore.r, 0, z1 - 6)), fs((-st_bore.r, 0, z1 - 6)) + (-16, 14), RA_BORE)
    D.ra("steerer_bore", RA_BORE)

    # ---- top view
    ts = top.to_sheet
    sh.dim(ts((Bm[0], y0, 0)), ts((Bm[0], y1, 0)), -22, _fmt(y1 - y0), u=(0, 1))
    D.rec("bar_clamp_w", y1 - y0, p.bar_clamp_w, _fmt(y1 - y0))
    sh.diameter_dim(ts((0, 0, 0)), k * st_out.r, 150, f"φ{_fmt(2 * st_out.r)}", outside=8)
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
        fthr = stem.usable_thread(min(np.subtract(*c.axial_extent()[::-1]) for c in taps), through=False)
        D.rec("fp_thread_depth", fthr, stem.thread_depths(p)["faceplate"], _fmt(fthr))
        e0, e1 = taps[0].axial_extent()      # opens on the mating face, so the extent is the drill depth
        D.each("fp_tap_drill_depth", [np.subtract(*c.axial_extent()[::-1]) for c in taps], p.tap_depth, _fmt)
        D.each("fp_tap_drill_d", [2 * c.r for c in taps], p.tap_drill_d, lambda v: f"φ{_fmt(v)}")
        sh.leader(aux.to_sheet(tr) + (1.2, 1.2), [(ab[1][0] + 4, ab[1][1] + 4), (ab[0][0] - 4, ab[1][1] + 4)], "",
                  lines=[f"4×M5×0.8 深さ{_fmt(fthr)}", f"  下穴φ{_fmt(2 * taps[0].r)} 深さ{_fmt(e1 - e0)}"])
        w_, h_ = sh.fcf((ab[0][0], ab[0][1] - 26), ["flatness", "0.05"])
        sh.text((ab[0][0], ab[0][1] - 32), "合わせ面 (図示面)", 3, 0, "left", "bottom")
        sh.roughness((ab[0][0] + w_ + 6, ab[0][1] - 26), RA_FACE)
    D.ra("mating_face", RA_FACE if len(taps) == 4 else None, RA_FACE)

    # ---- removed sections D-D, E-E (views placed last, in the space left over)
    _body_sections(sh, D, p, body, Sm, dv, cuts, k)

    # section label under everything drawn below the front view
    tbs = [text_box((it[1], it[2]), it[3], it[4], it[5], it[6], it[7]) for it in sh.items if it[0] == "text"]
    ys = [b0[1] for b0, b1 in tbs if b1[0] > fb[0][0] - 20 and b0[0] < fb[1][0] + 20 and 112 < b0[1] < fb[0][1]]
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


def _body_sections(sh, D, p, body, Sm, dv, cuts, k=1.0):
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
        v = View(f"{lab}-{lab}", faces[0], tuple(dv), (0, -1, 0), (0, 0), k).build()
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
        ww, hh = (b1 - b0) / k               # sheet extent back to model size
        wi, hi = p.body_w + (we - p.body_w) * t / p.length, p.body_h + (he - p.body_h) * t / p.length
        sh.dim((b0[0], b0[1]), (b1[0], b0[1]), -7, _fmt(ww))
        sh.dim((b1[0], b0[1]), (b1[0], b1[1]), -7, _fmt(hh), u=(0, 1))
        D.rec(f"section_{lab}_w", ww, wi, _fmt(ww)); D.rec(f"section_{lab}_h", hh, hi, _fmt(hh))


def _corners(pts, w, h, tol=0.01):
    """How many of the four pattern positions (±w, ±h) hold a hole axis; a hole
    off the rectangle, or two at one corner, leaves the count below four."""
    hit = {(sy, sn) for y, n in pts for sy in (-1, 1) for sn in (-1, 1)
           if abs(y - sy * w) < tol and abs(n - sn * h) < tol}
    return len(hit) if len(pts) == 4 else 0


def _clear_of_lines(sh, box, pad=0.5):
    """No drawn line (dimension, leader, centre line...) passes through the box."""
    (x0, y0), (x1, y1) = box
    for it in sh.items:
        if it[0] != "line" or it[2] == "frame":
            continue
        P = it[1]
        for a, b in zip(P[:-1], P[1:]):
            q = a + np.outer(np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.5) + 1)), b - a)
            if ((q[:, 0] > x0 - pad) & (q[:, 0] < x1 + pad) & (q[:, 1] > y0 - pad) & (q[:, 1] < y1 + pad)).any():
                return False
    return True


def _pts(e, n=24):
    return [np.array(tuple(e.position_at(k / n))) for k in range(n + 1)]


def _boxes_hit(a, b):
    return not (a[1][0] < b[0][0] or b[1][0] < a[0][0] or a[1][1] < b[0][1] or b[1][1] < a[0][1])


# ======================================================================= faceplate
def _window(fp, Bm, dv, nv, p):
    """Through window at the faceplate centre, measured on the solid: a line along d
    misses the faceplate inside the window. Returns (width, height, corner radius,
    centre offset across Y, centre offset along n) or None."""
    from build123d import Edge
    yv = np.array([0.0, 1.0, 0.0])

    def open_at(y, h):
        c = Bm + yv * y + nv * h
        return fp.distance_to(Edge.make_line(tuple(c - dv * 100), tuple(c + dv * 100))) > 1e-6

    if not open_at(0, 0):
        return None

    def edge(u, at=(0.0, 0.0), step=0.5):    # walk from `at` to the first wall along u (y, n), then bisect
        u, at = np.array(u, float), np.array(at, float); lo = 0.0
        while open_at(*(at + u * (lo + step))):
            lo += step
            if lo > 200:
                return float("inf")
        hi = lo + step
        for _ in range(30):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if open_at(*(at + u * m)) else (lo, m)
        return lo

    def span(axis, at):             # (+, -) extents along axis 0 (y) or 1 (n) through `at`
        u = np.eye(2)[axis]
        return edge(u, at), edge(-u, at)
    # each size is measured through the centre found on the other axis, so an off-centre
    # window is not read short at its rounded ends
    yp, ym = span(0, (0.0, 0.0)); cy = (yp - ym) / 2
    np_, nm = span(1, (cy, 0.0)); cn = (np_ - nm) / 2
    yp, ym = span(0, (0.0, cn)); cy = (yp - ym) / 2
    ww, wh = yp + ym, np_ + nm       # cy, cn: window centre relative to the bar axis
    rs = [c.r for c in find_cyl(fp, dv, inner=True)
          if abs((c.loc - Bm) @ yv - cy) < ww / 2 and abs((c.loc - Bm) @ nv - cn) < wh / 2]
    return ww, wh, (min(rs) if rs else 0.0), cy, cn


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
    bore = max(find_cyl(fp, (0, 1, 0), inner=True, near=tuple(Bm)), key=lambda c: c.r, default=None)
    if bore is None:
        return _unrecognised(p, size, "STEM_FACEPLATE", "ステム フェイスプレート (FACEPLATE)",
                             f"{dwg_no or doc_no(p)}-02", dict(bar_bore=False))
    # every view and dimension is laid out about the nominal bar axis Bm
    D.rec("bar_bore_axis_offset", math.hypot((bore.loc - Bm) @ dv, (bore.loc - Bm) @ nv), 0.0, "")

    def off_window(cs):
        # the window's end radii are cylinders too: one of the same radius is not a bolt hole
        if not (p.fp_window_w > 0 and p.fp_window_h > 0):
            return cs
        return [c for c in cs if abs((c.loc - Bm) @ np.array([0, 1, 0])) > p.fp_window_w / 2 + 0.01
                or abs((c.loc - Bm) @ nv) > p.fp_window_h / 2 + 0.01]
    holes = off_window(find_cyl(fp, dv, r=p.bolt_clear_d / 2, inner=True))
    cb = off_window(find_cyl(fp, dv, r=p.cbore_d / 2, inner=True))
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
    top_n = max(ext_n)
    sh.dim(ss(Bm + nv * top_n), ss(Bm + nv * top_n + dv * t_front), 8, _fmt(t_front))
    D.rec("front_thickness", t_front, p.fp_front, _fmt(t_front))
    sh.dim(ss(Bm + dv * t_back + nv * min(ext_n)), ss(Bm + dv * t_back + nv * top_n), 12, _fmt(hn_all), u=(0, 1))
    D.rec("height", hn_all, 2 * (p.fp_bolt_off + p.boss_margin), _fmt(hn_all))
    if t_back > 1e-3:          # zero gap: the mating face lies on the bar axis, nothing to dimension
        sh.dim(ss(Bm + nv * min(ext_n)), ss(Bm + nv * min(ext_n) + dv * t_back), -8, _fmt(t_back))
    D.rec("gap_half", t_back, p.fp_gap / 2, _fmt(t_back))
    sh.fcf(ss(Bm + dv * t_back + nv * min(ext_n)) + (-26, -16), ["flatness", "0.05"])
    sh.text(ss(Bm + dv * t_back + nv * min(ext_n)) + (-26, -21), "合わせ面", 3, 0, "left", "bottom")
    # mating-face texture on a leader to the face itself (an ear, clear of the bore), in free space
    for n_, off in ((min(ext_n) + 3, (-44, -10)), (top_n - 3, (-44, 10)), (min(ext_n) + 3, (-44, -22)),
                    (top_n - 3, (-44, 22)), (min(ext_n) + 3, (-22, -12))):
        tip = ss(Bm + dv * t_back + nv * n_)
        pos = tip + off
        sym = (pos + (-3, 0), pos + (6 + text_width(f"Ra {RA_FACE}", 3.5) + 1.5, 9.5))
        if sh.is_free(text_box(pos + (5.4, 7.6), f"Ra {RA_FACE}", 3.5, 0, "left", "bottom")) and \
                sh.is_free(sym) and _clear_of_lines(sh, sym):
            break
    sh.roughness_leader(tip, pos, RA_FACE)
    D.ra("mating_face", RA_FACE)
    # bore roughness, after the dimensions so it can avoid their lines: a leader from the
    # faceplate half of the bore (+d side of the bar axis) out through the open side
    for a_deg, off in ((-45, (-60, -18)), (45, (-60, 18)), (-60, (-48, -30)), (60, (-48, 30)), (-30, (-70, -8))):
        u = np.array([math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))])
        tip = ss(Bm) + u * scale * bore.r
        pos = tip + off
        sym = (pos + (-3, 0), pos + (6 + text_width(f"Ra {RA_BORE}", 3.5) + 1.5, 9.5))
        if sh.is_free(text_box(pos + (5.4, 7.6), f"Ra {RA_BORE}", 3.5, 0, "left", "bottom")) and \
                sh.is_free(sym) and _clear_of_lines(sh, sym):
            break
    sh.roughness_leader(tip, pos, RA_BORE)
    D.ra("bar_bore", RA_BORE)
    # front: width, bolt pattern, window
    lo_n = min(ext_n)
    sh.dim(fs(Bm + np.array([0, y0, 0]) + nv * lo_n), fs(Bm + np.array([0, y1, 0]) + nv * lo_n), -10, _fmt(y1 - y0))
    D.rec("width", y1 - y0, p.bar_clamp_w, _fmt(y1 - y0))
    for c in holes:
        sh.center_mark(fs(c.loc), scale * c.r, 2)
    D.need("bolt_holes", len(holes), 4); D.need("cbores", len(cb), 4)
    D.need("bolt_pattern", _corners([(c.loc[1] - Bm[1], (c.loc - Bm) @ nv) for c in holes], p.fp_bolt_w, p.fp_bolt_off), 4)
    # the 4x callout says every counterbore is concentric with a clearance hole: check their axes too
    D.need("cbore_pattern", _corners([(c.loc[1] - Bm[1], (c.loc - Bm) @ nv) for c in cb], p.fp_bolt_w, p.fp_bolt_off), 4)
    D.each("bolt_clear_d", [2 * c.r for c in holes], p.bolt_clear_d, lambda v: f"φ{_fmt(v)}")   # all four

    def span_d(c):                       # extent along d, whichever way the cylinder axis points
        e = np.array(c.axial_extent()) * np.sign(c.axis @ dv)
        return e.min() - Bm @ dv, e.max() - Bm @ dv
    # "キリ" with no depth = through: each hole runs from the mating face to its counterbore bottom
    for i, c in enumerate(holes):
        h0, h1 = span_d(c)
        mate = [o for o in cb if np.linalg.norm(np.cross(o.loc - c.loc, dv)) < 0.05]
        sfx = "" if i == 0 else f"#{i + 1}"
        D.rec("bolt_hole_from_mating_face" + sfx, h0 - t_back, 0.0, "")
        D.rec("bolt_hole_to_cbore" + sfx, (span_d(mate[0])[0] - h1) if mate else float("inf"), 0.0, "")
    D.each("cbore_d", [2 * c.r for c in cb], p.cbore_d, lambda v: f"φ{_fmt(v)}")
    if len(holes) == 4:
        q = np.array([[c.loc[1], (c.loc - Bm) @ nv] for c in holes])
        wy, hn = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        tl = holes[int(np.argmax(-q[:, 0] + q[:, 1]))].loc      # top-left hole
        tr = tl + np.array([0, wy, 0])
        sh.dim(fs(tl), fs(tr), (top_n - (tl - Bm) @ nv) * scale + 8, _fmt(wy))
        sh.dim(fs(tr - nv * hn), fs(tr), -((y1 - tr[1]) * scale + 10), _fmt(hn), u=(0, 1))
        D.rec("bolt_pitch_y", wy, 2 * p.fp_bolt_w, _fmt(wy)); D.rec("bolt_pitch_n", hn, 2 * p.fp_bolt_off, _fmt(hn))
        cb_depth = t_front - (cb[0].axial_extent()[0] - Bm @ dv) if cb else 0.0
        D.each("cbore_depth", [t_front - (c.axial_extent()[0] - Bm @ dv) for c in cb] or [0.0], 2.5, _fmt)
        fb_ = front.bbox()
        sh.leader(fs(tl) + np.array([-1, 1]) * scale * holes[0].r * 0.7,
                  [fs(Bm) + (-75, 62), (fb_[0][0] - 10, fb_[1][1] + 10), (fb_[0][0] - 30, fb_[1][1] + 18),
                   (fb_[0][0] + 10, fb_[1][1] + 22)], "", right=True,
                  lines=[f"4×φ{_fmt(2 * holes[0].r)} キリ", f"φ{_fmt(2 * cb[0].r) if cb else '-'} 深ザグリ 深さ{_fmt(cb_depth)}"])
    win = _window(fp, Bm, dv, nv, p)
    want_win = p.fp_window_w > 0 and p.fp_window_h > 0     # stem.build() cuts a window only then
    if win or want_win:
        ww, wh, wr, wcy, wcn = win or (0.0, 0.0, 0.0, 0.0, 0.0)
        if win:
            fb_ = front.bbox()
            sh.leader(fs(Bm - np.array([0, ww / 2, 0])),
                      [fs(Bm) + (-60, -48), (fb_[0][0] - 8, fb_[0][1] - 10), (fb_[0][0] - 20, fb_[0][1] - 20)], "", right=False,
                      lines=[f"窓 {_fmt(ww)}×{_fmt(wh)}", f"R{_fmt(wr)} 貫通"])
        D.rec("window_w", ww, p.fp_window_w if want_win else 0.0, _fmt(ww))
        D.rec("window_h", wh, p.fp_window_h if want_win else 0.0, _fmt(wh))
        # full-round slot: the end radius is half the smaller size (stem._window_profile)
        D.rec("window_r", wr, min(p.fp_window_w, p.fp_window_h) / 2 if want_win else 0.0, _fmt(wr))
        # the drawing shows the window centred on the bar axis: equal +/- extents
        D.rec("window_center_y", wcy, 0.0, ""); D.rec("window_center_n", wcn, 0.0, "")
    mass = fp.volume / 1000 * stem.density(p)
    _title(sh, p, "ステム フェイスプレート (FACEPLATE)", f"{dwg_no or doc_no(p)}-02", mass, scale=_scale_text(scale))
    _notes(sh, _common_notes(sh, p, [f"※ φ{_fmt(2 * bore.r)} は本体と組み付けた状態で仕上げる (共加工)。",
                                 f"本体との合わせ面のすき間 {_fmt(2 * t_back)} (締結前、参考)。",
                                 "図示なき外形形状は3Dモデル (STEP AP242、同図番) による。"]), (16, 106))
    _general_tol_table(sh, (16, 36))
    sh.meta["part"] = "STEM_FACEPLATE"
    return sh
