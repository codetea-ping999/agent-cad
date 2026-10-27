"""Fastener calculations (締結計算書) of the project's bolted joints.

    rep = stem_joints(StemParams(), HandlebarParams())    # steerer pinch + faceplate, bar interface
    rep = bracket_joints(BracketParams())                 # M8 through bolts of the die-cast bracket
    rep = housing_screws(HousingParams())                 # thread-forming screws in the resin bosses

Each joint is a VDI 2230 calculation (agent_cad/fastener.py) whose geometry is read
from the part parameters, so a parameter change re-runs the calculation. What each
joint must hold:

  steerer pinch    2 x M5 close the slotted clamp ring on the fork steerer; friction
                   must carry the stem-to-steerer torsion test torque.
  faceplate        4 x M5 pull the faceplate onto the bar; friction must carry the
                   handlebar-to-stem torsion test torque and, without slipping, the peak
                   bar torque of the in-phase hood fatigue case of the handlebar
                   verification. The forward pull of the static hood case is the
                   bolts' working load. The bar's clamp section must not yield under
                   the clamp pressure.
  bracket          4 x M8 through bolts with nuts on a steel mount (assumed): no slip
                   under the hub load, no separation under its moment.

The test torques are PROVISIONAL values in the spirit of ISO 4210-5 (handlebar-stem and
stem-steerer torsional security tests), like the other ISO 4210 loads of this project:
take the exact values from the purchased standard before a release. Friction
coefficients and the embedding allowance are VDI 2230 table values; the tightening
torques checked are the ones the drawings state (TORQUE_SPEC).
"""
import math
from dataclasses import asdict
from ..fastener import BOLTS, Plate, Joint, calc, EMBED_UM, E_STEEL
from ..materials import MATERIALS
from ..parts import stem as stem_mod

# tightening torques the drawings state (agent_cad/parts/stem_drawing.py note), N·m per bolt
# (the bracket's is a recommendation of this calculation: its drawing states none yet)
# faceplate 4-5 -> 5-6 N·m in AC-1001 rev A (2026-10-27): 4 N·m with the highest friction
# left too little clamp force for the bar torque
TORQUE_SPEC = {"pinch": (5.0, 6.0), "faceplate": (5.0, 6.0), "bracket": (18.0, 22.0)}
# bolts the calculation assumes: 10.9, not 12.9, for the M5s: in the A6061 stem the
# pinch thread (7.5 mm engaged) would strip just before a 12.9 bolt breaks (0.997)
BOLT_SPEC = {"pinch": "M5-10.9", "faceplate": "M5-10.9", "bracket": "M8-8.8"}



def torque_notes():
    """The fastener notes of the stem body drawing, from TORQUE_SPEC and BOLT_SPEC (one
    source for the drawing and this calculation)."""
    (a, b), (c, d) = TORQUE_SPEC["pinch"], TORQUE_SPEC["faceplate"]
    grades = sorted({BOLTS[BOLT_SPEC[k]].grade for k in ("pinch", "faceplate")})
    return [f"締付トルク: ピンチボルト {a:g}〜{b:g} N·m、フェイスプレートボルト {c:g}〜{d:g} N·m (締結計算書 AC-CALC-F01)。",
            f"締結ボルト: 六角穴付きボルト M5×0.8 (JIS B 1176) 強度区分 {'/'.join(grades)}。"]


# PROVISIONAL ISO 4210-5 style torsional security torques (N·m)
T_STEERER_TEST = 40.0         # stem on the fork steerer, about the steerer axis
T_BAR_TEST = 60.0             # handlebar in the stem, about the bar axis
SG_SLIP_STATIC = 1.2          # VDI 2230: slip safety under static transverse load
SG_SLIP_FATIGUE = 1.0         # no slip at the peak of the (provisional) fatigue load
MU_CLAMP = 0.15               # aluminium clamp on aluminium bar / steerer, clean, no grease (minimum)
FP_CBORE_DEPTH = 2.5          # faceplate counterbore depth (agent_cad/parts/stem.py build())
BAR_SF_YIELD = 1.2            # bar clamp section, hoop compression under the clamp pressure

# bracket: service load at the hub (assumed until the vehicle load is known)
BRACKET_HUB_LOAD_N = 2000.0   # radial load on the hub, transverse to the bolts
BRACKET_MOUNT_T = 6.0         # steel mount plate under the bracket (assumed)
SG_SLIP_DYNAMIC = 1.8         # VDI 2230: slip safety under alternating transverse load


def _slotted_ring_compliance(r_mean, t, w, E):
    """Closure of the slot per unit bolt force: the ring is a hoop of length 2 pi r_m with
    section t x w (one bolt's share of the clamp height). The fork steerer's own radial
    compliance is left out (stiffer clamp side: more of the preload is lost to embedding,
    the conservative direction)."""
    return 2 * math.pi * r_mean / (E * t * w)


def _stem_fatigue_N(p):
    from .stem_verify import LOAD_CASES
    return max(abs(min(lc.force_N)) for lc in LOAD_CASES if lc.kind == "fatigue")


def pinch_joint(p, bolt=None, torque=None):
    mat = MATERIALS[p.material]
    b = BOLTS[bolt or BOLT_SPEC["pinch"]]
    n = 2
    r = p.steerer_d / 2
    # torque the friction must carry: T = mu * N * r with N = pi * F_sum (pressure over the
    # ring, half the ideal thin-ring value 2 pi F to allow for a stiff clamp that does not
    # wrap fully)
    F_req = SG_SLIP_STATIC * T_STEERER_TEST * 1000 / (MU_CLAMP * math.pi * r) / n
    t_ear = p.ear_half_w - p.cbore_depth - p.slot_w / 2       # clearance-side ear under the head
    DA = 2 * (p.cbore_d / 2 + p.ear_end_margin)                # ear boss around the bolt
    ring = _slotted_ring_compliance(r + p.steerer_wall / 2, p.steerer_wall, p.stack_h / n, mat.E_MPa)
    e = EMBED_UM["10-40"]["inner"]
    j = Joint("ステアリングクランプ (ピンチボルト)", b, n, torque or TORQUE_SPEC["pinch"],
              (Plate(t_ear, mat.E_MPa, DA, "ピンチ耳 (クリアランス側)"),), p.bolt_clear_d, True,
              stem_mod.thread_depths(p)["pinch"], mat.tensile_MPa, mat.tensile_MPa, thread_E_MPa=mat.E_MPa,
              gap_mm=p.slot_w, extra_compliance=ring,
              # the steerer contact embeds radially; the slot closes by 2 pi times that
              embed_extra_um=2 * math.pi * e,
              # the stem's forward-bending fatigue moment at the steerer clamp, as a couple over the
              # clamp height taken by one bolt (upper bound of the slot-opening load)
              FA_alt_N=_stem_fatigue_N(p) * p.length / p.stack_h,
              F_req_N=F_req,
              F_req_basis=f"ステアラ ねじり {T_STEERER_TEST:g} N·m (暫定) × SG {SG_SLIP_STATIC} / (μ {MU_CLAMP} · π · r {r:g} mm) / {n}本")
    return j


def faceplate_clamp_compliance(p, faceplate=None, h=1.5):
    """Compliance of the faceplate as the bolt sees it (mm/N per bolt), by FEA: the bar
    bore is held (rigid bar: the bar's own ring compliance is left out, the conservative
    direction), each counterbore bottom carries the bolt force toward the body, and the
    mean axial displacement of the counterbore bottoms over the force is the compliance.
    It includes the shell bending between the bolt pairs that the VDI cone leaves out."""
    import numpy as np
    from ..fea.static import mesh_shape, displacement
    if faceplate is None:
        _, faceplate = stem_mod.build(p)
    d, n_, S, B = stem_mod.frame(p)
    dv, nv, Bv = np.array([d.X, d.Y, d.Z]), np.array([n_.X, n_.Y, n_.Z]), np.array([B.X, B.Y, B.Z])
    mesh = mesh_shape(faceplate, h)
    u_cb = p.fp_front - FP_CBORE_DEPTH
    bolts = [(-sv * p.fp_bolt_off, sw * p.fp_bolt_w) for sv in (-1, 1) for sw in (-1, 1)]   # (along n, along Y)

    def local(c):
        rel = c - Bv[:, None]
        return dv @ rel, nv @ rel, rel[1]

    def fixed(c):
        a, b_, _ = local(c)
        return np.abs(np.hypot(a, b_) - p.bar_d / 2) < 0.3

    def cbore(k):
        def sel(c):
            a, b_, y = local(c)
            r = np.hypot(b_ - bolts[k][0], y - bolts[k][1])
            return (np.abs(a - u_cb) < 0.05) & (r > p.bolt_clear_d / 2) & (r < p.cbore_d / 2)
        return sel
    F = 1000.0
    ib, u = displacement(mesh, MATERIALS[p.material], fixed, [(cbore(k), tuple(-F * dv)) for k in range(4)])
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    nodes = np.unique(np.concatenate([mesh.facets[:, bf[cbore(k)(fc)]].ravel() for k in range(4)]))
    U = np.array([u[ib.nodal_dofs[k]] for k in range(3)])[:, nodes]
    return float(np.mean(-(dv @ U))) / F


def faceplate_joint(p, bar=None, bolt=None, torque=None, fea_compliance=None):
    mat = MATERIALS[p.material]
    b = BOLTS[bolt or BOLT_SPEC["faceplate"]]
    n = 4
    r = p.bar_d / 2
    need = bar_torque_demand(p, bar)
    T_need = max(need["test_Nm"] * SG_SLIP_STATIC, need["fatigue_peak_Nm"] * SG_SLIP_FATIGUE,
                 need.get("static_peak_Nm", 0.0) * SG_SLIP_FATIGUE)
    # two half shells pressed together by F_sum with a cosine pressure: N = (8 / pi) F_sum
    F_req = T_need * 1000 / (MU_CLAMP * 8 / math.pi * r) / n
    t_fp = p.fp_front - p.fp_gap / 2 - FP_CBORE_DEPTH
    DA = 2 * p.boss_margin
    j = Joint("フェイスプレート (ハンドルクランプ)", b, n, torque or TORQUE_SPEC["faceplate"],
              (Plate(t_fp, mat.E_MPa, DA, "フェイスプレート"),), p.bolt_clear_d, True,
              stem_mod.thread_depths(p)["faceplate"], mat.tensile_MPa, mat.tensile_MPa, thread_E_MPa=mat.E_MPa,
              gap_mm=p.fp_gap,
              # the faceplate shell: FEA compliance less the VDI cone already counted
              extra_compliance=max(0.0, (fea_compliance or 0.0) - _cone_only(b, t_fp, mat.E_MPa, DA, p.bolt_clear_d)),
              interfaces=2,                                    # faceplate / bar and bar / body contacts
              FA_static_N=need["forward_pull_N"] / n,
              FA_alt_N=need["fatigue_pull_N"] / n,
              F_req_N=F_req,
              F_req_basis=(f"ハンドル回転 max(試験 {need['test_Nm']:g} N·m (暫定) × SG {SG_SLIP_STATIC}, "
                           f"疲労ピーク {need['fatigue_peak_Nm']:.0f} N·m, 静的フード {need.get('static_peak_Nm', 0.0):.0f} N·m × SG {SG_SLIP_FATIGUE}) / "
                           f"(μ {MU_CLAMP} · 8/π · r {r:g} mm) / {n}本"))
    return j, need


def _cone_only(b, t, E, DA, dh):
    from ..fastener import plate_compliance
    j = Joint("", b, 1, (1, 1), (Plate(t, E, DA),), dh, True, 1.0, 1.0, 1.0)
    return plate_compliance(j)[0]


def bar_torque_demand(p, bar=None):
    """Bar torques and pulls at the stem from the handlebar's load cases: the provisional
    torsion test torque, the peak torque of the in-phase hood fatigue case (both hands
    push down on the hoods ahead of the clamp), and the forward pull of the static hood
    case (taken by the faceplate bolts)."""
    out = {"test_Nm": T_BAR_TEST, "fatigue_peak_Nm": 0.0, "forward_pull_N": 0.0, "fatigue_pull_N": 0.0}
    if bar is None:
        return out
    import numpy as np
    from ..parts import handlebar as hb
    from . import handlebar_verify as hv
    sol = hb.solve(bar)
    segs = {g.name: g for g in hb.path(bar, 1, sol)}
    nm, s = hv.load_points(bar, sol)["hoods"]
    q = np.array(segs[nm].point(s)[0]) + np.array(hv.HOOD_OFFSET)
    for lc in hv.LOAD_CASES:
        if lc.where != "hoods":
            continue
        F = np.array(lc.force_N)
        My = q[2] * F[0] - q[0] * F[2]                        # about the bar axis (Y), one side
        both = 2 if lc.phase > 0 else 0                        # out of phase: the two sides cancel
        if lc.kind == "fatigue":
            out["fatigue_peak_Nm"] = max(out["fatigue_peak_Nm"], float(abs(My)) * both / 1000)
            out["fatigue_pull_N"] = max(out["fatigue_pull_N"], float(max(F[0], 0.0)) * 2)
        else:
            out["forward_pull_N"] = max(out["forward_pull_N"], float(max(F[0], 0.0)) * 2)
            out["static_peak_Nm"] = max(out.get("static_peak_Nm", 0.0), float(abs(My)) * both / 1000)
    out["hood_point_mm"] = [round(float(c), 1) for c in q]
    return out


def _rule(name, ok, value, limit, basis, **kw):
    return {"check": name, "pass": bool(ok), "value": value, "limit": limit, "basis": basis, **kw}


def stem_joints(p, bar=None, bolt=None, fea=True, faceplate=None):
    """Steerer pinch and faceplate joints of a stem, and the bar under the faceplate.
    fea: the faceplate's clamp compliance from FEA (otherwise the VDI cone alone, which
    overstates the preload lost to embedding)."""
    pj = pinch_joint(p, bolt)
    cf = faceplate_clamp_compliance(p, faceplate) if fea else None
    fj, need = faceplate_joint(p, bar, bolt, fea_compliance=cf)
    pr, fr = calc(pj), calc(fj)
    # what the friction holds at the minimum residual clamp force
    r_s, r_b = p.steerer_d / 2, p.bar_d / 2
    pr["capacity"] = {"steerer_torque_Nm": round(MU_CLAMP * math.pi * pj.n * pr["preload"]["F_residual_min"] * r_s / 1000, 1),
                      "required_Nm": T_STEERER_TEST, "slip_safety": None}
    pr["capacity"]["slip_safety"] = round(pr["capacity"]["steerer_torque_Nm"] / T_STEERER_TEST, 2)
    # steerer: clamp pressure at the highest preload (thin ring: p = F_sum / (r w))
    p_st = pj.n * pr["preload"]["FM_max"] / (r_s * p.stack_h)
    pr["steerer_pressure_MPa"] = round(p_st, 1)
    capf = MU_CLAMP * 8 / math.pi * fj.n * fr["preload"]["F_residual_min"] * r_b / 1000
    fr["capacity"] = {"bar_torque_Nm": round(capf, 1), "test_Nm": need["test_Nm"],
                      "fatigue_peak_Nm": round(need["fatigue_peak_Nm"], 1),
                      "static_hood_peak_Nm": round(need.get("static_peak_Nm", 0.0), 1),
                      "slip_safety_test": round(capf / need["test_Nm"], 2),
                      "slip_safety_fatigue": round(capf / need["fatigue_peak_Nm"], 2) if need["fatigue_peak_Nm"] else None,
                      "hood_point_mm": need.get("hood_point_mm")}
    interface = None
    if bar is not None:
        bm = MATERIALS[bar.material]
        F_sum = fj.n * fr["preload"]["FM_max"]
        p_avg = F_sum / (p.bar_d * p.bar_clamp_w)              # projected
        p_peak = 4 / math.pi * p_avg                           # cosine distribution
        s_hoop = p_peak * (bar.clamp_d / 2) / bar.wall_clamp   # thin-wall ring under external pressure
        allow = bm.yield_MPa / BAR_SF_YIELD
        interface = {"bar": f"{bar.material} φ{bar.clamp_d:g} × t{bar.wall_clamp:g}",
                     "clamp_force_max_N": round(F_sum, 0), "pressure_avg_MPa": round(p_avg, 1),
                     "pressure_peak_MPa": round(p_peak, 1), "hoop_stress_MPa": round(s_hoop, 0),
                     "rules": [
                         _rule("bar.clamp_hoop_stress", s_hoop <= allow, round(s_hoop, 0), round(allow, 0),
                               f"p_peak r / t <= yield {bm.yield_MPa:g} / {BAR_SF_YIELD} (F_M,max of all 4 bolts)"),
                         _rule("bar.clamp_fits", bar.clamp_d == p.bar_d, bar.clamp_d, p.bar_d, "bar clamp diameter = stem bore"),
                         _rule("bar.clamp_zone", bar.clamp_zone_w >= p.bar_clamp_w, bar.clamp_zone_w, p.bar_clamp_w,
                               "full-diameter zone of the bar >= stem clamp width")]}
        interface["pass"] = all(r["pass"] for r in interface["rules"])
    fr["compliance"]["faceplate_fea_mm_per_N"] = cf
    joints = {"pinch": pr, "faceplate": fr}
    ok = all(r["pass"] for r in joints.values()) and (interface is None or interface["pass"])
    return {"params": stem_mod.asdict(p), "bar": asdict(bar) if bar else None, "joints": joints,
            "bar_interface": interface, "assumptions": assumptions(),
            "pass": ok, "failed": [f"{k}.{c}" for k, r in joints.items() for c in r["failed"]] +
            ([r["check"] for r in interface["rules"] if not r["pass"]] if interface else [])}


def bracket_joints(p, bolt=None, torque=None, hub_load_N=BRACKET_HUB_LOAD_N):
    """M8 through bolts with nuts clamping the bracket's bosses to a steel mount."""
    mat = MATERIALS[p.material]
    b = BOLTS[bolt or BOLT_SPEC["bracket"]]
    n = 4
    # transverse: the hub load shared by the four bolts, held by friction
    F_slip = SG_SLIP_DYNAMIC * hub_load_N / (n * MU_CLAMP)
    # moment of the hub load about the mounting face, carried by the bolt pair on one side
    M = hub_load_N * p.hub_h
    FA = M / p.boss_pitch_z / 2
    j = Joint("ブラケット取付 (M8 通しボルト)", b, n, torque or TORQUE_SPEC["bracket"],
              (Plate(p.boss_h, mat.E_MPa, p.boss_d, "ブラケットボス (ADC12)"),
               Plate(BRACKET_MOUNT_T, E_STEEL, 3 * b.d, "相手板 (鋼, 仮定)")),
              p.hole_d, False, 0.8 * b.d, 400, mat.tensile_MPa, interfaces=1,
              FA_static_N=FA, FA_alt_N=FA, F_req_N=F_slip,
              F_req_basis=f"ハブ荷重 {hub_load_N:g} N (仮定) × SG {SG_SLIP_DYNAMIC} / (μ {MU_CLAMP} · {n}本)")
    r = calc(j)
    return {"params": asdict(p), "joints": {"bracket": r}, "assumptions": assumptions(),
            "pass": r["pass"], "failed": [f"bracket.{c}" for c in r["failed"]]}


# thread-forming screws for glass-filled polyamide: design guide values of screw makers
# (hole ~0.8 d, boss >= ~1.8 d, engagement ~2 d); the tightening torque must be set from
# the drive and stripping torques measured on moulded parts
PT_GUIDE = {"PA66-GF30": dict(hole=(0.78, 0.86), boss_min=1.8, engage_min=1.7),
            "PA6-GF30": dict(hole=(0.78, 0.86), boss_min=1.8, engage_min=1.7),
            "PBT-GF30": dict(hole=(0.78, 0.86), boss_min=1.8, engage_min=1.7),
            "PP": dict(hole=(0.68, 0.75), boss_min=2.0, engage_min=2.0)}


def housing_screws(p, d=4.0):
    g = PT_GUIDE[p.material]
    hr, br, er = p.hole_d / d, p.boss_d / d, (p.hole_depth - 1.0) / d    # 1 mm lead-in not engaged
    rules = [
        _rule("hole_ratio", g["hole"][0] <= hr <= g["hole"][1], round(hr, 3), list(g["hole"]),
              "下穴径 / 呼び径 (ねじメーカー設計ガイドの目安)"),
        _rule("boss_ratio", br >= g["boss_min"], round(br, 3), g["boss_min"], "ボス外径 / 呼び径"),
        _rule("engagement_ratio", er >= g["engage_min"], round(er, 3), g["engage_min"], "有効ねじ込み長 / 呼び径"),
    ]
    return {"params": asdict(p), "screw": f"樹脂用タッピンねじ 呼び{d:g}", "rules": rules,
            "torque_note": "締付トルクは成形品で ねじ込みトルク M_E と破断(空転)トルク M_Ü を実測し、"
                           "M_A ≈ M_E + 0.3〜0.5 (M_Ü − M_E) で決める (図面に記載する前に試験が必要)。",
            "pass": all(r["pass"] for r in rules), "failed": [r["check"] for r in rules if not r["pass"]]}


def assumptions():
    return [
        "ねじ部・座面の摩擦係数 μ_G, μ_K = 0.10〜0.16 (VDI 2230 表A5 クラスB: 黒染/亜鉛めっき鋼, 油・グリース塗布)。",
        "締付トルクのばらつきは図面記載の範囲 (下限〜上限) で考慮し、締付係数 α_A = F_M,max / F_M,min を算出。",
        f"クランプ面の摩擦係数 μ = {MU_CLAMP} (アルミ同士, 脱脂・無潤滑の下限値)。グリースを塗る場合は低下するため不可。",
        "なじみ量 f_Z は VDI 2230 表5 (Rz 10〜40 μm)。ステアラ側のなじみはスリットの閉じ量 (2π倍) に換算。",
        f"ISO 4210-5 相当のねじり試験トルク (ステアラ {T_STEERER_TEST:g} N·m, ハンドル {T_BAR_TEST:g} N·m) は暫定値。規格本文で確認のこと。",
        "ハンドルの疲労ピークトルクは handlebar_verify の同相フード疲労荷重 (暫定) から算出。",
        "ねじ山せん断強さ τ_B = 0.6 Rm (アルミ), 座面許容面圧 p_G = 0.9 Rm (VDI 表A9 の代用, 安全側)。",
        "ボルト疲労限度は転造後熱処理ねじ σ_ASV = 0.85 (150/d + 45) MPa (VDI 2230 R9)。",
    ]
