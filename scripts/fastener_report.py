"""Fastener calculation reports (締結計算書) -> out/fastener/.

    python -m scripts.fastener_report            # every case; exit 1 if a joint fails

For the stem cases of scripts/make_stem.py (with the handlebar of scripts/make_handlebar.py
in the bar clamp), the bracket and the housing: a JSON with every intermediate value and
a PDF calculation sheet in Japanese. The stem's tightening torques are the ones its
drawing states (agent_cad/verify/fastener_check.py TORQUE_SPEC).
"""
import sys, os, json, datetime
from agent_cad.verify import fastener_check as fc
from agent_cad.fastener import BOLTS
from agent_cad.report import Report
from agent_cad.parts import stem
from scripts import make_stem, make_diecast, make_molding, make_handlebar

OUT = os.path.join("out", "fastener")
BAR_CASE = "drop420_7075"
NUMBERS = {"stem90_6061": "AC-1001 / AC-1002 (+ AC-1301)", "stem110_-17_7075": "ステム 110 mm A7075 (図番未登録)",
           "bracket_std": "AC-1101", "housing_std": "AC-1201"}
OK, NG = "合格", "不合格"
RULE_JP = {"assembly_stress": "締付時ボルト応力 (ν)", "service_stress": "使用時ボルト応力", "clamp_force": "残留締付力",
           "fatigue": "ボルト疲労", "head_bearing": "座面圧", "thread_strip": "雌ねじせん断 (ボルト破断が先)",
           "bar.clamp_hoop_stress": "ハンドル挟み部 周方向応力", "bar.clamp_fits": "ハンドル径 = クランプ径",
           "bar.clamp_zone": "ハンドル平行部幅 ≧ クランプ幅",
           "hole_ratio": "下穴径比", "boss_ratio": "ボス径比", "engagement_ratio": "ねじ込み長比"}


def _f(v, nd=0):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " 〜 ".join(_f(x, nd) for x in v)
    if v == float("inf"):
        return "∞"
    return f"{v:,.{nd}f}"


def _hl(row):
    return "#fde2e2" if row[-1] == NG else None


def joint_section(doc, key, r, extra=None):
    g, c, pl, st, lf, tw = r["geometry"], r["compliance"], r["preload"], r["stress"], r["load_factor"], r["torque_window_Nm"]
    doc.h2(f"{r['joint']}  ({r['bolt']} × {r['n_bolts']} 本, 締付トルク {_f(r['torque_Nm'], 1)} N·m)")
    doc.table(["項目", "記号", "値", "備考"], [
        ["有効径 / 谷径", "d2 / d3", f"{g['d2']:.3f} / {g['d3']:.3f} mm", "JIS B 0205"],
        ["有効断面積", "As", f"{g['As']:.2f} mm²", "JIS B 1082"],
        ["座面外径 / 穴径", "dw / dh", f"{g['dw']:g} / {g['dh']:g} mm", "座面摩擦径 D_Km = " + f"{g['D_Km']:g} mm"],
        ["締結部材厚 / 隙間", "lK / s", f"{g['l_K']:g} / {g['gap']:g} mm",
         "、".join(f"{p['name']} t{p['t']:g}" for p in g["plates"])],
        ["はめあい長さ", "m", f"{g['engagement']:g} mm", ""],
        ["ボルトばね定数", "kS", f"{c['k_S_kN_mm']:g} kN/mm", "VDI 2230 R3 (頭部・自由ねじ部・はめあい部)"],
        ["被締結側ばね定数", "kP", f"{c['k_P_kN_mm']:g} kN/mm",
         f"{c['plate'].get('model', '')} (tanφ {c['plate'].get('tan_phi', 0):.3f})"
         + (f" + 追加コンプライアンス {c['plate']['extra'] * 1e6:.2f} μm/kN" if c["plate"].get("extra") else "")],
        ["内力係数", "Φn", f"{lf['Phi_n']:.3f}", f"Φk {lf['Phi_K']:.3f} × 負荷導入係数 n {lf['n']:g}"],
        ["摩擦係数 (ねじ / 座面)", "μG / μK", f"{_f(r['mu_G'], 2)} / {_f(r['mu_K'], 2)}", "VDI 2230 表A5 クラスB"],
        ["トルク係数", "K", f"{pl['K_min']:.3f} 〜 {pl['K_max']:.3f}", "JIS B 1083: T = K d F"],
        ["締付軸力", "F_M", f"{_f(pl['FM_min'])} 〜 {_f(pl['FM_max'])} N", f"締付係数 αA = {pl['alpha_A']:.2f}"],
        ["許容締付軸力", "F_M,zul", f"{_f(pl['FM_zul'])} N", "ν = 0.9 (VDI 2230 R7)"],
        ["なじみ量 / 軸力低下", "fZ / FZ", f"{pl['fZ_um']:.1f} μm / {_f(pl['FZ'])} N", "VDI 2230 表5"],
        ["作用軸力 (静的 / 変動)", "FA", f"{_f(r['loads']['FA_static'])} / {_f(r['loads']['FA_alt'])} N", "1 本あたり"],
        ["必要締付力", "F_Kerf", f"{_f(r['loads']['F_req'])} N", r["loads"]["F_req_basis"]],
        ["最小残留締付力", "F_KR,min", f"{_f(pl['F_residual_min'])} N", "F_M,min − FZ − (1 − Φn) FA"],
        ["締付時相当応力", "σred,M", f"{_f(st['sigma_red_M'])} MPa", f"σ {_f(st['sigma_M'])} / τ {_f(st['tau_M'])} MPa"],
        ["応力振幅 / 疲労限度", "σa / σASV", f"{_f(st['sigma_a'], 1)} / {_f(st['sigma_ASV'], 1)} MPa", "VDI 2230 R9"],
        ["座面圧 / 許容", "p / pG", f"{_f(st['p_head'])} / {_f(st['p_G'])} MPa", "VDI 2230 R10"],
        ["雌ねじせん断力 / ボルト破断力", "F_mGM / F_mS", f"{_f(st['F_strip'])} / {_f(st['F_break'])} N", "VDI 2230 R11"],
        ["成立する締付トルク範囲", "MA", f"{tw['min']:.2f} 〜 {tw['max']:.2f} N·m" + ("" if tw["feasible"] else " (成立せず)"),
         "下限: 必要締付力 (μ最大)、上限: ν = 0.9 (μ最小)"],
    ], widths=[3.2, 1.7, 3.0, 5.6])
    rows = [[RULE_JP.get(x["check"], x["check"]), _f(x["value"], 2 if isinstance(x["value"], float) and x["value"] < 10 else 0),
             _f(x["limit"], 2 if isinstance(x["limit"], float) and x["limit"] < 10 else 0), OK if x["pass"] else NG]
            for x in r["rules"]]
    doc.table(["判定項目", "計算値", "許容 / 必要", "判定"], rows, widths=[5, 2.5, 2.5, 1.5], highlight=_hl)
    if extra:
        for t in extra:
            doc.para(t)


def stem_case(name):
    p = make_stem.CASES[name]
    bar = make_handlebar.CASES[BAR_CASE]
    rep = fc.stem_joints(p, bar)
    alt = fc.calc(fc.pinch_joint(p, bolt="M5-12.9"))
    rep["alternatives"] = {"pinch_with_12.9": {"pass": alt["pass"], "failed": alt["failed"],
                                               "F_strip": alt["stress"]["F_strip"], "F_break": alt["stress"]["F_break"]}}
    return rep


def write_stem(doc, name, rep):
    p = make_stem.CASES[name]
    pj, fj = rep["joints"]["pinch"], rep["joints"]["faceplate"]
    doc.h1(f"ステム {p.length:g} mm / {p.angle_deg:g}° / {p.material}  ({NUMBERS.get(name, '')})")
    cap_p, cap_f = pj["capacity"], fj["capacity"]
    joint_section(doc, "pinch", pj, [
        f"摩擦で伝えられるステアラまわりトルク (最小残留締付力時): {cap_p['steerer_torque_Nm']:g} N·m "
        f"(要求 {cap_p['required_Nm']:g} N·m, 安全率 {cap_p['slip_safety']:g})。",
        f"ステアラ面圧 (最大締付時, 薄肉リング近似): {rep['joints']['pinch']['steerer_pressure_MPa']:g} MPa。"
        "カーボンコラムの場合はメーカー許容値 (締付トルク上限) を優先すること。",
        "12.9 ボルトの場合: " + ("全項目合格。" if rep["alternatives"]["pinch_with_12.9"]["pass"] else
                               f"雌ねじせん断 {rep['alternatives']['pinch_with_12.9']['F_strip']:,.0f} N < ボルト破断 "
                               f"{rep['alternatives']['pinch_with_12.9']['F_break']:,.0f} N (ねじ山が先に抜ける) → 10.9 を指定。")])
    joint_section(doc, "faceplate", fj, [
        f"摩擦で伝えられるハンドル回転トルク (最小残留締付力時): {cap_f['bar_torque_Nm']:g} N·m。"
        f" 要求: 試験 {cap_f['test_Nm']:g} N·m (暫定, 安全率 {cap_f['slip_safety_test']:g})、"
        f"フード同相疲労ピーク {cap_f['fatigue_peak_Nm']:g} N·m (安全率 {cap_f['slip_safety_fatigue']:g})、"
        f"静的フード前方荷重 {cap_f['static_hood_peak_Nm']:g} N·m。",
        f"フェイスプレートのたわみ (FEA, 1 本あたり): {fj['compliance']['faceplate_fea_mm_per_N'] * 1e6:.2f} μm/kN "
        "(バー穴固定、座ぐり底に軸力)。VDI の円錐モデルに含まれない曲げ分を被締結側コンプライアンスに加算。"
        if fj["compliance"].get("faceplate_fea_mm_per_N") else "フェイスプレートの曲げは考慮せず (円錐モデルのみ: 安全側)。"])
    bi = rep["bar_interface"]
    if bi:
        doc.h2(f"ハンドルバー挟み部 ({bi['bar']})")
        doc.para(f"4 本の最大締付力合計 {bi['clamp_force_max_N']:,.0f} N、平均面圧 {bi['pressure_avg_MPa']:g} MPa、"
                 f"ピーク面圧 (cos 分布) {bi['pressure_peak_MPa']:g} MPa、周方向圧縮応力 p·r/t = {bi['hoop_stress_MPa']:g} MPa。")
        doc.table(["判定項目", "計算値", "許容 / 必要", "判定"],
                  [[RULE_JP.get(x["check"], x["check"]), _f(x["value"], 1), _f(x["limit"], 1), OK if x["pass"] else NG]
                   for x in bi["rules"]], widths=[5, 2.5, 2.5, 1.5], highlight=_hl)


def main():
    os.makedirs(OUT, exist_ok=True)
    today = datetime.date.today().isoformat()
    doc = Report("締結計算書", "ステム・ハンドルバー・ブラケット・ハウジング", today, "AC-CALC-F01")
    reports, summary = {}, []
    for name in make_stem.CASES:
        print(f"[{name}]", file=sys.stderr)
        reports[name] = stem_case(name)
    reports["bracket_std"] = fc.bracket_joints(make_diecast.CASES["bracket_std"])
    reports["housing_std"] = fc.housing_screws(make_molding.CASES["housing_std"])
    for name, rep in reports.items():
        for k, j in rep.get("joints", {}).items():
            cap = j.get("capacity", {})
            sf = cap.get("slip_safety") or cap.get("slip_safety_test")
            summary.append([NUMBERS.get(name, name), j["joint"], f"{j['bolt']} × {j['n_bolts']}", _f(j["torque_Nm"], 1),
                            f"{j['torque_window_Nm']['min']:.1f} 〜 {j['torque_window_Nm']['max']:.1f}",
                            ", ".join(RULE_JP.get(x, x) for x in j["failed"]) or "-", OK if j["pass"] else NG])
        if "bar_interface" in rep and rep["bar_interface"]:
            bi = rep["bar_interface"]
            summary.append([NUMBERS.get(name, name), "ハンドルバー挟み部", bi["bar"], "-", "-",
                            ", ".join(RULE_JP.get(x["check"], x["check"]) for x in bi["rules"] if not x["pass"]) or "-",
                            OK if bi["pass"] else NG])
        if name == "housing_std":
            summary.append([NUMBERS[name], "カバー固定ねじ", rep["screw"] + " × 4", "試験で決定", "-",
                            ", ".join(RULE_JP.get(x, x) for x in rep["failed"]) or "-", OK if rep["pass"] else NG])
        json.dump(rep, open(os.path.join(OUT, f"{name}_fastener.json"), "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1, default=float)
    doc.h1("1. 概要")
    doc.para("各部品のボルト締結部について、VDI 2230 Part 1 (2015) の簡易計算法と JIS B 1083 のトルク係数式で、"
             "締付軸力のばらつき・なじみによる軸力低下・作用荷重を考慮した残留締付力、ボルト応力 (締付時・使用時)、"
             "疲労、座面圧、雌ねじのせん断を確認した。形状はすべて部品パラメータから読み取るため、寸法・材料を変えると再計算される。")
    doc.table(["対象", "継手", "ボルト", "締付トルク N·m", "成立範囲 N·m", "不足項目", "判定"], summary,
              widths=[3, 3.6, 3, 1.8, 2, 3, 1.3], highlight=_hl)
    doc.h1("2. 前提条件")
    doc.bullets(fc.assumptions())
    doc.pagebreak()
    for i, name in enumerate(make_stem.CASES):
        if i:
            doc.pagebreak()
        write_stem(doc, name, reports[name])
    doc.pagebreak()
    b = reports["bracket_std"]["joints"]["bracket"]
    doc.h1("ブラケット取付 (AC-1101)")
    doc.para(f"ADC12 ブラケットのボス (高さ {make_diecast.CASES['bracket_std'].boss_h:g} mm) を鋼の相手板 "
             f"(t{fc.BRACKET_MOUNT_T:g}, 仮定) に M8 通しボルト + ナット + 平座金で締結。ハブ荷重 {fc.BRACKET_HUB_LOAD_N:g} N "
             "(仮定、車両側の荷重が決まり次第差し替え) を横荷重として摩擦で受け、その曲げモーメントを片側 2 本の軸力で受ける。"
             f"締付トルク {_f(b['torque_Nm'], 0)} N·m は本計算からの推奨値 (図面未記載)。")
    joint_section(doc, "bracket", b, ["アルミダイカスト座面のクリープによる軸力低下を避けるため、平座金 (JIS B 1256, 外径 16 mm) を必ず使用する。"])
    h = reports["housing_std"]
    doc.h1("電装ハウジング カバー固定ねじ (AC-1201)")
    doc.para(f"{make_molding.CASES['housing_std'].material} のボスに {h['screw']} を直接ねじ込む。下穴・ボス・ねじ込み長を設計ガイドの目安と比較した。")
    doc.table(["判定項目", "計算値", "目安", "判定"],
              [[RULE_JP.get(x["check"], x["check"]), _f(x["value"], 2), _f(x["limit"], 2), OK if x["pass"] else NG] for x in h["rules"]],
              widths=[5, 2.5, 2.5, 1.5], highlight=_hl)
    doc.para(h["torque_note"])
    n = doc.save(os.path.join(OUT, "締結計算書.pdf"))
    bad = [r[0] + " " + r[1] for r in summary if r[-1] == NG]
    json.dump({"date": today, "summary": summary, "pass": not bad, "failed": bad},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{OUT}/締結計算書.pdf ({n} pages); " + ("all joints pass" if not bad else f"FAIL: {bad}"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
