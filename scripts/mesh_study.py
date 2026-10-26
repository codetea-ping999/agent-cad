"""Mesh convergence report of the solid FEA (メッシュ収束解析) -> out/mesh/.

    python -m scripts.mesh_study                 # stem and handlebar, ~30 min
    python -m scripts.mesh_study stem            # one part
    python -m scripts.mesh_study --report        # rebuild the PDF from the saved JSON

For the default stem (stem90_6061) and handlebar (drop420_7075): four meshes refined by
sqrt(2), every ISO 4210 style load case of the verification, mesh quality per level,
Richardson extrapolation and GCI (agent_cad/verify/mesh_study.py); JSON, convergence
plots and a PDF report. Exit 1 when the production mesh is not within the accepted band.
"""
import sys, os, json, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agent_cad.verify import mesh_study as ms
from agent_cad.report import Report
from agent_cad.drawing.render import _font
from scripts import make_stem, make_handlebar

OUT = os.path.join("out", "mesh")
CASE = {"stem": "stem90_6061", "handlebar": "drop420_7075"}
TITLE = {"stem": "ステム本体 (AC-1001, stem90_6061)", "handlebar": "ハンドルバー 右半分 (AC-1301, drop420_7075)"}
KEY_JP = {"max_disp_mm": "最大変位 mm", "vm_p999_MPa": "von Mises 99.9%値 MPa", "vm_p999_away_MPa": "同 クランプ端除外 MPa",
          "vm_peak_away_MPa": "支持部以外の最大 MPa", "vm_max_MPa": "全体最大 (特異点含む) MPa"}
STATUS_JP = {"converging": "収束", "converged": "収束 (変化なし)", "oscillating": "振動", "diverging": "発散 (特異)"}


def run(part):
    log = lambda s: print(s, file=sys.stderr, flush=True)
    if part == "stem":
        r = ms.stem_study(make_stem.CASES[CASE[part]], log=log)
    else:
        r = ms.handlebar_study(make_handlebar.CASES[CASE[part]], log=log)
    json.dump(r, open(os.path.join(OUT, f"{part}_study.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    return r


def plot(part, r):
    font = _font()
    cases = list(r["summary"])
    keys = list(r["summary"][cases[0]])
    h = [l["quality"]["h_eff_mm"] for l in r["levels"]]
    prod = [l["h"] for l in r["levels"]].index(r["production_mesh"]["h"])
    nr = (len(keys) + 2) // 3
    fig, axs = plt.subplots(nr, 3, figsize=(9.0, 2.9 * nr), squeeze=False)
    axs = axs.ravel()
    for ax in axs[len(keys):]:
        ax.axis("off")
    for ax, k in zip(axs, keys):
        for c in cases:
            e = r["summary"][c][k]
            v = e["values"]
            ref = e["f_extrap"] or v[-1]
            ax.plot(h, [x / ref for x in v], "o-", ms=3, lw=1, label=c)
        ax.axvline(h[prod], color="0.6", lw=0.8, ls="--")          # production mesh
        ax.invert_xaxis(); ax.set_xscale("log")
        ax.set_title(KEY_JP.get(k, k), fontsize=7.5, family=font)
        ax.set_xlabel("代表要素寸法 h_eff [mm]", fontsize=6.5, family=font)
        ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
    for i in range(0, len(keys), 3):
        axs[i].set_ylabel("値 / 外挿値", fontsize=6.5, family=font)
    axs[len(keys) - 1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{part}_convergence.png"), dpi=150)
    return fig


def section(doc, part, r):
    doc.h1(f"{TITLE[part]}")
    pm = r["production_mesh"]
    doc.para(f"検証ループで使う本番メッシュ: h = {pm['h']:g} mm, 曲率分割 {pm['curvature']:g} 点/周 (P2 四面体)。"
             "全体寸法と曲率分割を √2 ずつ同時に細かくした 4 段階 (本番の 2 段粗い〜1 段細かい) で同じ荷重ケースを解いた。")
    doc.h2("メッシュ品質")
    rows = [[f"{l['h']:g} / {l['curvature']:g}", f"{l['quality']['tets']:,}", f"{l['dofs']:,}", f"{l['quality']['h_eff_mm']:.2f}",
             f"{l['quality']['quality_min']:.3f}", f"{l['quality']['quality_p01']:.3f}", f"{l['quality']['quality_median']:.3f}",
             f"{l['quality']['share_q_below_0.2_pct']:.3f}", f"{l['quality']['dihedral_min_deg']:.1f}",
             l["solver"], f"{l['seconds']:.0f}"] for l in r["levels"]]
    doc.table(["h / 曲率", "要素数", "自由度", "h_eff", "品質 最小", "品質 1%", "品質 中央", "q<0.2 %", "最小二面角°", "解法",
               "時間 s"], rows, widths=[1.6, 1.3, 1.5, 1, 1.1, 1.1, 1.1, 1.1, 1.3, 1, 1], size=6.6)
    doc.para("品質 q = 12(3V)^(2/3)/Σl² (正四面体で 1)。q < 0.2 の要素が 1 % 未満、最小二面角 5° 以上なら P2 要素の応力評価に支障なし。", 7.5)
    doc.h2("収束 (Richardson 外挿・GCI、細かい 3 段)")
    for c, res in r["summary"].items():
        rows = []
        for k, e in res.items():
            rows.append([KEY_JP.get(k, k), " / ".join(f"{v:.4g}" for v in e["values"]), STATUS_JP.get(e["status"], e["status"]),
                         f"{e['p']:.2f}" if e["p"] is not None else "-",
                         f"{e['f_extrap']:.4g}" if e["f_extrap"] is not None else "-",
                         f"{e['gci_fine_pct']:.2f}" if e["gci_fine_pct"] is not None else "-",
                         f"{e['production_error_pct']:.2f}" if e["production_error_pct"] is not None else
                         (f"(変化 {e['change_fine_pct']:.1f})" if e["change_fine_pct"] is not None else "-")])
        fin = r["levels"][-1]["results"][c]
        doc.para(f"荷重ケース: {c}  (最も細かいメッシュでの支持部以外の最大点 {fin['at_away_mm']} mm)", 8)
        doc.table(["結果", "粗 → 細 の値", "傾向", "次数 p", "外挿値", "GCI %", "本番誤差 %"], rows,
                  widths=[3.2, 4.4, 1.6, 1, 1.4, 1, 1.4], size=6.6)
    v = r["verdict"]
    doc.para(("判定: 合格 — " if v["pass"] else "判定: 不合格 — ") +
             (f"判定に使う値はすべて本番メッシュで外挿値の {v['accept_pct']:g} % 以内。" if v["pass"]
              else "外挿値から離れている項目: " + ", ".join(v["failed"])))
    doc.bullets(r["notes"])


def report(parts):
    today = datetime.date.today().isoformat()
    doc = Report("メッシュ収束解析報告書", "ステム本体・ハンドルバー (ISO 4210-5 相当荷重, 暫定)", today, "AC-CALC-M01")
    doc.h1("1. 目的と方法")
    doc.para("強度判定に使う固体 FEA (gmsh P2 四面体 + scikit-fem) が要素寸法に依存しないことを確認する。"
             "同じ形状・荷重を 4 段階のメッシュで解き、最も細かい 3 段から Richardson 外挿で要素寸法 0 の値を推定し、"
             "Roache の GCI (安全係数 1.25, Celik ほか 2008 の手順) で誤差帯を求めた。本番メッシュの値が外挿値の "
             f"{ms.ACCEPT_GCI_PCT:g} % 以内なら合格とする。剛体支持の角部は理論上応力が無限大になる特異点で、"
             "そこでの最大値は細分化するほど増え続けるため判定には使わない (99.9% 値・支持部除外の値で判定)。")
    ok = True
    for part in parts:
        r = ms.summarise(json.load(open(os.path.join(OUT, f"{part}_study.json"), encoding="utf-8")))
        doc.pagebreak()
        section(doc, part, r)
        doc.figure(plot(part, r), 120)
        ok &= r["verdict"]["pass"]
    n = doc.save(os.path.join(OUT, "メッシュ収束解析報告書.pdf"))
    print(f"{OUT}/メッシュ収束解析報告書.pdf ({n} pages): {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    parts = [a for a in argv if not a.startswith("-")] or ["stem", "handlebar"]
    if "--report" not in argv:
        for part in parts:
            run(part)
    done = [p for p in ("stem", "handlebar") if os.path.exists(os.path.join(OUT, f"{p}_study.json"))]
    return 0 if report(done) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
