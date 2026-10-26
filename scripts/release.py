"""Drawing release (提出図): build every drawing in the register, check it, and write the
submission package to out/release/.

    python -m scripts.release                      # build + check everything (exit 1 if any gate fails)
    python -m scripts.release --only AC-1101       # some numbers only
    python -m scripts.release --write-history      # also record new first issues / refreshed drafts
    python -m scripts.release revise AC-1101 "ハブ穴径変更"   # start the next revision of an approved drawing
    python -m scripts.release approve AC-1101 --by <name>    # record a person's approval (all: --all)

For each number: the JIS drawing (PDF, DXF) stamped with revision, status and sign-off,
the STEP AP242 (3D drawing: shape plus semantic PMI), and the checks the release gate
runs (drawing check, STEP exchange check, PMI read-back, the part's or tool's design /
process checks, revision state). Plus the drawing list (PDF, CSV), the change history and
a manifest with SHA-256 of every file. `approve` and `revise` read the manifest of the
last build, so they record the drawing that was actually checked.
"""
import sys, os, json, csv, argparse
from functools import lru_cache
from build123d import Compound, scale
from agent_cad import release as rl, pmi
from agent_cad.step_io import write_step, PartSpec
from agent_cad.verify.step_check import check_step, summary
from agent_cad.drawing.check import check_sheet
from agent_cad.drawing.render import to_pdf, to_dxf
from agent_cad.drawing.sheet import Sheet, A3
from agent_cad.drawing import tool
from agent_cad.parts import stem, stem_drawing, bracket, bracket_drawing, housing, housing_drawing, handlebar as hb, handlebar_drawing
from agent_cad.process import forging as fg, diecast as dc, molding as mo, tube_bending as tb
from agent_cad.verify.forging_check import verify_forging
from agent_cad.verify.diecast_check import verify_diecast
from agent_cad.verify.molding_check import verify_molding
from scripts import make_stem, make_diecast, make_molding, make_handlebar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "release", "registry.json")
HISTORY = os.path.join(ROOT, "release", "history.json")
OUT = os.path.join("out", "release")
STEEL_RGB, PIN_RGB, SLIDE_RGB = (0.42, 0.47, 0.52), (0.85, 0.65, 0.2), (0.55, 0.5, 0.42)
SKD61 = ("SKD61 (JIS G 4404)", "焼入れ焼戻し 44〜48 HRC")
NAK80 = ("NAK80 相当 プリハードン鋼", "プリハードン 37〜41 HRC (熱処理なし)")


# ------------------------------------------------------------------ model families (built once)
@lru_cache(None)
def stem_case(case):
    p = make_stem.CASES[case]
    b, f = stem.build(p)
    return p, b, f, stem.checks(p, b, f)


@lru_cache(None)
def forging_case(case):
    p = make_stem.CASES[case]
    fp = fg.ForgingParams()
    ds = fg.die_set(p, fp)
    return p, fp, ds, verify_forging(p, fp, ds=ds)


@lru_cache(None)
def diecast_case(case):
    p, dp = make_diecast.CASES[case], dc.DiecastParams()
    ds = dc.die_set(p, dp)
    return p, dp, ds, verify_diecast(p, dp, ds=ds)


@lru_cache(None)
def mould_case(case):
    p, mp = make_molding.CASES[case], mo.MouldParams()
    ms = mo.mould_set(p, mp)
    return p, mp, ms, verify_molding(p, mp, ms=ms)


@lru_cache(None)
def handlebar_case(case):
    p, tp, bar, rep, tools, prep = make_handlebar.case(case)
    return p, tp, bar, rep, tools, prep, tb.tool_parts(p, tp, tools, make_handlebar.SETUP)


@lru_cache(None)
def fastener_case(case):
    """Fastener calculation of a stem case with the registered handlebar in its bar clamp
    (scripts/fastener_report.py writes the full calculation sheet). Recorded with the
    stem body drawing, whose note states the torques it checks."""
    from agent_cad.verify import fastener_check as fc
    r = fc.stem_joints(make_stem.CASES[case], make_handlebar.CASES["drop420_7075"])
    return dict(pass_=bool(r["pass"]), failed=r["failed"], torque_note=fc.torque_note(),
                torque_window_Nm={k: j["torque_window_Nm"] for k, j in r["joints"].items()})


def _design(rep):
    return dict(pass_=bool(rep["pass"]), failed=list(rep.get("failed", [])))


# ------------------------------------------------------------------ generators
# each returns dict(sheet, parts=[PartSpec], assembly=name|None, shape, design, kind)

def g_stem_body(it, ctx):
    p, b, f, chk = stem_case(it["case"])
    sh = stem_drawing.body_sheet(p, b, dwg_no=it["number"])
    ps = PartSpec(f"{it['number']}_STEM_BODY", b, make_stem.BODY_RGB, p.material, stem.density(p), pmi=pmi.from_sheet(sh))
    return dict(sheet=sh, parts=[ps], assembly=None, shape=b, design=dict(pass_=chk["pass"], failed=[]), kind="製品",
                fastener=fastener_case(it["case"]))


def g_stem_faceplate(it, ctx):
    p, b, f, chk = stem_case(it["case"])
    sh = stem_drawing.faceplate_sheet(p, f, dwg_no=it["number"])
    ps = PartSpec(f"{it['number']}_STEM_FACEPLATE", f, make_stem.FP_RGB, p.material, stem.density(p), pmi=pmi.from_sheet(sh))
    return dict(sheet=sh, parts=[ps], assembly=None, shape=f, design=dict(pass_=chk["pass"], failed=[]), kind="製品")


def g_bracket(it, ctx):
    p, dp, ds, rep = diecast_case(it["case"])
    sh = bracket_drawing.part_sheet(p, ds["finished"], it["number"])
    ps = PartSpec(f"{it['number']}_BRACKET", ds["finished"], make_diecast.CAST_RGB, p.material, bracket.density(p),
                  pmi=pmi.from_sheet(sh))
    return dict(sheet=sh, parts=[ps], assembly=None, shape=ds["finished"], design=_design(rep), kind="製品")


def g_housing(it, ctx):
    p, mp, ms, rep = mould_case(it["case"])
    sh = housing_drawing.part_sheet(p, ms["part"], dwg_no=it["number"])
    ps = PartSpec(f"{it['number']}_HOUSING", ms["part"], make_molding.PART_RGB, p.material, housing.density(p),
                  pmi=pmi.from_sheet(sh))
    return dict(sheet=sh, parts=[ps], assembly=None, shape=ms["part"], design=_design(rep), kind="製品")


def _up(shape):
    bb = shape.bounding_box()
    return -1 if bb.min.Y >= -1e-6 else 1


def _product_note(it, ctx, what):
    no = it["product"]
    return f"製品図 {no} 改訂 {ctx['revs'][no]} に対応。{what}"


def _component(it, ctx, shape, mat, block, holes, notes, en):
    info = dict(number=it["number"], title=it["title"], material=mat[0], finish=mat[1], up=_up(shape), block=block,
                holes=holes, notes=notes, density_g_cm3=7.8)
    sh = tool.component_sheet(shape, info)
    ps = PartSpec(f"{it['number']}_{en}", shape, STEEL_RGB, mat[0].split(" (")[0], 7.8, pmi=pmi.from_sheet(sh))
    return sh, ps


def _forging_block(p, fp, ds):
    """Die block size as fg.dies() makes it."""
    k = fg.shrink_factor(p, fp)
    cavity = scale(ds["blank"], k)
    outline, _, t, b, tg = fg.flash_dims(cavity, fp)
    x0, z0, x1, z1 = fg._clean(outline.buffer(b + fp.gutter_w_mm, quad_segs=8)).bounds
    w = fp.die_wall_mm
    return (x1 - x0 + 2 * w, cavity.bounding_box().max.Y + fp.die_base_mm, z1 - z0 + 2 * w), k, t


def g_forging_die(which):
    def g(it, ctx):
        p, fp, ds, rep = forging_case(it["case"])
        block, k, t = _forging_block(p, fp, ds)
        sh, ps = _component(it, ctx, ds[which], SKD61, block, [],
                            [_product_note(it, ctx, f"キャビティは鍛造品を熱収縮分 {(k - 1) * 100:.2f} % 拡大。"),
                             f"見切り面は型合わせ面。バリ厚さ {t:.2f} (上下計)、バリ溜まりは外周。型温 {fp.die_temp_C:g} °C で使用。",
                             "型彫り面は窒化処理。"], f"FORGING_DIE_{which.upper()}")
        return dict(sheet=sh, parts=[ps], assembly=None, shape=ds[which], design=_design(rep), kind="型部品")
    return g


def g_forging_assembly(it, ctx):
    p, fp, ds, rep = forging_case(it["case"])
    block, k, t = _forging_block(p, fp, ds)
    up, lo, fo = ds["upper"], ds["lower"], ds["forging"]
    shape = Compound([up, lo, fo])
    n = ctx["numbers"]
    bom = [("上型", n["forging_upper"], "SKD61", 1, ""), ("下型", n["forging_lower"], "SKD61", 1, ""),
           ("鍛造品 (バリ付き)", it["product"], p.material, 1, "参考")]
    mass = (up.volume + lo.volume) / 1000 * 7.8
    info = dict(number=it["number"], title=it["title"], up=1, block=(block[0], 2 * block[1], block[2]),
                dim_shape=Compound([up, lo]), bom=bom, parts_count=3, mass_g=mass,
                notes=[_product_note(it, ctx, "型の開閉方向 Y、見切り面 Y = 0。上下型は位置決めピン (図示せず) で合わせる。")])
    sh = tool.assembly_sheet(shape, info)
    steel = fg.DIE_STEEL
    parts = [PartSpec(f"{n['forging_upper']}_FORGING_DIE_UPPER", up, STEEL_RGB, *steel),
             PartSpec(f"{n['forging_lower']}_FORGING_DIE_LOWER", lo, STEEL_RGB, *steel),
             PartSpec(f"{it['product']}_FORGING_WITH_FLASH", fo, make_stem.FP_RGB, p.material, stem.density(p))]
    return dict(sheet=sh, parts=parts, assembly=f"{it['number']}_FORGING_DIE_SET", shape=shape, design=_design(rep), kind="型組立")


def g_diecast_die(which):
    def g(it, ctx):
        p, dp, ds, rep = diecast_case(it["case"])
        lx, hb, lz = ds["block_mm"]
        k = dc.shrink_factor(dp)
        if which == "moving":
            block = (lx, hb, lz)
            holes = [("冷却穴", dp.cool_d_mm, "X", len(ds["channels_moving"])), ("突出しピン穴", dp.pin_d_mm, "Y", len(ds["pins"]))]
            note = f"キャビティ・方案は方案付き鋳造品を {dp.shrink_pct:g} % 拡大。製品は鋳抜きピンとともに可動型に残る。"
        else:
            block = (lx, dp.cover_t_mm, lz)
            holes = [("冷却穴", dp.cool_d_mm, "X", len(ds["channels_cover"])), ("射出スリーブ穴", dp.biscuit_d_mm * k, "Y", 1)]
            note = "射出スリーブ穴はスリーブとはめあい (H7/g6 相当、協議)。"
        sh, ps = _component(it, ctx, ds[which], SKD61, block, holes,
                            [_product_note(it, ctx, note), f"冷却穴 φ{dp.cool_d_mm:g} は貫通、両端に管用テーパねじ (協議)。"],
                            f"DIECAST_DIE_{which.upper()}")
        return dict(sheet=sh, parts=[ps], assembly=None, shape=ds[which], design=_design(rep), kind="型部品")
    return g


def g_diecast_assembly(it, ctx):
    p, dp, ds, rep = diecast_case(it["case"])
    lx, hb, lz = ds["block_mm"]
    mv, cv, pins, shot = ds["moving"], ds["cover"], ds["pins"], ds["shot"]
    shape = Compound([mv, cv, *pins, shot])
    n = ctx["numbers"]
    bom = [("可動型", n["diecast_moving"], "SKD61", 1, ""), ("固定型", n["diecast_cover"], "SKD61", 1, ""),
           (f"突出しピン φ{dp.pin_d_mm:g}", "JIS B 5103 相当", "SKD61", len(pins), "標準部品"),
           ("方案付き鋳造品", it["product"], p.material, 1, "参考")]
    mass = (mv.volume + cv.volume + sum(x.volume for x in pins)) / 1000 * 7.8
    info = dict(number=it["number"], title=it["title"], up=1, block=(lx, hb + dp.cover_t_mm, lz), dim_shape=Compound([mv, cv]),
                bom=bom, parts_count=3 + len(pins), mass_g=mass,
                notes=[_product_note(it, ctx, "コールドチャンバー式、見切り面 Y = 0。"),
                       f"突出しピンの後端は可動型背面から {dp.pin_tail_mm:g} (突出し板へ)。"])
    sh = tool.assembly_sheet(shape, info)
    st = dc.DIE_STEEL
    parts = [PartSpec(f"{n['diecast_moving']}_DIE_MOVING", mv, STEEL_RGB, *st),
             PartSpec(f"{n['diecast_cover']}_DIE_COVER", cv, STEEL_RGB, *st)] + \
            [PartSpec(f"{it['number']}-P_EJECTOR_PIN_{i + 1}", x, PIN_RGB, *st) for i, x in enumerate(pins)] + \
            [PartSpec(f"{it['product']}_SHOT_WITH_GATING", shot, make_diecast.CAST_RGB, p.material, bracket.density(p))]
    return dict(sheet=sh, parts=parts, assembly=f"{it['number']}_DIECAST_DIE_SET", shape=shape, design=_design(rep), kind="型組立")


def g_mould_part(which):
    def g(it, ctx):
        p, mp, ms, rep = mould_case(it["case"])
        lx, h, lz = ms["block_mm"]
        hm, hf = -ms["plate_y"][0], ms["plate_y"][1]
        k = mo.shrink_factor(p)
        if which == "fixed":
            block, holes = (lx, hf, lz), [("冷却穴", mp.cool_d_mm, "X", len(ms["channels_fixed"]))]
            note = f"キャビティ・スプルー・ランナー・ゲートはショットを成形収縮分 {(k - 1) * 100:.2f} % 拡大。"
        elif which == "moving":
            block = (lx, hm + housing.floor_y(p) * k, lz)       # the core rises to the part's floor underside
            holes = [("冷却穴", mp.cool_d_mm, "Z", len(ms["channels_moving"])), ("バブラー穴", mp.bubbler_d_mm, "Y", len(ms["bubblers"])),
                     ("突出しピン穴", mp.pin_d_mm, "Y", len(ms["pins"]))]
            note = f"コアは製品内面を成形収縮分 {(k - 1) * 100:.2f} % 拡大。スライド溝・コアピン・スリーブ穴は3Dデータによる。"
        else:
            bb = Compound([ms["slide_block"], ms["window_insert"]]).bounding_box()   # body + the window core
            block, holes = (bb.size.X, bb.size.Y, bb.size.Z), []
            note = f"アンギュラピン {mp.angle_pin_deg:g}°、ロッキング面 {mp.lock_deg:g}°。窓成形面は成形収縮分を拡大。"
        mat = NAK80 if which != "slide" else SKD61
        sh, ps = _component(it, ctx, ms[which], mat, block, holes, [_product_note(it, ctx, note)], f"MOULD_{which.upper()}")
        return dict(sheet=sh, parts=[ps], assembly=None, shape=ms[which], design=_design(rep), kind="型部品")
    return g


def g_mould_assembly(it, ctx):
    p, mp, ms, rep = mould_case(it["case"])
    lx, h, lz = ms["block_mm"]
    k = mo.shrink_factor(p)
    shot_k = scale(ms["shot"], k)
    others = [ms["angle_pin"], *ms["core_pins"], *ms["sleeves"], *ms["pins"]]
    shape = Compound([ms["fixed"], ms["moving"], ms["slide"], *others, shot_k])
    n = ctx["numbers"]
    bom = [("固定側", n["mould_fixed"], "NAK80 相当", 1, ""), ("可動側", n["mould_moving"], "NAK80 相当", 1, ""),
           ("スライド", n["mould_slide"], "SKD61", 1, ""),
           (f"アンギュラピン φ{mp.angle_pin_d_mm:g}", "標準部品", "SUJ2 相当", 1, "標準部品"),
           ("コアピン", "標準部品", "SKD61 相当", len(ms["core_pins"]), "標準部品"),
           ("突出しスリーブ", "標準部品", "SKD61 相当", len(ms["sleeves"]), "標準部品"),
           (f"突出しピン φ{mp.pin_d_mm:g}", "JIS B 5103 相当", "SKD61 相当", len(ms["pins"]), "標準部品"),
           ("ショット (成形品)", it["product"], p.material, 1, "参考")]
    mass = sum(x.volume for x in [ms["fixed"], ms["moving"], ms["slide"], *others]) / 1000 * 7.8
    info = dict(number=it["number"], title=it["title"], up=1, block=(lx, h, lz), dim_shape=Compound([ms["fixed"], ms["moving"]]),
                bom=bom, parts_count=4 + len(others), mass_g=mass,
                notes=[_product_note(it, ctx, "1個取り、コールドランナー、見切り面 Y = 0。"),
                       "窓のアンダーカットはスライドで抜く (固定側のアンギュラピンで駆動)。"])
    sh = tool.assembly_sheet(shape, info)
    st = mo.MOULD_STEEL
    s = make_molding.specs(p, ms)
    parts = [PartSpec(f"{n['mould_fixed']}_MOULD_FIXED", ms["fixed"], STEEL_RGB, *st),
             PartSpec(f"{n['mould_moving']}_MOULD_MOVING", ms["moving"], STEEL_RGB, *st),
             PartSpec(f"{n['mould_slide']}_MOULD_SLIDE", ms["slide"], SLIDE_RGB, *st)] + \
            [PartSpec(x.name.replace(make_molding.doc_no(p), it["number"]), x.shape, x.color, x.material, x.density_g_cm3)
             for x in [s["angle_pin"], *s["core_pins"], *s["sleeves"], *s["pins"]]] + \
            [PartSpec(f"{it['product']}_SHOT", shot_k, make_molding.PART_RGB, p.material, housing.density(p))]
    return dict(sheet=sh, parts=parts, assembly=f"{it['number']}_MOULD", shape=shape, design=_design(rep), kind="型組立")


def g_handlebar(it, ctx):
    p, tp, bar, rep, tools, prep, parts = handlebar_case(it["case"])
    sh = handlebar_drawing.part_sheet(p, bar, it["number"])
    ps = PartSpec(f"{it['number']}_HANDLEBAR", bar, make_handlebar.BAR_RGB, p.material, hb.density(p), pmi=pmi.from_sheet(sh))
    return dict(sheet=sh, parts=[ps], assembly=None, shape=bar, design=_design(rep), kind="製品")


# bending / swaging tool parts: (key in tb.tool_parts, STEP name, note)
BEND_TOOLS = {
    "bend_die": ("BEND_DIE", "溝は管外径 + 片側 {c:g} の円弧溝、曲げ半径 R{R:g} (溝中心)。クランプインサートのポケット付き。"),
    "clamp_straight": ("CLAMP_DIE_STRAIGHT", "B1・B3 用 (直管部をつかむ)。溝面はつかみ用ローレット (協議)。"),
    "insert_straight": ("CLAMP_INSERT_STRAIGHT", "B1・B3 用。曲げ型のポケットにボルト締結 (協議)。"),
    "clamp": ("CLAMP_DIE_B2", "B2 用。溝は B1 の曲げ部に沿う (3Dデータによる)。"),
    "insert": ("CLAMP_INSERT_B2", "B2 用。溝は B1 の曲げ部に沿う (3Dデータによる)。"),
    "pressure_die": ("PRESSURE_DIE", "全曲げ共用。曲げ型とのすきま {g:g}。"),
    "wiper": ("WIPER_DIE", "全曲げ共用。先端は曲げ型の溝に合わせて仕上げる (フェザーエッジ)。"),
    "mandrel": ("BALL_MANDREL", "ボール {n} 個、管内径とのすきま片側 {m:g}。リンクはメーカーと協議。"),
    "swage_die": ("SWAGE_DIE_HALF", "ロータリースエージング型 (2 つ割の片側、同形 2 個で 1 組)。"),
    "swage_mandrel": ("SWAGE_MANDREL", "段付きマンドレル。肉厚 (バテッド) を決める。"),
}


def g_bend_tool(which):
    def g(it, ctx):
        p, tp, bar, rep, tools, prep, parts = handlebar_case(it["case"])
        t = parts[which]
        en, note = BEND_TOOLS[which]
        mr = tools["plan"][make_handlebar.SETUP]["mandrel"]
        note = note.format(c=tp.groove_clr, R=tools["arc"].R, g=tp.pd_gap, n=mr["balls"], m=tp.mandrel_clr)
        surface = "溝面" if not which.startswith("mandrel") and which != "swage_mandrel" else "外周面"
        info = dict(number=it["number"], title=it["title"], material=t["material"][0], finish=t["material"][1], up=1,
                    block=t["block"], holes=t["holes"], density_g_cm3=t["density"], surface=surface, cavity_ra=0.2,
                    shape_note="溝・成形面の形状は3Dデータ (STEP AP242、同図番) による。",
                    edge_note="指示なきかどは C0.5、溝の縁は R0.5。",
                    notes=[_product_note(it, ctx, note)])
        sh = tool.component_sheet(t["shape"], info)
        rgb = make_handlebar.BRONZE_RGB if which == "wiper" else STEEL_RGB
        ps = PartSpec(f"{it['number']}_{en}", t["shape"], rgb, t["material"][0].split(" (")[0], t["density"], pmi=pmi.from_sheet(sh))
        return dict(sheet=sh, parts=[ps], assembly=None, shape=t["shape"], design=_design(prep), kind="型部品")
    return g


def g_bend_assembly(it, ctx):
    p, tp, bar, rep, tools, prep, parts = handlebar_case(it["case"])
    n = ctx["numbers"]
    keys = ["bend_die", "clamp", "insert", "pressure_die", "wiper"]
    mand = Compound([tools["mandrel_shank"], *tools["mandrel_balls"]])
    shape = Compound([tools[k] for k in keys] + [mand, tools["tube"]])
    dim = Compound([tools[k] for k in keys])
    bb = dim.bounding_box()
    st = tb.TOOL_STEEL[0].split(" (")[0]
    bom = [("曲げ型", n["bend_die"], st, 1, ""), ("クランプ型 (B2)", n["bend_clamp_b2"], st, 1, ""),
           ("クランプインサート (B2)", n["bend_insert_b2"], st, 1, ""), ("プレッシャー型", n["bend_pressure_die"], st, 1, ""),
           ("ワイパー型", n["bend_wiper"], "CAC703", 1, ""), ("ボールマンドレル", n["bend_mandrel"], st, 1, ""),
           ("曲げ途中の管 (B1 曲げ後)", it["product"], tb.BENDING_TEMPER.get(p.material, p.material), 1, "参考")]
    mass = sum(tools[k].volume for k in keys) / 1000 * 7.8 + mand.volume / 1000 * 7.8
    b = tools["plan"][make_handlebar.SETUP]
    info = dict(number=it["number"], title=it["title"], up=1, block=(bb.size.X, bb.size.Y, bb.size.Z), dim_shape=dim,
                bom=bom, parts_count=7, mass_g=mass,
                notes=[_product_note(it, ctx, "回転引き曲げ、右側 B2 (フード部) の段取り。曲げ型軸 Z、接点 (0, -R, 0)。"),
                       f"曲げ角 {b['angle']:g}°、オーバーベンド {b['springback']['overbend_deg']:g}° (O 材、試し曲げで補正)。",
                       f"B1・B3 は直線クランプ型・インサート ({n['bend_clamp']}・{n['bend_insert']}) に替える。",
                       "曲げ型・プレッシャー型・ワイパー型・マンドレルは全曲げ共用。",
                       "左側は逆勝手 (両方向曲げ機) で同じ型を使う。"])
    sh = tool.assembly_sheet(shape, info)
    specs = make_handlebar.setup_specs(p, tools, it["number"])
    return dict(sheet=sh, parts=specs, assembly=f"{it['number']}_BENDING_SETUP_B2", shape=shape, design=_design(prep),
                kind="型組立")


SOURCES = {
    "stem_body": g_stem_body, "stem_faceplate": g_stem_faceplate, "bracket": g_bracket, "housing": g_housing,
    "forging_upper": g_forging_die("upper"), "forging_lower": g_forging_die("lower"), "forging_assembly": g_forging_assembly,
    "diecast_moving": g_diecast_die("moving"), "diecast_cover": g_diecast_die("cover"), "diecast_assembly": g_diecast_assembly,
    "mould_fixed": g_mould_part("fixed"), "mould_moving": g_mould_part("moving"), "mould_slide": g_mould_part("slide"),
    "mould_assembly": g_mould_assembly,
    "handlebar": g_handlebar, "bend_assembly": g_bend_assembly, "bend_die": g_bend_tool("bend_die"),
    "bend_clamp": g_bend_tool("clamp_straight"), "bend_insert": g_bend_tool("insert_straight"),
    "bend_clamp_b2": g_bend_tool("clamp"), "bend_insert_b2": g_bend_tool("insert"),
    "bend_pressure_die": g_bend_tool("pressure_die"), "bend_wiper": g_bend_tool("wiper"),
    "bend_mandrel": g_bend_tool("mandrel"), "swage_die": g_bend_tool("swage_die"),
    "swage_mandrel": g_bend_tool("swage_mandrel"),
}


# ------------------------------------------------------------------ build
def file_stem(number, rev):
    return number if rev == rl.FIRST_REV else f"{number}_{rev}"


def build(only=None, write_history=False, out=OUT):
    os.makedirs(out, exist_ok=True)
    reg = json.load(open(REGISTRY, encoding="utf-8"))["items"]
    hist = rl.History(HISTORY)
    today = rl.today()
    ctx = dict(revs={}, numbers={it["source"]: it["number"] for it in reg})
    # products first: tool drawings cite the product's current revision
    reg = sorted(reg, key=lambda it: ("product" in it, it["number"]))
    results = []
    for it in reg:
        if only and it["number"] not in only and not any(it["number"] == x.get("product") for x in reg if x["number"] in only):
            continue
        print(f"[{it['number']}] {it['title']}", file=sys.stderr)
        g = SOURCES[it["source"]](it, ctx)
        sh = g["sheet"]
        dchk = check_sheet(sh)
        fp = rl.fingerprint(sh, g["shape"])
        st = hist.state(it["number"], fp, today)
        ctx["revs"][it["number"]] = st["rev"]
        if only and it["number"] not in only:
            continue                       # built only for its revision (cited by a tool drawing)
        stem_ = file_stem(it["number"], st["rev"])
        step_path = os.path.join(out, stem_ + ".step")
        write_step(g["parts"], step_path, g["assembly"])
        xchk = check_step(step_path, g["parts"], g["assembly"])
        print(" ", summary(xchk), file=sys.stderr)
        pmichk = None
        main = g["parts"][0]
        if main.pmi:
            pmichk = pmi.check(step_path, main.pmi, main.pmi_skipped or [])
        gates = {"drawing": dchk["pass"], "step_exchange": xchk["pass"], "pmi": pmichk is None or pmichk["pass"],
                 "design": g["design"]["pass_"], "revision": st["error"] is None}
        ok = all(gates.values())
        status = rl.stamp(sh, it["number"], st, hist.entries(it["number"]), ok, today)
        # stamping must not disturb the layout
        post = check_sheet(sh)
        gates["drawing"] = post["pass"]
        ok = all(gates.values())
        to_pdf(sh, os.path.join(out, stem_ + ".pdf"))
        to_dxf(sh, os.path.join(out, stem_ + ".dxf"))
        if write_history and st["pending"] is not None:
            hist.apply(it["number"], st["pending"])
        rep = dict(number=it["number"], title=it["title"], kind=g["kind"], rev=st["rev"], status=status, fingerprint=fp,
                   product=it.get("product"), product_rev=ctx["revs"].get(it.get("product")), sheet=f"{'A3' if sh.size == A3 else 'A2'}",
                   scale=sh.meta["title_block"]["尺度"], gates=gates, pass_=ok, revision_error=st["error"],
                   design_failed=g["design"]["failed"], drawing_failed=[c["check"] for c in post["checks"] if not c["pass"]],
                   pmi=pmichk, step_exchange=xchk["pass"], fastener=g.get("fastener"),
                   files=[stem_ + e for e in (".pdf", ".dxf", ".step")])
        json.dump(dict(rep, drawing=post, step=xchk), open(os.path.join(out, stem_ + "_report.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1, default=str)
        rep["files"].append(stem_ + "_report.json")
        print(f"  rev {st['rev']}  {status}  {'PASS' if ok else 'FAIL ' + str([k for k, v in gates.items() if not v])}",
              file=sys.stderr)
        if rep["fastener"] and not rep["fastener"]["pass_"]:
            print(f"  fastener calculation (informative, not a gate yet): FAIL {rep['fastener']['failed']}", file=sys.stderr)
        results.append(rep)
    if write_history:
        hist.save()
    _package(results, hist, out, today)
    return results


def _package(results, hist, out, today):
    results = sorted(results, key=lambda r: r["number"])
    with open(os.path.join(out, "drawing_list.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["図番", "改訂", "図名", "区分", "用紙", "尺度", "状態", "対応製品図", "形状指紋", "検図"])
        for r in results:
            w.writerow([r["number"], r["rev"], r["title"], r["kind"], r["sheet"], r["scale"], r["status"],
                        f"{r['product']} 改訂 {r['product_rev']}" if r["product"] else "", r["fingerprint"],
                        "合格" if r["pass_"] else "不合格"])
    _list_sheet(results, os.path.join(out, "図面一覧表.pdf"), today)
    lines = ["# 変更履歴", "", f"出力日 {today}。改訂記号: 初版「-」、以降 A, B, …", ""]
    for r in results:
        lines += [f"## {r['number']} {r['title']}", "", "| 改訂 | 日付 | 改訂内容 | 承認 | 形状指紋 |", "|---|---|---|---|---|"]
        es = hist.entries(r["number"]) or []
        if not es or es[-1]["fingerprint"] != r["fingerprint"]:
            es = es + [dict(rev=r["rev"], date=today, change="(未記録の変更)" if es else "初版", fingerprint=r["fingerprint"])]
        for e in reversed(es):
            ap = f"{e['approved_by']} {e['approved_on']}" if e.get("approved_by") else "未承認"
            lines.append(f"| {e['rev']} | {e['date']} | {e['change']} | {ap} | `{e['fingerprint']}` |")
        lines.append("")
    open(os.path.join(out, "変更履歴.md"), "w", encoding="utf-8").write("\n".join(lines))
    files = {}
    for r in results:
        for fn in r["files"]:
            files[fn] = rl.sha256(os.path.join(out, fn))
    for fn in ("drawing_list.csv", "図面一覧表.pdf", "変更履歴.md"):
        files[fn] = rl.sha256(os.path.join(out, fn))
    json.dump(dict(date=today, pass_=all(r["pass_"] for r in results),
                   drawings=[{k: v for k, v in r.items() if k != "files"} | {"files": r["files"]} for r in results],
                   sha256=files),
              open(os.path.join(out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


def _list_sheet(results, path, today):
    sh = Sheet(A3)
    sh.frame_and_title([("図名", "図面一覧表"), ("図番", "AC-0000"), ("単位", "mm"), ("投影法", "第三角法")])
    sh.text((20, 280), f"図面一覧表 (出力日 {today})", 5.0, 0, "left", "top")
    cols = [("図番", 22), ("改訂", 12), ("図名", 52), ("区分", 18), ("用紙", 12), ("尺度", 14), ("状態", 34),
            ("対応製品図", 34), ("検図", 14)]
    rows = [[r["number"], r["rev"], r["title"], r["kind"], r["sheet"], r["scale"], r["status"],
             f"{r['product']} 改訂 {r['product_rev']}" if r["product"] else "", "合格" if r["pass_"] else "不合格"] for r in results]
    tool.table(sh, (20, 268), cols, rows, rh=6.5, size=3.0)
    to_pdf(sh, path)


# ------------------------------------------------------------------ approval / revision
def _manifest(out=OUT):
    p = os.path.join(out, "manifest.json")
    if not os.path.exists(p):
        sys.exit("no manifest: run `python -m scripts.release` first")
    return {d["number"]: d for d in json.load(open(p, encoding="utf-8"))["drawings"]}


def approve(numbers, by, out=OUT):
    man = _manifest(out)
    hist = rl.History(HISTORY)
    for n in numbers:
        d = man[n]
        if not d["pass_"]:
            sys.exit(f"{n}: the release gate failed ({d['gates']}); not approved")
        hist.approve(n, d["fingerprint"], by, rl.today())
        print(f"{n} rev {d['rev']} approved by {by}")
    hist.save()


def revise(number, change, out=OUT):
    man = _manifest(out)
    hist = rl.History(HISTORY)
    hist.revise(number, change, man[number]["fingerprint"], rl.today())
    hist.save()
    print(f"{number}: revision {hist.entries(number)[-1]['rev']} started ({change})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "approve", "revise"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--write-history", action="store_true")
    ap.add_argument("--by")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "approve":
        if not a.by:
            sys.exit("--by <name> is required: only a person approves")
        approve(list(_manifest()) if a.all else a.args, a.by)
    elif a.cmd == "revise":
        revise(a.args[0], a.args[1])
    else:
        res = build(a.only, a.write_history)
        bad = [r["number"] for r in res if not r["pass_"]]
        print(f"{len(res)} drawings, {len(res) - len(bad)} pass" + (f", FAIL: {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
