"""Injection-moulded housing and its mould -> out/.

For each case: the part, the shot (part with gate, runner, sprue and cold-slug well),
the fixed and moving halves, the slide and the mould assembly (with the angle pin,
core pins, ejector sleeves and pins) as AP242 STEP, each checked with the STEP
exchange check; the moulding/mould verification report; the JIS part drawing
(PDF + DXF, checked); preview renders.
"""
import sys, json, os, hashlib
from build123d import scale
from agent_cad.parts import housing, housing_drawing
from agent_cad.process import molding as mo
from agent_cad.verify.molding_check import verify_molding
from agent_cad.verify.step_check import check_step, summary
from agent_cad.step_io import write_step, PartSpec
from agent_cad.drawing.render import to_pdf, to_dxf
from agent_cad.drawing.check import check_sheet
from agent_cad import render

CASES = {
    "housing_std": housing.HousingParams(),
    "housing_pbt_long": housing.HousingParams(length=130.0, boss_pitch_x=106.0, material="PBT-GF30",
                                              mass_target_g=110.0),
}
PART_RGB, STEEL_RGB, SLIDE_RGB, PIN_RGB = (0.18, 0.19, 0.2), (0.42, 0.47, 0.52), (0.55, 0.5, 0.42), (0.85, 0.65, 0.2)


def doc_no(p):
    """Unique to the parameter set and the geometry revision of the model code."""
    key = dict(housing.asdict(p), _geometry_rev=housing.GEOMETRY_REV)
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12].upper()
    return f"AC-HSG-{h}"


def specs(p, ms):
    """Products named after the part's drawing number (S = shot, M = mould)."""
    dn, rho = doc_no(p), housing.density(p)
    steel, srho = mo.MOULD_STEEL
    s = {"part": PartSpec(f"{dn}_HOUSING", ms["part"], PART_RGB, p.material, rho),
         "shot": PartSpec(f"{dn}-S1_SHOT_WITH_FEED", ms["shot"], PART_RGB, p.material, rho),
         "fixed": PartSpec(f"{dn}-M1_FIXED_HALF", ms["fixed"], STEEL_RGB, steel, srho),
         "moving": PartSpec(f"{dn}-M2_MOVING_HALF", ms["moving"], STEEL_RGB, steel, srho),
         "slide": PartSpec(f"{dn}-M3_SLIDE", ms["slide"], SLIDE_RGB, steel, srho),
         "angle_pin": PartSpec(f"{dn}-M4_ANGLE_PIN", ms["angle_pin"], PIN_RGB, steel, srho)}
    s["core_pins"] = [PartSpec(f"{dn}-M5_CORE_PIN_{i + 1}", c, PIN_RGB, steel, srho) for i, c in enumerate(ms["core_pins"])]
    s["sleeves"] = [PartSpec(f"{dn}-M6_EJECTOR_SLEEVE_{i + 1}", c, PIN_RGB, steel, srho) for i, c in enumerate(ms["sleeves"])]
    s["pins"] = [PartSpec(f"{dn}-M7_EJECTOR_PIN_{i + 1}", c, PIN_RGB, steel, srho) for i, c in enumerate(ms["pins"])]
    return s


def make(name, p, mp=mo.MouldParams(), out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    ms = mo.mould_set(p, mp)
    rep = verify_molding(p, mp, ms=ms)
    s = specs(p, ms)
    k = mo.shrink_factor(p)
    shot_k = PartSpec(s["shot"].name, scale(ms["shot"], k), PART_RGB, p.material, housing.density(p))
    files = {f"{out}/{name}.step": ([s["part"]], None),
             f"{out}/{name}_shot.step": ([s["shot"]], None),
             f"{out}/{name}_mould_fixed.step": ([s["fixed"]], None),
             f"{out}/{name}_mould_moving.step": ([s["moving"]], None),
             f"{out}/{name}_mould_slide.step": ([s["slide"]], None),
             f"{out}/{name}_mould_assembly.step": ([s["fixed"], s["moving"], s["slide"], s["angle_pin"], *s["core_pins"],
                                                   *s["sleeves"], *s["pins"], shot_k], f"{name.upper()}_MOULD")}
    xchk = []
    for path, (parts, an) in files.items():
        write_step(parts, path, an)
        x = check_step(path, parts, an)
        print(summary(x), file=sys.stderr)
        xchk.append(x)
    sheet = housing_drawing.part_sheet(p, ms["part"], dwg_no=doc_no(p))
    to_pdf(sheet, f"{out}/{name}_drawing.pdf")
    to_dxf(sheet, f"{out}/{name}_drawing.dxf")
    dchk = check_sheet(sheet)
    dchk["dims"] = sheet.meta["dims"]
    dchk["drawing_no"] = sheet.meta["title_block"]["図番"]
    print(f"{name}_drawing: {'PASS' if dchk['pass'] else 'FAIL'}", file=sys.stderr)
    rep.update(params=housing.asdict(p), mould_params=mo.asdict(mp), step_exchange=xchk, drawing=dchk)
    rep["pass"] = rep["pass"] and all(x["pass"] for x in xchk) and dchk["pass"]
    json.dump(rep, open(f"{out}/{name}_molding_report.json", "w"), ensure_ascii=False, indent=1, default=str)
    if do_render:
        render.render([(ms["part"], '#4a4d52')], f"{out}/{name}.png", views=((25, -60), (-30, 120), (-90, -90), (0, -90)),
                      titles=["Iso", "Iso from below", "Window side (-Z)", "Opening side (-Y)"])
        render.render([(ms["shot"], '#4a4d52')], f"{out}/{name}_shot.png", titles=["Iso", "Rear iso", "Top", "Side"])
        # mould internals around the shot: slide (brown), angle pin, core pins, sleeves and
        # ejector pins (orange), cooling channels and bubblers (blue)
        render.render([(scale(ms["shot"], k), '#4a4d52'), (ms["slide"], '#8c806b'), (ms["angle_pin"], '#d9a633')] +
                      [(c, '#d9a633') for c in ms["pins"] + ms["sleeves"] + ms["core_pins"]] +
                      [(c, '#3d7cc9') for c in ms["channels_fixed"] + ms["channels_moving"] + ms["bubblers"]],
                      f"{out}/{name}_mould_layout.png", titles=["Iso", "Rear iso", "Top", "Side"])
    return rep


if __name__ == "__main__":
    ok = True
    for name, p in CASES.items():
        rep = make(name, p)
        pr = rep["process"]
        print(name, f"part {pr['mass_part_g']} g, shot {pr['mass_shot_g']} g, cooling {pr['cooling_time_s']} s,",
              f"clamp ~{pr['clamp_force_required_t']} t -> {pr['machine_t']} t machine", "pass=", rep["pass"], rep["failed"])
        ok &= rep["pass"]
    sys.exit(0 if ok else 1)
