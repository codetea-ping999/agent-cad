"""Build the stem variants, write AP242 STEP (body, faceplate, assembly),
run design checks and the STEP exchange check, generate JIS drawings
(PDF + DXF) and check them, render previews -> out/."""
import sys, json, os
from agent_cad.parts import stem
from agent_cad.parts.stem import StemParams
from agent_cad import render
from agent_cad.step_io import write_step, PartSpec
from agent_cad.verify.step_check import check_step, summary
from agent_cad.parts import stem_drawing
from agent_cad.drawing.render import to_pdf, to_dxf
from agent_cad.drawing.check import check_sheet

CASES = {"stem90_6061": StemParams(),
         "stem110_-17_7075": StemParams(length=110, angle_deg=-17, material="A7075-T6", mass_target_g=165)}
BODY_RGB, FP_RGB = (0.604, 0.643, 0.694), (0.788, 0.541, 0.239)


def part_specs(p, b, f):
    """STEP products named by drawing number, so the controlling model and its
    drawing ("図示なき外形形状は3Dモデル (同図番) による") identify each other."""
    dn, rho = stem_drawing.doc_no(p), stem.density(p)
    return (PartSpec(f"{dn}-01_STEM_BODY", b, BODY_RGB, p.material, rho),
            PartSpec(f"{dn}-02_STEM_FACEPLATE", f, FP_RGB, p.material, rho))


def make(name, p, out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    b, f = stem.build(p)
    r = stem.checks(p, b, f)
    rho = stem.density(p)
    body, fp = part_specs(p, b, f)
    asm = name.upper()
    files = {f"{out}/{name}_body.step": ([body], None),
             f"{out}/{name}_faceplate.step": ([fp], None),
             f"{out}/{name}_assembly.step": ([body, fp], asm)}
    xchk = []
    for path, (parts, an) in files.items():
        write_step(parts, path, an)
        rep = check_step(path, parts, an)
        print(summary(rep), file=sys.stderr)
        xchk.append(rep)
    drawings = []
    for sheet, part in ((stem_drawing.body_sheet(p, b), "body"), (stem_drawing.faceplate_sheet(p, f), "faceplate")):
        to_pdf(sheet, f"{out}/{name}_{part}_drawing.pdf")
        to_dxf(sheet, f"{out}/{name}_{part}_drawing.dxf")
        dc = check_sheet(sheet)
        dc["dims"] = sheet.meta["dims"]
        dc["drawing_no"] = sheet.meta["title_block"]["図番"]
        print(f"{name}_{part}_drawing: {'PASS' if dc['pass'] else 'FAIL'}", file=sys.stderr)
        drawings.append(dc)
    report = {"params": stem.asdict(p), "density_g_cm3": rho, "checks": r,
              "step_exchange": xchk, "drawings": drawings,
              "pass": r["pass"] and all(x["pass"] for x in xchk) and all(d["pass"] for d in drawings)}
    json.dump(report, open(f"{out}/{name}_report.json", "w"), ensure_ascii=False, indent=1)
    if do_render:
        render.render([(b, '#9aa4b1'), (f, '#c98a3d')], f"{out}/{name}.png", titles=["Iso", "Rear iso", "Top", "Side"])
    return report


if __name__ == "__main__":
    ok = True
    for name, p in CASES.items():
        rep = make(name, p)
        print(name, json.dumps(rep["checks"]["mass_g"]), "pass=", rep["pass"])
        ok &= rep["pass"]
    sys.exit(0 if ok else 1)
