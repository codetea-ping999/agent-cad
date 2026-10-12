"""Build the stem variants, write AP242 STEP (body, faceplate, assembly),
run design checks and the STEP exchange check, render previews -> out/."""
import sys, json, os
from agent_cad.parts import stem
from agent_cad.parts.stem import StemParams
from agent_cad import render
from agent_cad.step_io import write_step, PartSpec
from agent_cad.verify.step_check import check_step, summary

CASES = {"stem90_6061": StemParams(),
         "stem110_-17_7075": StemParams(length=110, angle_deg=-17, material="A7075-T6", mass_target_g=165)}
BODY_RGB, FP_RGB = (0.604, 0.643, 0.694), (0.788, 0.541, 0.239)


def make(name, p, out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    b, f = stem.build(p)
    r = stem.checks(p, b, f)
    rho = stem.density(p)
    body = PartSpec("STEM_BODY", b, BODY_RGB, p.material, rho)
    fp = PartSpec("STEM_FACEPLATE", f, FP_RGB, p.material, rho)
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
    report = {"params": stem.asdict(p), "density_g_cm3": rho, "checks": r,
              "step_exchange": xchk, "pass": r["pass"] and all(x["pass"] for x in xchk)}
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
