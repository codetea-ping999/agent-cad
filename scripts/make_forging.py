"""Forging and forging dies for the stem variants -> out/.

For each case: the trimmed forging, the forging with flash, the upper and lower
dies and the die assembly as AP242 STEP (each checked with the STEP exchange
check), the forging/die verification report, and preview renders.
"""
import sys, json, os
from build123d import Pos
from agent_cad.parts import stem
from agent_cad.process import forging as fg
from agent_cad.verify.forging_check import verify_forging
from agent_cad.verify.step_check import check_step, summary
from agent_cad.step_io import write_step, PartSpec
from agent_cad.parts.stem_drawing import doc_no
from agent_cad import render
from scripts.make_stem import CASES

FORGING_RGB, DIE_RGB = (0.788, 0.541, 0.239), (0.42, 0.47, 0.52)


def specs(p, ds):
    """Products named after the part's drawing number (F = forging, D = die)."""
    dn, rho = doc_no(p), stem.density(p)
    steel, srho = fg.DIE_STEEL
    return {"blank": PartSpec(f"{dn}-F1_FORGING_TRIMMED", ds["blank"], FORGING_RGB, p.material, rho),
            "forging": PartSpec(f"{dn}-F2_FORGING_WITH_FLASH", ds["forging"], FORGING_RGB, p.material, rho),
            "upper": PartSpec(f"{dn}-D1_FORGING_DIE_UPPER", ds["upper"], DIE_RGB, steel, srho),
            "lower": PartSpec(f"{dn}-D2_FORGING_DIE_LOWER", ds["lower"], DIE_RGB, steel, srho)}


def make(name, p, fp=fg.ForgingParams(), out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    ds = fg.die_set(p, fp)
    rep = verify_forging(p, fp, ds=ds)
    s = specs(p, ds)
    files = {f"{out}/{name}_forging.step": ([s["blank"]], None),
             f"{out}/{name}_forging_flash.step": ([s["forging"]], None),
             f"{out}/{name}_die_upper.step": ([s["upper"]], None),
             f"{out}/{name}_die_lower.step": ([s["lower"]], None),
             f"{out}/{name}_die_assembly.step": ([s["upper"], s["lower"], s["forging"]], f"{name.upper()}_FORGING_DIE_SET")}
    xchk = []
    for path, (parts, an) in files.items():
        write_step(parts, path, an)
        x = check_step(path, parts, an)
        print(summary(x), file=sys.stderr)
        xchk.append(x)
    rep.update(params=stem.asdict(p), forging_params=fg.asdict(fp), step_exchange=xchk)
    rep["pass"] = rep["pass"] and all(x["pass"] for x in xchk)
    json.dump(rep, open(f"{out}/{name}_forging_report.json", "w"), ensure_ascii=False, indent=1, default=str)
    if do_render:
        b, f = stem.build(p)
        render.render([(ds["forging"], '#c98a3d')], f"{out}/{name}_forging.png", titles=["Iso", "Rear iso", "Top", "Side"])
        lift = ds["upper"].bounding_box().size.Y * 0.6
        render.render([(Pos(0, lift, 0) * ds["upper"], '#6b7884'), (ds["forging"], '#c98a3d'), (Pos(0, -lift, 0) * ds["lower"], '#6b7884')],
                      f"{out}/{name}_die_set.png", titles=["Iso", "Rear iso", "Top", "Side"])
    return rep


if __name__ == "__main__":
    ok = True
    for name, p in CASES.items():
        rep = make(name, p)
        pr = rep["process"]
        print(name, f"forging {pr['mass_forging_with_flash_g']} g, yield {pr['material_yield_pct']} %,",
              f"press ~{pr['press_load_t']} t", "pass=", rep["pass"], rep["failed"])
        ok &= rep["pass"]
    sys.exit(0 if ok else 1)
