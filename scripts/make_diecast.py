"""Die-cast bracket and its die -> out/.

For each case: the finished part, the casting, the shot (casting with runner, gate,
biscuit and overflows), the moving and cover dies and the die assembly with the
ejector pins as AP242 STEP (each checked with the STEP exchange check), the
casting/die verification report, and preview renders.
"""
import sys, json, os, hashlib
from build123d import scale
from agent_cad.parts import bracket
from agent_cad.process import diecast as dc
from agent_cad.verify.diecast_check import verify_diecast
from agent_cad.verify.step_check import check_step, summary
from agent_cad.step_io import write_step, PartSpec
from agent_cad import render

CASES = {
    "bracket_std": bracket.BracketParams(),
    "bracket_long": bracket.BracketParams(plate_l=140.0, boss_pitch_x=116.0),
}
CAST_RGB, DIE_RGB, PIN_RGB = (0.75, 0.76, 0.78), (0.42, 0.47, 0.52), (0.85, 0.65, 0.2)


def doc_no(p):
    h = hashlib.sha1(json.dumps(bracket.asdict(p), sort_keys=True, default=str).encode()).hexdigest()[:12].upper()
    return f"AC-BRKT-{h}"


def specs(p, ds):
    """Products named after the part's drawing number (C = casting, D = die)."""
    dn, rho = doc_no(p), bracket.density(p)
    steel, srho = dc.DIE_STEEL
    s = {"finished": PartSpec(f"{dn}_BRACKET", ds["finished"], CAST_RGB, p.material, rho),
         "casting": PartSpec(f"{dn}-C1_CASTING", ds["casting"], CAST_RGB, p.material, rho),
         "shot": PartSpec(f"{dn}-C2_SHOT_WITH_GATING", ds["shot"], CAST_RGB, p.material, rho),
         "moving": PartSpec(f"{dn}-D1_DIE_MOVING", ds["moving"], DIE_RGB, steel, srho),
         "cover": PartSpec(f"{dn}-D2_DIE_COVER", ds["cover"], DIE_RGB, steel, srho)}
    s["pins"] = [PartSpec(f"{dn}-D3_EJECTOR_PIN_{i + 1}", pin, PIN_RGB, steel, srho) for i, pin in enumerate(ds["pins"])]
    return s


def make(name, p, dp=dc.DiecastParams(), out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    ds = dc.die_set(p, dp)
    rep = verify_diecast(p, dp, ds=ds)
    s = specs(p, ds)
    files = {f"{out}/{name}.step": ([s["finished"]], None),
             f"{out}/{name}_casting.step": ([s["casting"]], None),
             f"{out}/{name}_shot.step": ([s["shot"]], None),
             f"{out}/{name}_die_moving.step": ([s["moving"]], None),
             f"{out}/{name}_die_cover.step": ([s["cover"]], None),
             f"{out}/{name}_die_assembly.step": ([s["moving"], s["cover"], *s["pins"], s["shot"]],
                                                 f"{name.upper()}_DIECAST_DIE_SET")}
    xchk = []
    for path, (parts, an) in files.items():
        write_step(parts, path, an)
        x = check_step(path, parts, an)
        print(summary(x), file=sys.stderr)
        xchk.append(x)
    rep.update(params=bracket.asdict(p), diecast_params=dc.asdict(dp), step_exchange=xchk)
    rep["pass"] = rep["pass"] and all(x["pass"] for x in xchk)
    json.dump(rep, open(f"{out}/{name}_diecast_report.json", "w"), ensure_ascii=False, indent=1, default=str)
    if do_render:
        render.render([(ds["finished"], '#c4c7cc')], f"{out}/{name}.png", titles=["Iso", "Rear iso", "Top", "Side"])
        render.render([(ds["shot"], '#c4c7cc')], f"{out}/{name}_shot.png", titles=["Iso", "Rear iso", "Top", "Side"])
        # die internals around the shot: ejector pins (orange) and cooling channels (blue);
        # the solid die blocks would hide all of it
        k = dc.shrink_factor(dp)
        render.render([(scale(ds["shot"], k), '#c4c7cc')] + [(pin, '#d9a633') for pin in ds["pins"]] +
                      [(c, '#3d7cc9') for c in ds["channels_moving"] + ds["channels_cover"]],
                      f"{out}/{name}_die_layout.png", titles=["Iso", "Rear iso", "Top", "Side"])
    return rep


if __name__ == "__main__":
    ok = True
    for name, p in CASES.items():
        rep = make(name, p)
        pr = rep["process"]
        print(name, f"casting {pr['mass_casting_g']} g, shot {pr['mass_shot_g']} g, yield {pr['material_yield_pct']} %,",
              f"clamp ~{pr['clamp_force_required_t']} t -> {pr['machine_t']} t machine", "pass=", rep["pass"], rep["failed"])
        ok &= rep["pass"]
    sys.exit(0 if ok else 1)
