"""Road drop handlebar and its tube-forming tools -> out/.

For each case: the bar as AP242 STEP (checked with the STEP exchange check); the design
verification (fit, walls, ISO 4210-5 style strength by solid FEA, stiffness); the
process and tooling verification (blank, swaging, each bend's clamp, sweep, springback,
mandrel and torque, tool overlap); the bending set-up of the hood bend B2 (bend die,
contoured clamp pair, pressure die, wiper, ball mandrel and the tube in it) and the
swaging tools as STEP; the JIS part drawing (PDF + DXF, checked); preview renders.
"""
import sys, json, os, hashlib
from functools import lru_cache
from build123d import Compound
from agent_cad.parts import handlebar as hb, handlebar_drawing
from agent_cad.process import tube_bending as tb
from agent_cad.verify import handlebar_verify as hv
from agent_cad.verify.tube_check import verify_process
from agent_cad.verify.step_check import check_step, summary
from agent_cad.step_io import write_step, PartSpec
from agent_cad.drawing.render import to_pdf, to_dxf
from agent_cad.drawing.check import check_sheet
from agent_cad import render

CASES = {
    "drop420_7075": hb.HandlebarParams(),
}
BAR_RGB, STEEL_RGB, BRONZE_RGB, TUBE_RGB = (0.12, 0.12, 0.13), (0.42, 0.47, 0.52), (0.72, 0.55, 0.3), (0.75, 0.76, 0.78)
SETUP = (1, "B2")                 # the set-up drawn and exported: right hood bend, contoured clamp


def doc_no(p):
    """Unique to the parameter set and the geometry revision of the model code."""
    key = dict(hb.asdict(p), _geometry_rev=hb.GEOMETRY_REV)
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12].upper()
    return f"AC-HB-{h}"


@lru_cache(None)
def case(name, solid=True):
    """Bar, design report, tools and process report of a case (built once)."""
    p, tp = CASES[name], tb.TubeProcessParams()
    bar = hb.build(p)
    rep = hv.verify(p, bar, solid=solid)
    tools = tb.tool_set(p, tp, SETUP)
    prep = verify_process(p, tp, tools)
    return p, tp, bar, rep, tools, prep


def setup_specs(p, tools, dn):
    """The B2 bending set-up as STEP products (T = tool, W = work)."""
    st, wp = tb.TOOL_STEEL[0].split(" (")[0], tb.WIPER_MAT[0]
    s = [PartSpec(f"{dn}-T1_BEND_DIE", tools["bend_die"], STEEL_RGB, st, tb.STEEL_DENSITY),
         PartSpec(f"{dn}-T2_CLAMP_DIE", tools["clamp"], STEEL_RGB, st, tb.STEEL_DENSITY),
         PartSpec(f"{dn}-T3_CLAMP_INSERT", tools["insert"], STEEL_RGB, st, tb.STEEL_DENSITY),
         PartSpec(f"{dn}-T4_PRESSURE_DIE", tools["pressure_die"], STEEL_RGB, st, tb.STEEL_DENSITY),
         PartSpec(f"{dn}-T5_WIPER_DIE", tools["wiper"], BRONZE_RGB, wp, 7.6),
         PartSpec(f"{dn}-T6_MANDREL_SHANK", tools["mandrel_shank"], STEEL_RGB, st, tb.STEEL_DENSITY)]
    s += [PartSpec(f"{dn}-T6_MANDREL_BALL_{i + 1}", b, STEEL_RGB, st, tb.STEEL_DENSITY)
          for i, b in enumerate(tools["mandrel_balls"])]
    s.append(PartSpec(f"{dn}-W1_TUBE_AT_B2", tools["tube"], TUBE_RGB, tb.BENDING_TEMPER.get(p.material, p.material),
                      hb.density(p)))
    return s


def make(name, out="out", do_render=True):
    os.makedirs(out, exist_ok=True)
    p, tp, bar, rep, tools, prep = case(name)
    dn = doc_no(p)
    st = tb.SWAGE_MAT[0].split(" (")[0]
    bar_s = PartSpec(f"{dn}_HANDLEBAR", bar, BAR_RGB, p.material, hb.density(p))
    files = {f"{out}/{name}.step": ([bar_s], None),
             f"{out}/{name}_bending_setup_B2.step": (setup_specs(p, tools, dn), f"{name.upper()}_BENDING_SETUP_B2"),
             f"{out}/{name}_swage_tools.step": ([PartSpec(f"{dn}-T7_SWAGE_DIE_HALF", tools["swage_die"], STEEL_RGB, st, 7.8),
                                                 PartSpec(f"{dn}-T8_SWAGE_MANDREL", tools["swage_mandrel"], STEEL_RGB, st, 7.8)],
                                                f"{name.upper()}_SWAGE_TOOLS")}
    xchk = []
    for path, (parts, an) in files.items():
        write_step(parts, path, an)
        x = check_step(path, parts, an)
        print(summary(x), file=sys.stderr)
        xchk.append(x)
    sheet = handlebar_drawing.part_sheet(p, bar, dn)
    to_pdf(sheet, f"{out}/{name}_drawing.pdf")
    to_dxf(sheet, f"{out}/{name}_drawing.dxf")
    dchk = check_sheet(sheet)
    dchk["dims"] = sheet.meta["dims"]
    dchk["drawing_no"] = sheet.meta["title_block"]["図番"]
    print(f"{name}_drawing: {'PASS' if dchk['pass'] else 'FAIL'}", file=sys.stderr)
    full = dict(design=rep, process=prep, step_exchange=xchk, drawing=dchk, process_params=tb.asdict(tp))
    full["pass"] = rep["pass"] and prep["pass"] and all(x["pass"] for x in xchk) and dchk["pass"]
    full["failed"] = rep["failed"] + prep["failed"] + ([] if dchk["pass"] else ["drawing"])
    json.dump(full, open(f"{out}/{name}_report.json", "w"), ensure_ascii=False, indent=1, default=str)
    if do_render:
        render.render([(bar, '#2a2b2e')], f"{out}/{name}.png", views=((25, -60), (20, 120), (90, -90), (0, -90)),
                      titles=["Iso", "Rear iso", "Top", "Front"])
        render.render([(tools["bend_die"], '#6b7785'), (tools["clamp"], '#8a96a3'), (tools["insert"], '#8a96a3'),
                       (tools["pressure_die"], '#6b7785'), (tools["wiper"], '#b88c4d'), (tools["tube"], '#c4c7cc')] +
                      [(b, '#d9a633') for b in tools["mandrel_balls"]],
                      f"{out}/{name}_bending_setup_B2.png", views=((35, -60), (25, 150), (90, -90), (0, -90)),
                      titles=["Iso", "Rear iso", "Top (die axis)", "Side"])
    return full


if __name__ == "__main__":
    ok = True
    for name in CASES:
        r = make(name)
        d, pr = r["design"], r["process"]
        sf = min(x["safety_factor"] for x in d["rules"] if x["check"].startswith("strength."))
        print(name, f"{d['mass_g']:.1f} g, stiffness {d['stiffness_mm_per_100N']:.2f} mm/100 N, min SF {sf},",
              f"blank {pr['blank']['cut_length']} mm, swaging {pr['swaging']['passes']} passes",
              "pass=", r["pass"], r["failed"])
        ok &= r["pass"]
    sys.exit(0 if ok else 1)
