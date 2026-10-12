import sys, json, os
from agent_cad.parts import stem
from agent_cad import render
from agent_cad.parts.stem import StemParams
os.makedirs("out", exist_ok=True)
cases = {"stem90_6061": StemParams(),
         "stem110_-17_7075": StemParams(length=110, angle_deg=-17, material="A7075-T6", mass_target_g=165)}
for name, p in cases.items():
    b, f = stem.build(p)
    r = stem.checks(p, b, f)
    for part, nm in ((b, "body"), (f, "faceplate")):
        stem.write_step_ap242(part, f"out/{name}_{nm}.step")
    stem.write_step_ap242(stem.Compound([b, f]), f"out/{name}_assembly.step")
    json.dump({"params": stem.asdict(p), "checks": r}, open(f"out/{name}_report.json", "w"), ensure_ascii=False, indent=1)
    render.render([(b, '#9aa4b1'), (f, '#c98a3d')], f"out/{name}.png",
                  titles=["Iso", "Rear iso", "Top", "Side"])
    print(name, json.dumps(r["mass_g"]), "pass=", r["pass"])
# reimport check
from build123d import import_step
b = import_step("out/stem90_6061_body.step"); print("reimport vol", round(b.volume, 2))
