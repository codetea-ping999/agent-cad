"""Parametric bicycle stem (2-bolt steerer clamp, 4-bolt faceplate) in build123d."""
import math, json, sys
from dataclasses import dataclass, asdict, field
from build123d import *

MATERIALS = {  # density g/cm3
    "A6061-T6": 2.70, "A7075-T6": 2.81, "A2014-T6": 2.80, "Ti-6Al-4V": 4.43,
}

@dataclass
class StemParams:
    # interface
    steerer_d: float = 28.6       # 1-1/8"
    bar_d: float = 31.8
    length: float = 90.0          # steerer axis -> bar axis
    angle_deg: float = 6.0        # vs. perpendicular to steerer
    stack_h: float = 40.0
    # sizing
    steerer_wall: float = 3.0
    bar_wall: float = 3.0
    bar_clamp_w: float = 40.0
    body_w: float = 21.0
    body_h: float = 24.0
    body_r: float = 6.0           # corner radius of body section
    body_w_end: float = 20.5       # section at bar end (0 -> same as start; taper follows bending moment)
    body_h_end: float = 22.0
    junction_r: float = 3.0       # fillet where body meets clamps
    light_d: float = 16.5         # lightening bore along stem axis
    # steerer clamp
    slot_w: float = 2.5
    ear_half_w: float = 10.0      # half width of pinch ear (Y)
    pinch_pitch: float = 20.0     # spacing of 2 pinch bolts
    pinch_wall: float = 3.0       # wall between steerer bore and pinch bolt hole
    ear_end_margin: float = 1.5   # material behind / around pinch counterbore
    boss_notch: bool = True       # notch bar-clamp bosses between bolt pairs
    ear_split: bool = True        # pinch ear as two bolt bosses instead of a full-height block
    # faceplate
    fp_gap: float = 1.0           # gap between faceplate and body at split plane
    fp_front: float = 7.0         # faceplate boss depth in front of bar axis
    boss_back: float = 12.0       # body boss depth behind bar axis
    fp_bolt_off: float = 20.2     # bolt offset from bar axis (vertical, local)
    fp_bolt_w: float = 12.5       # bolt offset across (Y)
    boss_margin: float = 6.0      # boss edge beyond faceplate bolt axis
    fp_window_w: float = 12.0      # faceplate centre window (across, Y); 0 -> none
    fp_window_h: float = 20.0      # faceplate centre window (vertical)
    # fasteners (M5)
    bolt_clear_d: float = 5.5
    tap_drill_d: float = 4.2
    cbore_d: float = 8.8
    cbore_depth: float = 5.0
    tap_depth: float = 10.0
    material: str = "A6061-T6"
    density_g_cm3: float = 0.0    # 0 -> from material table
    mass_target_g: float = 145.0

def frame(p):
    a = math.radians(p.angle_deg)
    d = Vector(math.cos(a), 0, math.sin(a))
    n = Vector(-math.sin(a), 0, math.cos(a))
    S = Vector(0, 0, p.stack_h / 2)
    B = S + d * p.length
    return d, n, S, B

def outer(p: StemParams):
    """External shape (no bores/holes): shared by the part model and the CFD model."""
    d, n, S, B = frame(p)
    Y = Vector(0, 1, 0)
    rs_o = p.steerer_d / 2 + p.steerer_wall
    rb_o = p.bar_d / 2 + p.bar_wall
    bar_pl = Plane(origin=B, x_dir=d, z_dir=Y)          # local u=d, v=-n, w=Y
    # --- positive volumes
    steer = Cylinder(rs_o, p.stack_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bx_abs = p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2   # pinch bolt axis |x|
    ear_x = bx_abs + p.cbore_d / 2 + p.ear_end_margin
    if p.ear_split:
        # one boss per pinch bolt; material between and above/below the bolts removed
        eb = p.cbore_d / 2 + p.ear_end_margin            # boss half height
        x0 = rs_o - 2.0                                   # overlap into the clamp ring
        ear = Part()
        for z in (p.stack_h / 2 - p.pinch_pitch / 2, p.stack_h / 2 + p.pinch_pitch / 2):
            ear += Pos(-(x0 + ear_x) / 2, 0, z) * extrude(
                Plane.XZ * RectangleRounded(ear_x - x0, 2 * eb, min(eb, ear_x - x0) / 2 - 0.01), p.ear_half_w, both=True)
    else:
        ear = Pos(-ear_x / 2, 0, 0) * extrude(RectangleRounded(ear_x, 2 * p.ear_half_w, 3), p.stack_h)
    body_pl = Plane(origin=S, x_dir=Y, z_dir=d)
    we, he = (p.body_w_end or p.body_w), (p.body_h_end or p.body_h)
    body = loft([body_pl * RectangleRounded(p.body_w, p.body_h, p.body_r),
                 body_pl.offset(p.length) * RectangleRounded(we, he, min(p.body_r, we / 2 - 0.5, he / 2 - 0.5))])
    clamp = bar_pl * Cylinder(rb_o, p.bar_clamp_w)
    bh = p.fp_bolt_off + p.boss_margin
    boss = bar_pl * Pos((p.fp_front - p.boss_back) / 2, 0, 0) * extrude(
        RectangleRounded(p.fp_front + p.boss_back, 2 * bh, 4), p.bar_clamp_w / 2, both=True)
    part = steer + ear + body + clamp + boss
    # junction fillets (body tube <-> clamps) to reduce stress concentration
    if p.junction_r > 0:
        jx = [e for e in part.edges() if e.geom_type not in (GeomType.LINE, GeomType.CIRCLE)
              and e.center().X > 0]  # body tube junctions only (front of steerer clamp)
        try:
            part = fillet(jx, p.junction_r)
        except Exception as ex:
            print("junction fillet failed:", ex, file=sys.stderr)
    if p.boss_notch:
        # remove bar-clamp boss material between the left/right bolt pairs, down to the clamp ring
        nw, H = 2 * (p.fp_bolt_w - p.boss_margin), 20.0
        for s_ in (-1, 1):
            o = B + d * (-p.boss_back - 1) + n * (s_ * (rb_o + H / 2))
            part -= extrude(Plane(origin=o, x_dir=Y, z_dir=d) * RectangleRounded(nw, H, 3), p.boss_back + p.fp_front + 2)
    return part

def build(p: StemParams):
    d, n, S, B = frame(p)
    Y = Vector(0, 1, 0)
    bar_pl = Plane(origin=B, x_dir=d, z_dir=Y)
    rs_o = p.steerer_d / 2 + p.steerer_wall
    bx_abs = p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2
    ear_x = bx_abs + p.cbore_d / 2 + p.ear_end_margin
    body_pl = Plane(origin=S, x_dir=Y, z_dir=d)
    part = outer(p)
    # --- cuts
    part -= Cylinder(p.steerer_d / 2, p.stack_h * 3)
    part -= bar_pl * Cylinder(p.bar_d / 2, p.bar_clamp_w * 3)
    part -= extrude(body_pl * Circle(p.light_d / 2), p.length)
    part -= Pos(-(ear_x + 5) / 2, 0, p.stack_h / 2) * Box(ear_x + 5, p.slot_w, p.stack_h * 3)
    bx = -bx_abs
    for z in (p.stack_h / 2 - p.pinch_pitch / 2, p.stack_h / 2 + p.pinch_pitch / 2):
        side = Plane(origin=(bx, p.ear_half_w, z), z_dir=(0, -1, 0))
        part -= side * Cylinder(p.cbore_d / 2, 2 * p.cbore_depth)                      # counterbore
        part -= Plane(origin=(bx, p.slot_w / 2 - 0.01, z), z_dir=(0, 1, 0)) * Cylinder(
            p.bolt_clear_d / 2, p.ear_half_w, align=(Align.CENTER, Align.CENTER, Align.MIN))  # clearance
        part -= Plane(origin=(bx, 0.01, z), z_dir=(0, -1, 0)) * Cylinder(
            p.tap_drill_d / 2, p.ear_half_w, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # --- split faceplate
    big = 400
    front = bar_pl * Pos(big / 2 + p.fp_gap / 2, 0, 0) * Box(big, big, big)
    back = bar_pl * Pos(-big / 2 - p.fp_gap / 2, 0, 0) * Box(big, big, big)
    faceplate = part & front
    body_part = part & back
    for sv in (-1, 1):
        for sw in (-1, 1):
            loc = Vector(0, sv * p.fp_bolt_off, sw * p.fp_bolt_w)
            # faceplate: clearance + counterbore from front face
            fpl = Plane(origin=bar_pl.from_local_coords((p.fp_front + 0.01, loc.Y, loc.Z)), z_dir=-d)
            faceplate -= fpl * Cylinder(p.cbore_d / 2, 2 * 2.5)
            faceplate -= fpl * Cylinder(p.bolt_clear_d / 2, 40, align=(Align.CENTER, Align.CENTER, Align.MIN))
            # body: tap drill
            bpl = Plane(origin=bar_pl.from_local_coords((0, loc.Y, loc.Z)), z_dir=-d)
            body_part -= bpl * Cylinder(p.tap_drill_d / 2, p.tap_depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
    if p.fp_window_w > 0 and p.fp_window_h > 0:
        # centre window through the faceplate, machined from the front
        win = Plane(origin=B, x_dir=Vector(0, 1, 0), z_dir=d)
        r = min(p.fp_window_w, p.fp_window_h) / 2 - 0.01
        faceplate -= extrude(win * RectangleRounded(p.fp_window_w, p.fp_window_h, r), 60)
    return body_part, faceplate

def cfd_solid(p: StemParams, bar_len=160.0, steer_below=30.0, steer_above=3.0, head_h=5.0):
    """Assembled, watertight shape for CFD: stem (gap/holes filled, bolt heads as installed)
    + handlebar stub + steerer stub. Internal features that see no airflow are omitted."""
    d, n, S, B = frame(p)
    Y = Vector(0, 1, 0)
    bar_pl = Plane(origin=B, x_dir=d, z_dir=Y)
    part = outer(p)
    if p.fp_window_w > 0 and p.fp_window_h > 0:
        win = Plane(origin=B, x_dir=Y, z_dir=d)
        r = min(p.fp_window_w, p.fp_window_h) / 2 - 0.01
        part -= extrude(win * RectangleRounded(p.fp_window_w, p.fp_window_h, r), 60)
    protr = head_h - 2.5   # faceplate counterbore depth is 2.5
    if protr > 0:
        for sv in (-1, 1):
            for sw in (-1, 1):
                c = bar_pl.from_local_coords((p.fp_front, sv * p.fp_bolt_off, sw * p.fp_bolt_w))
                part += Plane(origin=c, z_dir=d) * Cylinder(p.cbore_d / 2 - 0.2, protr, align=(Align.CENTER, Align.CENTER, Align.MIN))
    part += bar_pl * Cylinder(p.bar_d / 2, bar_len)
    part += Pos(0, 0, -steer_below) * Cylinder(p.steerer_d / 2, steer_below + p.stack_h + steer_above,
                                               align=(Align.CENTER, Align.CENTER, Align.MIN))
    return part

def density(p):
    return p.density_g_cm3 or MATERIALS[p.material]

def checks(p, body_part, faceplate):
    d, n, S, B = frame(p)
    rho = density(p)
    res = {}
    def solid_count(s): return len(s.solids())
    res["body_valid"] = body_part.is_valid and solid_count(body_part) == 1
    res["faceplate_valid"] = faceplate.is_valid and solid_count(faceplate) == 1
    # mating parts
    bar = Plane(origin=B, x_dir=d, z_dir=(0, 1, 0)) * Cylinder(p.bar_d / 2 - 0.01, 200)
    steerer = Pos(0, 0, -50) * Cylinder(p.steerer_d / 2 - 0.01, p.stack_h + 100, align=(Align.CENTER, Align.CENTER, Align.MIN))
    def inter(a, b):
        x = a & b
        return 0.0 if x is None or not x.solids() else x.volume
    res["interference_mm3"] = {
        "body_vs_faceplate": round(inter(body_part, faceplate), 3),
        "bar_vs_stem": round(inter(bar, body_part) + inter(bar, faceplate), 3),
        "steerer_vs_stem": round(inter(steerer, body_part), 3),
    }
    # design-rule walls (analytic, mm)
    bx = p.steerer_d / 2 + p.pinch_wall + p.bolt_clear_d / 2
    we, he = (p.body_w_end or p.body_w), (p.body_h_end or p.body_h)
    res["min_wall_mm"] = {
        "steerer_clamp": p.steerer_wall,
        "bar_clamp": p.bar_wall,
        "pinch_bolt_to_bore": round(bx - p.bolt_clear_d / 2 - p.steerer_d / 2, 2),
        "fp_bolt_to_bar_bore": round(p.fp_bolt_off - p.bolt_clear_d / 2 - p.bar_d / 2, 2),
        "fp_cbore_to_boss_edge": round(p.boss_margin - p.cbore_d / 2, 2),
        "pinch_cbore_to_ear_edge": p.ear_end_margin,
        **({"fp_window_to_cbore": round(min(p.fp_bolt_w - p.fp_window_w / 2, p.fp_bolt_off - p.fp_window_h / 2) - p.cbore_d / 2, 2)}
           if p.fp_window_w > 0 else {}),
        "pinch_cbore_to_ear_end": p.ear_end_margin,
        "body_side_around_lightening": round((min(p.body_w, we) - p.light_d) / 2, 2),
        "body_top_around_lightening": round((min(p.body_h, he) - p.light_d) / 2, 2),
    }
    res["thread_engagement_mm"] = {"pinch": round(p.ear_half_w - p.slot_w / 2, 2), "faceplate": p.tap_depth}
    mass = (body_part.volume + faceplate.volume) / 1000 * rho
    res["mass_g"] = {"body": round(body_part.volume / 1000 * rho, 1), "faceplate": round(faceplate.volume / 1000 * rho, 1), "total_excl_bolts": round(mass, 1)}
    MIN_WALL = 1.5
    res["pass"] = bool(res["body_valid"] and res["faceplate_valid"]
                       and all(v == 0 for v in res["interference_mm3"].values())
                       and min(res["min_wall_mm"].values()) >= MIN_WALL
                       and min(res["thread_engagement_mm"].values()) >= 1.5 * 5
                       and mass <= p.mass_target_g)
    return res

def write_step_ap242(shape, path):
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.Interface import Interface_Static
    Interface_Static.SetIVal_s("write.step.schema", 5)
    w = STEPControl_Writer(); w.Model(True)
    w.Transfer(shape.wrapped, STEPControl_AsIs)
    assert int(w.Write(path)) == 1

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for k, v in asdict(StemParams()).items():
        ap.add_argument("--" + k, type=type(v), default=v)
    ap.add_argument("--out", default="out/stem")
    a = vars(ap.parse_args()); out = a.pop("out")
    p = StemParams(**a)
    body_part, faceplate = build(p)
    r = checks(p, body_part, faceplate)
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_step_ap242(body_part, out + "_body.step")
    write_step_ap242(faceplate, out + "_faceplate.step")
    write_step_ap242(Compound([body_part, faceplate]), out + "_assembly.step")
    rep = {"params": asdict(p), "density_g_cm3": density(p), "checks": r}
    json.dump(rep, open(out + "_report.json", "w"), ensure_ascii=False, indent=1)
    if not r["pass"]:
        print("CHECK FAILED", file=sys.stderr)
    print(json.dumps(r, ensure_ascii=False))
