"""Mesh convergence studies of the solid FEA that decides the stem's and the handlebar's
strength checks.

    rep = stem_study(StemParams())                 # 4 meshes, every ISO 4210 style load case
    rep = handlebar_study(HandlebarParams())

Each study meshes the part on a family of meshes refined uniformly by sqrt(2) per level
(the global size h and the curvature refinement together, so that fillets and bores are
refined as much as the bulk), with the production mesh of the verification loop as one
level and one level finer than it. For each load case and each result it reports the
values per level, the Richardson extrapolation and GCI over the three finest meshes, and
how far the production mesh is from the extrapolated value. Mesh quality is reported per
level. The largest meshes are solved with AMG-preconditioned CG (agent_cad/fea/static.py).

Results:
  max_disp_mm           largest displacement: smooth, converges fast
  vm_p999_MPa           99.9th percentile of element peak von Mises: the checked value
  vm_peak_away_MPa      largest element von Mises outside the support zone
  vm_max_MPa            largest element value anywhere: includes the re-entrant edge of the
                        rigid support, which is singular (grows without bound on refinement);
                        reported to show why it is not used for the checks
"""
import math, time
import numpy as np
from ..parts import stem as stem_mod
from ..parts import handlebar as hb
from ..materials import MATERIALS
from ..fea.static import mesh_shape, solve_cases
from ..fea.convergence import mesh_quality, richardson
from . import stem_verify as sv
from . import handlebar_verify as hv

R = math.sqrt(2)
STEM_PROD = (2.0, 12)               # (h, curvature points) of stem_verify.verify()
BAR_PROD = (hv.SOLID_H, 12)          # of handlebar_verify.solid_fea()
AWAY_MM = 2.0                        # support zone left out of vm_peak_away
AMG_ABOVE_TETS = 60_000              # direct LU below, AMG-CG above
ACCEPT_GCI_PCT = 5.0                 # production mesh accepted when within this of the extrapolation


def levels(prod, n_coarse=2, n_fine=1):
    """(h, curvature) from coarse to fine; the production mesh is level n_coarse."""
    h0, c0 = prod
    return [(round(h0 * R ** k, 3), round(c0 / R ** k, 2)) for k in range(n_coarse, -n_fine - 1, -1)]


def _summarise(rows, prod_index, keys):
    out = {}
    for case in rows[0]["results"]:
        out[case] = {}
        for k in keys:
            hs = [r["quality"]["h_eff_mm"] for r in rows]
            fs = [r["results"][case][k] for r in rows]
            ext = richardson(hs, fs)
            fp = fs[prod_index]
            if ext["f_extrap"]:
                ext["production_error_pct"] = round(100 * abs(fp - ext["f_extrap"]) / abs(ext["f_extrap"]), 2)
            else:
                ext["production_error_pct"] = None
            ext["values"] = fs
            ext["production"] = fp
            out[case][k] = ext
    return out


def summarise(r):
    """(Re)compute the convergence summary and the verdict from the levels of a study."""
    hs = [l["h"] for l in r["levels"]]
    ph = r["production_mesh"]["h"]
    prod = min(range(len(hs)), key=lambda i: abs(hs[i] - ph))
    r["summary"] = _summarise(r["levels"], prod, r["keys"])
    r["verdict"] = _verdict(r["summary"], r["checked"])
    return r


def _verdict(summary, checked_keys):
    """Production mesh is adequate when every checked result converges and is within
    ACCEPT_GCI_PCT of its extrapolation (or changes less than that on the finest step)."""
    bad = []
    for case, res in summary.items():
        for k in checked_keys:
            e = res[k]
            err = e["production_error_pct"]
            ch = e["change_fine_pct"]
            ok = (err is not None and err <= ACCEPT_GCI_PCT) or (e["status"] in ("oscillating", "converged")
                                                                   and ch is not None and ch <= ACCEPT_GCI_PCT)
            if not ok:
                bad.append(f"{case}.{k}")
    return {"pass": not bad, "failed": bad, "accept_pct": ACCEPT_GCI_PCT}


def stem_study(p=None, lv=None, cases=sv.LOAD_CASES, log=print):
    p = p or stem_mod.StemParams()
    lv = lv or levels(STEM_PROD)
    body, _ = stem_mod.build(p)
    fixed, bar = sv.body_supports(p)
    mat = MATERIALS[p.material]
    rs = p.steerer_d / 2
    rows = []
    for h, curv in lv:
        t0 = time.time()
        mesh = mesh_shape(body, h, curv)
        solver = "amg" if mesh.t.shape[1] > AMG_ABOVE_TETS else "direct"
        res = solve_cases(mesh, mat, fixed, {lc.name: [(bar, lc.force_N)] for lc in cases}, solver, fields=True)
        out = {}
        for name, r in res.items():
            vm, c = r.fields["vm_el"], r.fields["centroids"]
            away = np.hypot(c[0], c[1]) > rs + AWAY_MM
            k = int(np.argmax(np.where(away, vm, 0)))
            out[name] = dict(max_disp_mm=r.max_disp_mm, vm_p999_MPa=r.vm_p999_MPa, vm_max_MPa=r.vm_max_MPa,
                             vm_peak_away_MPa=round(float(vm[k]), 1), at_away_mm=[round(float(x), 1) for x in c[:, k]])
        q = mesh_quality(mesh)
        rows.append(dict(h=h, curvature=curv, solver=solver, dofs=3 * (mesh.p.shape[1] + mesh.edges.shape[1]),
                         seconds=round(time.time() - t0, 1), quality=q, results=out))
        log(f"stem h={h} curv={curv}: {q['tets']} tets, {rows[-1]['seconds']} s")
    return summarise({"part": "stem body", "params": stem_mod.asdict(p),
                      "production_mesh": dict(h=STEM_PROD[0], curvature=STEM_PROD[1]), "levels": rows,
                      "keys": ["max_disp_mm", "vm_p999_MPa", "vm_peak_away_MPa", "vm_max_MPa"],
                      "checked": ["max_disp_mm", "vm_p999_MPa"],
                      "notes": [f"支持部 (ステアラ穴) から {AWAY_MM:g} mm 以内の要素は vm_peak_away から除外。",
                      "vm_max は剛体支持の端部特異点を含む (細分化で発散する)。判定には使わない。"]})


def handlebar_study(p=None, lv=None, cases=hv.LOAD_CASES, log=print):
    p = p or hb.HandlebarParams()
    lv = lv or levels(BAR_PROD)
    bar = hb.build(p)
    rows = []
    for h, curv in lv:
        t0 = time.time()
        # the solid FEA meshes the half itself; the mesh size decides the solver
        res, info = hv.solid_fea(p, bar, cases, h, curv, solver="direct", keep_mesh=True) if h >= BAR_PROD[0] / R - 1e-6 \
            else hv.solid_fea(p, bar, cases, h, curv, solver="amg", keep_mesh=True)
        mesh = info.pop("mesh")
        out = {name: dict(max_disp_mm=r["max_disp_mm"], vm_p999_MPa=r["vm_p999_MPa"], vm_max_MPa=r["vm_max_MPa"],
                          vm_p999_away_MPa=r["vm_p999_away_from_clamp_MPa"],
                          vm_peak_away_MPa=r["vm_max_away_from_clamp_MPa"], at_away_mm=r["at_away_mm"])
               for name, r in res.items()}
        q = mesh_quality(mesh)
        rows.append(dict(h=h, curvature=curv, solver="direct" if h >= BAR_PROD[0] / R - 1e-6 else "amg",
                         dofs=info["dofs"], seconds=round(time.time() - t0, 1), quality=q, results=out))
        log(f"handlebar h={h} curv={curv}: {q['tets']} tets, {rows[-1]['seconds']} s")
    return summarise({"part": "handlebar (right half)", "params": hb.asdict(p),
                      "production_mesh": dict(h=BAR_PROD[0], curvature=BAR_PROD[1]), "levels": rows,
                      "keys": ["max_disp_mm", "vm_p999_MPa", "vm_p999_away_MPa", "vm_peak_away_MPa", "vm_max_MPa"],
                      "checked": ["max_disp_mm", "vm_p999_away_MPa"],
                      "notes": [f"ステムクランプ端から {hv.STEM_CLAMP_W / 2 + 3:g} mm (y) 以内は away 値から除外 (剛体クランプ端の特異点)。",
                      "判定に使う値は solid_fea と同じ max(vm_p999, vm_p999_away)。"]})
