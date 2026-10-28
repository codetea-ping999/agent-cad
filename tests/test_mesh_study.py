import numpy as np
from agent_cad.fea.convergence import richardson, mesh_quality, tet_volumes
from agent_cad.verify import mesh_study as ms


def test_richardson_recovers_order_and_limit():
    h = [4.0, 2.8, 2.0, 1.4]
    f = [10.0 + 0.5 * x ** 2 for x in h]
    r = richardson(h, f)
    assert r["status"] == "converging"
    assert abs(r["p"] - 2.0) < 1e-6 and abs(r["f_extrap"] - 10.0) < 1e-6
    assert 0 < r["gci_fine_pct"] < 15


def test_richardson_flags_a_singular_peak():
    h = [4.0, 2.8, 2.0, 1.4]
    r = richardson(h, [100 * x ** -0.5 for x in h])        # grows without bound
    assert r["status"] == "diverging" and r["f_extrap"] is None
    assert richardson(h, [1.0, 1.2, 0.9, 1.1])["status"] == "oscillating"
    assert richardson(h, [1.0, 1.001, 1.002, 1.003])["status"] == "converged"     # < 1 % steps


def test_richardson_equal_changes_are_diverging():
    r = richardson([4, 2, 1], [100, 110, 120])
    assert r["status"] == "diverging" and r["f_extrap"] is None


def test_handlebar_checks_the_governing_stress():
    """handlebar_verify accepts on max(vm_p999, vm_p999_away): the study checks that value."""
    lv = [dict(h=h, results={"c": dict(max_disp_mm=1.0, vm_p999_MPa=a, vm_p999_away_MPa=b)},
               quality=dict(h_eff_mm=h)) for h, a, b in [(4, 100, 90), (2.8, 140, 91), (2, 180, 91.5), (1.4, 220, 91.6)]]
    r = ms.summarise({"levels": lv, "production_mesh": dict(h=2), "keys": ["max_disp_mm", "vm_p999_MPa", "vm_p999_away_MPa"],
                      "checked": ["max_disp_mm", "vm_p999_away_MPa"]})
    assert "vm_check_MPa" in r["checked"]
    assert not r["verdict"]["pass"]


def test_mesh_quality_of_regular_and_flat_tets():
    from skfem import MeshTet
    reg = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float).T
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.02]], float).T
    q = mesh_quality(MeshTet(np.hstack([reg, flat]), np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).T))
    assert abs(q["quality_median"] - (1 + q["quality_min"]) / 2) < 2e-3
    assert q["quality_min"] < 0.2 and q["share_q_below_0.2_pct"] == 50.0
    assert abs(abs(tet_volumes(MeshTet(reg, np.array([[0, 1, 2, 3]]).T))[0]) - 8 / 3) < 1e-9


def test_levels_bracket_the_production_mesh():
    lv = ms.levels(ms.STEM_PROD)
    assert len(lv) == 4 and lv[2] == ms.STEM_PROD
    assert lv[0][0] > lv[1][0] > lv[2][0] > lv[3][0]
    assert abs(lv[0][0] / lv[1][0] - np.sqrt(2)) < 1e-3


def test_stem_study_on_coarse_meshes():
    """Smoke test of the study on three cheap meshes: the displacement converges, the
    report has every case and level."""
    lv = [(6.0, 4.0), (4.243, 5.66), (3.0, 8.0)]
    r = ms.stem_study(lv=lv, log=lambda s: None)
    assert len(r["levels"]) == 3
    assert set(r["summary"]) == {lc.name for lc in ms.sv.LOAD_CASES}
    d = r["summary"]["forward_bending_static"]["max_disp_mm"]
    assert d["change_fine_pct"] < 5
    for l in r["levels"]:
        assert l["quality"]["quality_min"] > 0.01
