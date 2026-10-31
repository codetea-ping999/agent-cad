import dataclasses
import numpy as np
import pytest
from agent_cad.parts import stem
from agent_cad.materials import MATERIALS
from agent_cad.verify.stem_verify import verify, MIN_WALL_MM
from agent_cad.verify.wall import min_wall


def test_wall_of_a_tube_is_measured():
    from build123d import Cylinder
    tube = Cylinder(10, 30) - Cylinder(8.5, 30)
    w = min_wall(tube)
    assert abs(w["min_mm"] - 1.5) < 0.01


def test_a_very_thin_wall_is_measured():
    from build123d import Box
    w = min_wall(Box(10, 10, 0.04))
    assert w and abs(w["min_mm"] - 0.04) < 1e-3


def test_a_minimum_between_the_grid_points_is_found():
    from build123d import Cylinder, Pos
    # off-centre bore: the thinnest wall (1.45 mm) lies between the angular samples
    w = min_wall(Cylinder(10, 30) - Pos(0.55, 0, 0) * Cylinder(8, 30))
    assert abs(w["min_mm"] - 1.45) < 0.01


def test_a_minimum_at_the_face_boundary_is_found():
    from build123d import Cylinder, Cone
    # tapered bore: the thinnest wall (1.4 mm) is at the wide end, outside the grid's band
    w = min_wall(Cylinder(10, 30) - Cone(6, 8.6, 30))
    assert abs(w["min_mm"] - 1.4) < 0.01


def test_counterbore_web_to_the_bar_bore_is_found():
    # faceplate bolts at 20.2 mm (geometry rev 2): the counterbore leaves ~0.58 mm to the bar
    # bore, between the coarse samples; the refined search finds it
    r = verify(dataclasses.replace(stem.StemParams(), fp_bolt_off=20.2), fea=False)
    w = {x["check"]: x for x in r["rules"]}["wall.faceplate"]
    assert "wall.faceplate" in r["failed"] and w["value"] < 0.7


def test_cantilever_matches_beam_theory():
    """Solver check: tip-loaded cantilever 100 x 10 x 10 vs P L^3 / 3EI (+ shear)."""
    from build123d import Box, Align
    from agent_cad.fea.static import mesh_shape, solve
    L, a, P = 100.0, 10.0, 100.0
    m = MATERIALS["A6061-T6"]
    mesh = mesh_shape(Box(L, a, a, align=(Align.MIN, Align.CENTER, Align.CENTER)), 2.5)
    r = solve(mesh, m, lambda c: c[0] < 1e-6, [(lambda c: c[0] > L - 1e-6, (0, 0, -P))])
    I = a ** 4 / 12
    G = m.E_MPa / (2 * (1 + m.nu))
    expect = P * L ** 3 / (3 * m.E_MPa * I) + P * L / (5 / 6 * a * a * G)
    assert abs(r.max_disp_mm - expect) / expect < 0.05


def test_default_stem_passes_without_fea():
    r = verify(stem.StemParams(), fea=False)
    assert r["pass"], r["failed"]
    assert {x["check"] for x in r["rules"]} >= {"wall.body", "wall.faceplate", "thread.faceplate", "mass.total_g"}


def test_thin_wall_fails_the_loop():
    p = dataclasses.replace(stem.StemParams(), steerer_wall=1.0)
    r = verify(p, fea=False)
    assert "wall.body" in r["failed"]
    assert min(x["value"] for x in r["rules"] if x["check"] == "wall.body") < MIN_WALL_MM


def test_fea_on_a_coarse_mesh():
    r = verify(stem.StemParams(), fea=True, h=5.0)
    s = {x["check"]: x for x in r["rules"] if x["check"].startswith("strength.")}
    assert len(s) == 3 and all(x["provisional_load"] for x in s.values())
    # forward bending: stress well below yield/1.2 for the 6061 default (h=2 gives ~103 MPa)
    assert 60 < s["strength.forward_bending_static"]["value"] < 160
    assert r["pass"], r["failed"]


def test_cli_exit_status(tmp_path):
    from scripts.verify_stem import main
    out = tmp_path / "new_dir" / "v.json"
    assert main(["--no-fea", "--json", str(out)]) == 0 and out.exists()
    assert main(["--no-fea", "--set", "steerer_wall=1.0"]) == 1


def test_overrides_follow_the_declared_types():
    from scripts.verify_stem import parse_set
    from scripts.make_stem import CASES
    p = parse_set(CASES["stem110_-17_7075"], ["angle_deg=-16.5", "length=110.5"])
    assert p.angle_deg == -16.5 and p.length == 110.5


def test_tap_drill_outside_the_M5_band_fails_the_loop():
    r = verify(dataclasses.replace(stem.StemParams(), tap_drill_d=3.3), fea=False)
    assert "thread.tap_drill_M5" in r["failed"]


def test_duplicate_load_case_names_are_rejected():
    from agent_cad.verify.stem_verify import LOAD_CASES
    with pytest.raises(ValueError):
        verify(stem.StemParams(), cases=(LOAD_CASES[0], LOAD_CASES[0]))
