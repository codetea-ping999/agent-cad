import math
import pytest
from build123d import Box, Pos, Align, Plane, mirror
from agent_cad.parts import stem
from agent_cad.process import forging as fg
from agent_cad.verify.forging_check import verify_forging, draft_and_undercut, allowance

P = stem.StemParams()


@pytest.fixture(scope="module")
def made():
    parts = stem.build(P)
    ds = fg.die_set(P)
    return parts, ds, verify_forging(P, parts=parts, ds=ds)


def test_stem_forging_and_dies_pass(made):
    _, _, rep = made
    assert rep["pass"], rep["failed"]
    assert {r["check"] for r in rep["rules"]} >= {"forging.allowance", "forging.draft", "forging.undercut",
                                                  "die.halves_clear", "die.upper.clears_forging"}


def test_forging_is_heavier_than_part_and_load_is_sane(made):
    _, _, rep = made
    pr = rep["process"]
    assert pr["mass_finished_g"] < pr["mass_forging_trimmed_g"] < pr["mass_forging_with_flash_g"]
    assert 10 < pr["material_yield_pct"] < 80
    assert 50 < pr["press_load_t"] < 2000
    assert 1.005 < pr["shrink_factor"] < 1.02           # aluminium at ~450 C, die at 200 C


def test_bar_bore_web_is_left_at_the_parting_plane(made):
    (b, f), ds, _ = made
    d, n, S, B = stem.frame(P)
    from build123d import Vector
    blank = ds["blank"]
    assert blank.is_inside(Vector(B.X, 0, B.Z))                       # web
    assert not blank.is_inside(Vector(B.X, P.bar_clamp_w / 2 - 1, B.Z))  # punched recess


def _drafted_box(deg):
    """A 40 x 20 x 30 block standing on the parting plane, drafted by `deg`, mirrored below."""
    from shapely.geometry import box
    up = fg.drafted_prism(box(-20, -15, 20, 15), 10, deg) if deg else \
        Pos(0, 5, 0) * Box(40, 10, 30)
    return up + mirror(up, Plane.XZ)


def test_draft_check_measures_the_wall_angle():
    (dmin, _), under = draft_and_undercut(_drafted_box(3.0))
    assert abs(dmin - 3.0) < 0.05 and not under
    (dmin, _), _ = draft_and_undercut(_drafted_box(0))
    assert abs(dmin) < 0.05                                            # straight walls: no draft


def test_undercut_is_found():
    # a mushroom: a wide cap above a narrow stem, both halves -> the cap overhangs
    up = Pos(0, 3, 0) * Box(10, 6, 10) + Pos(0, 8, 0) * Box(30, 4, 30)
    _, under = draft_and_undercut(up + mirror(up, Plane.XZ))
    assert under


def test_envelope_contains_shape_with_allowance():
    shape = Pos(0, 0, 0) * Box(30, 16, 20) + Pos(10, 0, 0) * Box(10, 30, 10)
    env = fg.drafted_envelope(shape, 1.0, 5.0, step=2.0, breaks=(5, 8), symmetric=True)
    assert (shape - env).volume < 1e-6
    a, _ = allowance([shape], env)
    assert a >= 1.0 - 0.05
    (dmin, _), under = draft_and_undercut(env)
    assert dmin >= 5.0 - 0.05 and not under


def test_stronger_alloy_needs_a_bigger_press():
    hard = fg.FLOW_STRESS_MPA["A7075-T6"] > fg.FLOW_STRESS_MPA["A6061-T6"]
    assert hard and fg.shrink_factor(stem.StemParams(material="Ti-6Al-4V")) < fg.shrink_factor(P)


def test_flash_land_keeps_its_thickness(made):
    """Between the cavity and the gutter the die face is recessed by exactly t/2."""
    from build123d import Vector
    from shapely.geometry import Point
    _, ds, rep = made
    pr = rep["process"]
    k = pr["shrink_factor"]
    outline, _, t, b, tg = fg.flash_dims(ds["blank"], fg.ForgingParams())
    ring = outline.buffer(b * 0.5).exterior                        # mid-land, outside the forging
    x, z = ring.interpolate(0.3, normalized=True).coords[0]
    x, z = x * k, z * k
    up = ds["upper"]
    assert not up.is_inside(Vector(x, t / 2 * k * 0.5, z))         # land recess
    assert up.is_inside(Vector(x, t / 2 * k + 0.2, z))             # die face above the land
    gx, gz = outline.buffer(b + 6).exterior.interpolate(0.3, normalized=True).coords[0]
    assert not up.is_inside(Vector(gx * k, tg / 2 * 0.9, gz * k))  # gutter
