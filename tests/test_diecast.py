import math
import pytest
from build123d import Vector
from agent_cad.parts import bracket
from agent_cad.process import diecast as dc
from agent_cad.verify.diecast_check import verify_diecast, machined_faces
from agent_cad.verify.forging_check import draft_and_undercut

P = bracket.BracketParams()
DP = dc.DiecastParams()


@pytest.fixture(scope="module")
def made():
    ds = dc.die_set(P, DP)
    return ds, verify_diecast(P, DP, ds=ds)


def test_bracket_casting_and_die_pass(made):
    _, rep = made
    assert rep["pass"], rep["failed"]
    assert {r["check"] for r in rep["rules"]} >= {
        "casting.machining_stock", "casting.draft", "casting.undercut", "shot.draft", "casting.min_wall",
        "die.halves_clear", "ejector.pins_seated", "cooling.to_cavity", "process.machine"}


def test_mass_parameters_and_process_numbers_are_sane(made):
    ds, rep = made
    pr = rep["process"]
    assert pr["mass_finished_g"] < pr["mass_casting_g"] < pr["mass_shot_g"]
    assert pr["mass_finished_g"] <= P.mass_target_g
    assert 40 < pr["material_yield_pct"] < 90
    assert 0.02 < pr["fill_time_s"] < 0.15                 # thin-wall aluminium HPDC
    assert 25 <= pr["gate_velocity_m_s"] <= 60
    assert pr["runner_area_mm2"] > pr["gate_area_mm2"]
    assert pr["machine_t"] == 125                          # ~11 000 mm2 at 60 MPa x 1.2


def test_density_follows_the_material():
    assert abs(bracket.density(P) - 2.70) < 1e-9


def test_cored_holes_are_open_at_the_parting_plane_and_widen_upward(made):
    ds, _ = made
    cast = ds["casting"]
    x, z = bracket.bosses(P)[0]
    assert not cast.is_inside(Vector(x, 0.01, z))           # through hole down to the back face
    assert not cast.is_inside(Vector(x + P.hole_d / 2 - 0.05, 0.5, z))
    # the hole is narrowest at the parting plane: the core pin can be pulled toward +Y
    (dmin, _), under = draft_and_undercut(cast, per_face=2, draft_deg=P.core_draft_deg)
    assert not under


def test_machining_finds_faced_tops_and_bore_only(made):
    ds, _ = made
    mf = machined_faces(ds["finished"], ds["casting"])
    kinds = sorted(f.geom_type.name for f in mf)
    # four boss tops and the hub top (planes), the reamed bore (cylinder)
    assert kinds.count("PLANE") == 5 and kinds.count("CYLINDER") >= 1


def test_ejector_pins_push_on_flat_faces_and_cooling_keeps_clear(made):
    ds, rep = made
    r = {x["check"]: x for x in rep["rules"]}
    assert len(ds["pins"]) >= 8
    assert r["ejector.pins_seated"]["value"] > 0.999
    assert r["cooling.to_cavity"]["value"] >= 1.5 * DP.cool_d_mm
    assert r["cooling.to_sleeve"]["value"] >= 1.5 * DP.cool_d_mm


def test_pin_on_an_edge_is_rejected(made):
    ds, _ = made
    # a pin straddling a rib foot is not offered as a site
    shot = ds["shot"]
    region = dc.top_faces(shot, P.plate_t)
    from shapely.geometry import Point
    x, z = bracket.bosses(P)[0]
    assert not region.contains(Point(x / 2, z / 2).buffer(DP.pin_d_mm / 2))   # on the rib


def test_shrink_and_clamp_follow_the_parameters():
    assert abs(dc.shrink_factor(DP) - 1.0055) < 1e-12
    hot = dc.DiecastParams(T_die_C=280.0)
    assert dc.fill_time(hot, 4.0) > dc.fill_time(DP, 4.0)      # hotter die -> more time to fill
    g1 = dc.gating_sizes(P, DP, 60_000)
    g2 = dc.gating_sizes(P, dc.DiecastParams(gate_v_m_s=50.0), 60_000)
    assert g2["gate_area_mm2"] < g1["gate_area_mm2"]


# default bracket's (casting volume, finished volume) per geometry revision: a change to
# bracket.build() that moves these must bump bracket.GEOMETRY_REV (it is in the document number)
GEOMETRY_FINGERPRINT = {1: (57460.0, 55272.6)}


def test_document_number_follows_the_geometry_revision(made, monkeypatch):
    from scripts.make_diecast import doc_no
    ds, _ = made
    assert GEOMETRY_FINGERPRINT[bracket.GEOMETRY_REV] == (round(ds["casting"].volume, 1), round(ds["finished"].volume, 1))
    before = doc_no(P)
    monkeypatch.setattr(bracket, "GEOMETRY_REV", bracket.GEOMETRY_REV + 1)
    assert doc_no(P) != before
