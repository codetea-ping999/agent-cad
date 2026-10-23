"""Release flow: revisions, fingerprints, approvals, sign-off stamp, semantic PMI and
the tooling drawings."""
import pytest
from build123d import Box, Cylinder, Pos
from agent_cad import release as rl, pmi
from agent_cad.drawing.sheet import Sheet, A3
from agent_cad.drawing.check import check_sheet
from agent_cad.drawing import tool
from agent_cad.step_io import write_step, PartSpec
from agent_cad.verify.step_check import check_step

FP1, FP2, FP3 = "a" * 16, "b" * 16, "c" * 16


def test_revision_letters_skip_i_and_o():
    assert rl.next_rev("-") == "A"
    assert rl.next_rev("H") == "J" and rl.next_rev("N") == "P"


def test_history_first_issue_draft_approval_and_revision(tmp_path):
    h = rl.History(str(tmp_path / "h.json"))
    st = h.state("AC-1", FP1, "2026-10-23")
    assert st["rev"] == "-" and not st["approved"] and st["pending"]["change"] == "初版"
    h.apply("AC-1", st["pending"])
    # an unapproved draft follows the model
    st = h.state("AC-1", FP2, "2026-10-24")
    assert st["rev"] == "-" and st["error"] is None and st["pending"]["fingerprint"] == FP2
    h.apply("AC-1", st["pending"])
    with pytest.raises(ValueError):
        h.approve("AC-1", FP1, "Reviewer", "2026-10-25")       # not the drawing that was checked
    h.approve("AC-1", FP2, "Reviewer", "2026-10-25")
    assert h.state("AC-1", FP2, "2026-10-26")["approved"]
    # an approved revision is frozen: a change needs a new revision
    st = h.state("AC-1", FP3, "2026-10-26")
    assert st["error"] and not st["approved"]
    h.revise("AC-1", "穴径変更", FP3, "2026-10-26")
    st = h.state("AC-1", FP3, "2026-10-26")
    assert st["rev"] == "A" and st["error"] is None and not st["approved"]
    with pytest.raises(ValueError):
        h.revise("AC-1", "again", FP3, "2026-10-26")            # A is not approved yet
    h.save()
    assert rl.History(str(tmp_path / "h.json")).entries("AC-1")[-1]["rev"] == "A"


def _sheet(text="10"):
    sh = Sheet(A3)
    sh.frame_and_title([("図名", "T"), ("図番", "AC-1"), ("材料", "S45C"), ("質量 (計算)", "1 g"), ("尺度", "1:1"),
                        ("投影法", "第三角法"), ("単位", "mm"), ("普通公差", "JIS B 0405-m"), ("作成", "x 2026-01-01")])
    sh.meta["dims"] = [dict(name="len", measured=10.0, intent=10.0, text=text, tol=None)]
    return sh


def test_fingerprint_follows_content_not_signoff():
    box = Box(10, 10, 10)
    a, b = _sheet(), _sheet()
    assert rl.fingerprint(a, box) == rl.fingerprint(b, box)
    st = dict(rev="-", approved=False, entry=dict(rev="-", date="2026-10-23", change="初版", fingerprint=FP1), error=None)
    rl.stamp(b, "AC-1", st, [], True, "2026-10-23")
    assert rl.fingerprint(a, box) == rl.fingerprint(b, box)
    assert rl.fingerprint(_sheet("10.5"), box) != rl.fingerprint(a, box)
    assert rl.fingerprint(a, Box(10, 10, 11)) != rl.fingerprint(a, box)


def test_title_block_signoff_and_revision_table():
    sh = _sheet()
    assert "作成" not in sh.meta["title_block"]               # the sign-off row carries it
    st = dict(rev="A", approved=True, entry=dict(rev="A", date="2026-10-24", change="穴径変更", fingerprint=FP2,
                                                 approved_by="Reviewer", approved_on="2026-10-25"), error=None)
    hist = [dict(rev="-", date="2026-10-23", change="初版", fingerprint=FP1, approved_by="Reviewer", approved_on="2026-10-23")]
    status = rl.stamp(sh, "AC-1", st, hist, True, "2026-10-25")
    assert status == rl.STATUS_RELEASED
    texts = {it[3] for it in sh.items if it[0] == "text"}
    assert {"A", "穴径変更", "初版", "Reviewer", rl.CHECKER, rl.STATUS_RELEASED} <= texts
    r = check_sheet(sh)
    assert all(c["pass"] for c in r["checks"] if c["check"].startswith("layout")), r
    # a failed gate never shows as released, and nobody signs the check
    sh2 = _sheet()
    assert rl.stamp(sh2, "AC-1", st, hist, False, "2026-10-25") == rl.STATUS_FAILED
    assert sh2.meta["release"]["checked"] is None


def _holed_block():
    return Box(60, 30, 40) - Pos(0, 0, 0) * Cylinder(5, 50) - Pos(15, 0, 0) * Cylinder(5, 50)


def test_pmi_written_linked_and_read_back(tmp_path):
    shape = _holed_block()
    reqs = [dict(name="hole", kind="diameter", value=10.0, upper=0.015, lower=0.0, text="φ10 H7"),
            dict(name="length", kind="linear", value=60.0, upper=0.3, lower=-0.3, text="60"),
            dict(name="pitch", kind="linear", value=15.0, upper=0.2, lower=-0.2, text="15"),
            dict(name="draft_root", kind="linear", value=12.345, upper=0.2, lower=-0.2, text="12.35")]
    ps = PartSpec("AC-9_BLOCK", shape, (0.5, 0.5, 0.5), "S45C", 7.85, pmi=reqs)
    path = str(tmp_path / "b.step")
    write_step([ps], path)
    assert [r["name"] for r in ps.pmi_skipped] == ["draft_root"]   # no faces that far apart
    r = pmi.check(path, reqs, ps.pmi_skipped)
    assert r["pass"] and r["written"] == 3, r
    assert check_step(path, [ps], None)["pass"]
    # a file whose PMI does not match the drawing fails
    bad = [dict(reqs[0], value=10.5)] + reqs[1:]
    assert not pmi.check(path, bad, ps.pmi_skipped)["pass"]


def test_hole_groups_and_component_sheet():
    shape = Box(120, 40, 80) - Pos(0, 10, 0) * Cylinder(4, 200, rotation=(0, 90, 0)) \
        - Pos(0, -10, 0) * Cylinder(4, 200, rotation=(0, 90, 0)) - Pos(30, 0, 20) * Cylinder(3, 60, rotation=(90, 0, 0))
    shape = Pos(0, 20, 0) * shape                                   # sits at Y >= 0
    assert tool.hole_groups(shape) == {("X", 8.0): 2, ("Y", 6.0): 1}
    info = dict(number="AC-9", title="テスト型", material="SKD61", finish="44〜48 HRC", up=-1, block=(120, 40, 80),
                holes=[("冷却穴", 8.0, "X", 2), ("突出しピン穴", 6.0, "Y", 1)], notes=[], density_g_cm3=7.8)
    sh = tool.component_sheet(shape, info)
    r = check_sheet(sh)
    assert r["pass"], [c for c in r["checks"] if not c["pass"]]
    info["holes"][0] = ("冷却穴", 8.0, "X", 3)                     # the tool lacks a channel the design asks for
    r = check_sheet(tool.component_sheet(shape, info))
    assert not r["pass"] and any(c["check"] == "feature.holes.冷却穴" and not c["pass"] for c in r["checks"])
