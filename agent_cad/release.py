"""Drawing release (提出図) bookkeeping: fixed drawing numbers, revisions, shape
fingerprints, approvals and the sign-off stamp on the sheet.

- The register (release/registry.json) gives every drawing a fixed number (AC-1xxx
  products, AC-5xxx tooling). A parameter hash no longer names a drawing: it would give a
  changed part a new number instead of a new revision.
- The fingerprint covers what the drawing says and the shape it describes: dimension
  texts and tolerances, notes, title block, required features, local roughness, sheet
  size, and the B-rep's volume, area and extent. It leaves out the sign-off and the
  date, so it changes only when the content does.
- The history (release/history.json) lists each number's revisions, oldest first:
  rev, date, fingerprint, change, approved_by, approved_on. The first issue is "-",
  then A, B, ... (I and O skipped). An unapproved revision follows the model (its
  fingerprint is refreshed); an approved one is frozen: a different fingerprint then
  needs a new revision (`python -m scripts.release revise ...`).
- Only a person approves. `approve` records who and when against the fingerprint of
  the drawing that was checked; the sheet shows 出図 only while that fingerprint holds.
"""
import hashlib, json, os, datetime

REV_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
FIRST_REV = "-"
STATUS_RELEASED = "出図"
STATUS_CHECKED = "検図済 (未承認)"
STATUS_FAILED = "検図不合格"
CHECKER = "自動検図"
AUTHOR = "agent-cad"


def next_rev(rev):
    if rev == FIRST_REV:
        return REV_LETTERS[0]
    i = REV_LETTERS.index(rev)
    if i + 1 >= len(REV_LETTERS):
        raise ValueError("revision letters exhausted")
    return REV_LETTERS[i + 1]


def _r(x, nd):
    return round(float(x), nd) + 0.0     # no -0.0


def shape_props(shape):
    bb = shape.bounding_box()
    return dict(volume=_r(shape.volume, 0), area=_r(shape.area, 0),
                bbox=[_r(v, 2) for v in (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)])


def fingerprint(sheet, shape):
    m = sheet.meta
    tb = {k: v for k, v in m.get("title_block", {}).items() if k not in ("図番", "改訂")}
    content = dict(
        size=list(sheet.size), title=tb,
        dims=[(d["name"], d["text"], d.get("tol")) for d in m.get("dims", [])],
        features=[(f["name"], f["expected"]) for f in m.get("features", [])],
        notes=m.get("notes", []), gdt=m.get("gdt", []),
        roughness=[(r["name"], r["required"]) for r in m.get("roughness_local", [])],
        roughness_general=m.get("roughness_general"), finish=m.get("finish"),
        tables=[m.get("hole_table"), m.get("bom")],
        shape=shape_props(shape))
    s = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class History:
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    def entries(self, number):
        return self.data.get(number, [])

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1, sort_keys=True)
            f.write("\n")

    def state(self, number, fp, today):
        """Revision and release state of `number` for a drawing with fingerprint fp.
        Returns dict(rev, approved, entry, pending): `pending` is the history entry this
        build would write (a new first issue, or a refreshed draft), None if unchanged.
        A changed drawing whose last revision is approved has no valid revision: rev is
        that revision and `error` says what to do."""
        es = self.entries(number)
        if not es:
            e = dict(rev=FIRST_REV, date=today, fingerprint=fp, change="初版", approved_by=None, approved_on=None)
            return dict(rev=FIRST_REV, approved=False, entry=e, pending=e, error=None)
        last = es[-1]
        if last["fingerprint"] == fp:
            return dict(rev=last["rev"], approved=bool(last.get("approved_by")), entry=last, pending=None, error=None)
        if last.get("approved_by"):
            return dict(rev=last["rev"], approved=False, entry=last, pending=None,
                        error=f"出図済みの改訂 {last['rev']} から内容が変わった: "
                              f"`python -m scripts.release revise {number} \"<改訂内容>\"` で改訂を起こすこと")
        e = dict(last, fingerprint=fp, date=today)
        return dict(rev=last["rev"], approved=False, entry=e, pending=e, error=None)

    def apply(self, number, entry):
        es = self.data.setdefault(number, [])
        if es and es[-1]["rev"] == entry["rev"]:
            es[-1] = entry
        else:
            es.append(entry)

    def revise(self, number, change, fp, today):
        es = self.entries(number)
        if not es:
            raise ValueError(f"{number}: no first issue yet")
        if not es[-1].get("approved_by"):
            raise ValueError(f"{number}: revision {es[-1]['rev']} is not approved yet; change it in place")
        self.data[number].append(dict(rev=next_rev(es[-1]["rev"]), date=today, fingerprint=fp, change=change,
                                      approved_by=None, approved_on=None))

    def approve(self, number, fp, by, today):
        es = self.entries(number)
        if not es:
            raise ValueError(f"{number}: not in the history (run the release build with --write-history)")
        last = es[-1]
        if last["fingerprint"] != fp:
            raise ValueError(f"{number}: the checked drawing differs from revision {last['rev']} in the history")
        last["approved_by"], last["approved_on"] = by, today


def stamp(sheet, number, st, history_entries, checked_ok, today):
    """Sign-off row and revision table on a checked sheet."""
    sheet.meta["title_block"]["図番"] = number
    status = STATUS_FAILED if not checked_ok or st["error"] else STATUS_RELEASED if st["approved"] else STATUS_CHECKED
    hist = [e for e in history_entries if e["rev"] != st["entry"]["rev"]] + [st["entry"]]
    rows = [(e["rev"], e["date"], e["change"], e.get("approved_by") or "") for e in reversed(hist)]
    e = st["entry"]
    info = dict(rev=st["rev"], status=status, drawn=(AUTHOR, e["date"]),
                checked=(CHECKER, today) if checked_ok else None,
                approved=(e["approved_by"], e["approved_on"]) if st["approved"] else None,
                history=rows)
    sheet.stamp_release(info)
    return status


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def today():
    return datetime.date.today().isoformat()
