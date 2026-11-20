"""JIS drawings of tooling: outline drawings of die / mould components and assembly
drawings with a parts list.

Cavities and cores are cut from the 3D data (CAM), as is usual for dies and moulds, so a
component drawing carries what the toolroom needs besides: the block size, the
parting-face view and a section through the cavity, the drilled holes as a hole table
(diameter, direction, count), material, hardness, tolerances, and the product drawing
and revision the cavity was made from. Every value is measured from the B-rep and
recorded next to its design intent from the process model, so check_sheet() catches a
drawing that does not match the tool.

Views (third angle): the front view with the parting face up, the parting-face view
above it and section A-A through the middle on the right.
"""
import math
import numpy as np
from .sheet import Sheet, View, A3, A2, text_width
from .measure import cylinders
from ..standards import jis

TRIES = [(A3, 1.0), (A2, 1.0), (A3, 0.5), (A2, 0.5), (A3, 0.2), (A2, 0.2), (A3, 0.1), (A2, 0.1)]
TOL_CLASS = "f"
AXES = "XYZ"


def _fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _scale_text(k):
    return f"{k:g}:1" if k >= 1 else f"1:{1 / k:g}"


def _views(shape, up, k, cut_x):
    """front / top / right(section) with the parting face toward the top of the sheet.
    up = +1: the tool sits at Y <= 0 (parting face +Y); up = -1: at Y >= 0."""
    if up > 0:
        front = View("front", shape, (0, 0, -1), (1, 0, 0), (0, 0), k)
        top = View("top", shape, (0, -1, 0), (1, 0, 0), (0, 0), k)
        right = View("A-A", shape, (-1, 0, 0), (0, 0, -1), (0, 0), k, section=((cut_x, 0, 0), (-1, 0, 0)), label="A-A")
    else:
        front = View("front", shape, (0, 0, 1), (1, 0, 0), (0, 0), k)
        top = View("top", shape, (0, 1, 0), (1, 0, 0), (0, 0), k)
        right = View("A-A", shape, (-1, 0, 0), (0, 0, 1), (0, 0), k, section=((cut_x, 0, 0), (-1, 0, 0)), label="A-A")
    return front.build(), top.build(), right.build()


def hole_groups(shape):
    """Drilled holes: inner cylinders whose axis runs along X, Y or Z, grouped by
    (axis letter, diameter) -> number of distinct hole axes."""
    lines = {}
    for c in cylinders(shape):
        if not c.inner:
            continue
        ax = int(np.argmax(np.abs(c.axis)))
        if abs(abs(c.axis[ax]) - 1) > 1e-6:
            continue
        off = tuple(round(float(c.loc[i]), 3) for i in range(3) if i != ax)
        lines.setdefault((AXES[ax], round(2 * c.r, 3)), set()).add(off)
    return {k: len(v) for k, v in sorted(lines.items())}


def _general_tol_table(sh, pos, cls=TOL_CLASS):
    x, y = pos
    rows = jis.b0405_table(cls)
    sh.text((x, y), f"普通許容差 JIS B 0405-{cls}", 3.2, 0, "left", "bottom")
    cw, rh = 17.0, 4.6
    y -= 1.5
    cols = [(f"{_fmt(a)}以上" if i == 0 else f"{_fmt(a)}を超え", f"{_fmt(b)}以下", f"±{_fmt(t)}")
            for i, (a, b, t) in enumerate(rows)]
    n = len(cols) + 1
    for j in (0, 2, 3):
        sh.line([(x, y - j * rh), (x + cw * n, y - j * rh)], "thin")
    sh.line([(x + cw, y - rh), (x + cw * n, y - rh)], "thin")
    sh.text((x + cw / 2, y - rh), "寸法区分", 2.5, 0, "center", "center")
    sh.text((x + cw / 2, y - 2.5 * rh), "許容差", 2.5, 0, "center", "center")
    for i, col in enumerate(cols, start=1):
        for j, t in enumerate(col):
            sh.text((x + i * cw + cw / 2, y - (j + 0.5) * rh), t, 2.5, 0, "center", "center")
    for i in range(n + 1):
        sh.line([(x + i * cw, y), (x + i * cw, y - 3 * rh)], "thin")


def _notes(sh, lines, pos):
    x, y = pos
    sh.text((x, y), "注記", 4.0, 0, "left", "top")
    for i, s in enumerate(lines):
        sh.text((x + 2, y - 7 - i * 5.2), f"{i + 1}. {s}", 3.0, 0, "left", "top")
    sh.meta["notes"] = lines
    return y - 7 - len(lines) * 5.2


def table(sh, pos, cols, rows, rh=5.0, size=2.5):
    """Table with a header row: cols = [(header, width)], rows = [[cell, ...]]. pos = top left."""
    x, y = pos
    w = sum(c[1] for c in cols)
    n = len(rows) + 1
    for j in range(n + 1):
        sh.line([(x, y - j * rh), (x + w, y - j * rh)], "thin")
    xx = x
    for h, cw in cols:
        sh.line([(xx, y), (xx, y - n * rh)], "thin")
        sh.text((xx + cw / 2, y - rh / 2), h, size, 0, "center", "center")
        for j, r in enumerate(rows, start=1):
            v = str(r[cols.index((h, cw))])
            sh.text((xx + 1.2, y - (j + 0.5) * rh), v, size, 0, "left", "center")
        xx += cw
    sh.line([(xx, y), (xx, y - n * rh)], "thin")
    return y - n * rh


def _title(sh, info, mass_g, k):
    rows = [("図名", info["title"]), ("図番", info["number"]), ("材料", info["material"]),
            ("質量 (計算)", f"{mass_g / 1000:.2f} kg" if mass_g >= 1000 else f"{mass_g:.1f} g"),
            ("尺度", _scale_text(k)), ("投影法", "第三角法"), ("単位", "mm"),
            ("普通公差", f"JIS B 0405-{TOL_CLASS}")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 140, ytop + 4))
    return x0, ytop


def _record(sh, name, measured, intent, text):
    sh.meta.setdefault("dims", []).append(dict(name=name, measured=round(float(measured), 4),
                                               intent=float(intent), text=text, tol=None))


def _need(sh, name, found, expected):
    sh.meta.setdefault("features", []).append(dict(name=name, found=int(found), expected=int(expected)))


def _block_dims(sh, shape, front, right, block, up):
    """Overall length (X) and height (Y) on the front view, width (Z) on the section."""
    bb = shape.bounding_box()
    L, Hh, W = bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z
    zf = bb.max.Z if up > 0 else bb.min.Z           # the face toward the viewer
    yb = bb.min.Y if up > 0 else bb.max.Y           # the back of the block: bottom of the view
    fs, rs = front.to_sheet, right.to_sheet
    sh.dim(fs((bb.min.X, yb, zf)), fs((bb.max.X, yb, zf)), -14, _fmt(L))
    _record(sh, "length", L, block[0], _fmt(L))
    ya, yz = (bb.min.Y, bb.max.Y) if up > 0 else (bb.max.Y, bb.min.Y)
    sh.dim(fs((bb.min.X, yz, zf)), fs((bb.min.X, ya, zf)), 14, _fmt(Hh), u=(0, 1))
    _record(sh, "height", Hh, block[1], _fmt(Hh))
    xc = (bb.min.X + bb.max.X) / 2
    za, zb = (bb.min.Z, bb.max.Z) if up > 0 else (bb.max.Z, bb.min.Z)
    a, b = sorted((rs((xc, yb, za)), rs((xc, yb, zb))), key=lambda q: q[0])   # left to right: the line goes below
    sh.dim(a, b, -14, _fmt(W))
    _record(sh, "width", W, block[2], _fmt(W))
    return L, Hh, W


def component_sheet(shape, info):
    """Outline drawing of one tooling component. info: number, title, material, finish
    (heat treatment / hardness), up (+1 parting face +Y, -1 parting face -Y), block
    (X, Y, Z design size), holes [(use, diameter, axis letter, expected count)],
    notes [str], density_g_cm3. A3 1:1 if it fits, else A2, then smaller scales."""
    sh = None
    for size, k in TRIES:
        cand = _component(shape, info, size, k)
        if cand is None:
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError(f"{info['number']}: views do not fit A2 at 1:10")
    return sh


def _fits(sh):
    from .check import check_sheet
    bad = {c["check"] for c in check_sheet(sh)["checks"] if not c["pass"]}
    return not bad & {"layout.inside_frame", "layout.text_overlap", "layout.text_on_geometry"}


def _component(shape, info, size, k):
    sh = Sheet(size)
    bb = shape.bounding_box()
    cx = (bb.min.X + bb.max.X) / 2
    front, top, right = _views(shape, info["up"], k, cx)
    if not sh.place_third_angle(front, top, right=right, gap=34, edge=16):
        return None
    mass = shape.volume / 1000 * info["density_g_cm3"]
    x0, ytop = _title(sh, info, mass, k)
    sh.meta["part"] = info["number"]
    sh.meta["finish"] = info["finish"]
    sh.meta["roughness_general"] = "Ra 6.3"
    sh.meta["drawing_kind"] = "tool_component"
    sh.roughness((x0 + 6, ytop + 4), 6.3)
    sh.text((x0 + 40, ytop + 5.5),
            f"( {info.get('surface', 'キャビティ・コア面')} Ra {info.get('cavity_ra', 0.4)} )", 3.2, 0, "left", "bottom")
    _block_dims(sh, shape, front, right, info["block"], info["up"])
    # section line through the middle on the parting-face view
    tb = top.bbox()
    xs = top.to_sheet((cx, 0, 0))[0]
    sh.section_line((xs, tb[0][1] - 6), (xs, tb[1][1] + 6), "A", look=(1, 0))
    rb = right.bbox()
    sh.text(((rb[0][0] + rb[1][0]) / 2, rb[1][1] + 10), "A-A", 5, 0, "center", "bottom")
    # hole table: measured groups, each matched to its design use
    found = hole_groups(shape)
    rows, used = [], set()
    for i, (use, d, ax, n) in enumerate(info.get("holes", [])):
        keys = [kk for kk in found if kk[0] == ax and abs(kk[1] - d) < 0.01]
        got = sum(found[kk] for kk in keys)
        used.update(keys)
        _need(sh, f"holes.{use}", got, n)
        rows.append([chr(ord("A") + i) if i < 26 else str(i), f"φ{_fmt(d)}", ax, str(got), use])
    for key, n in found.items():
        if key not in used:
            rows.append([chr(ord("A") + len(rows)), f"φ{_fmt(key[1])}", key[0], str(n), "その他 (3Dデータ参照)"])
    sh.meta["hole_table"] = rows
    notes = [f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS} による。",
             f"材料 {info['material']}、{info['finish']}。",
             info.get("shape_note", "キャビティ・コア形状および穴位置は3Dデータ (STEP AP242、同図番) による。"),
             info.get("edge_note", "指示なきかどは C0.5、キャビティ稜線はシャープエッジ (面取りなし)。")] + info.get("notes", [])
    ybot = _notes(sh, notes, (16, 112))
    if rows:
        sh.text((16, ybot - 4), "穴表", 3.2, 0, "left", "bottom")
        cols = [("記号", 10), ("穴径", 20), ("方向", 12), ("数量", 12), ("用途", 40)]
        table(sh, (16, ybot - 5), cols, rows)
    _general_tol_table(sh, (116, ybot - 4))
    return sh


def assembly_sheet(shape, info):
    """Assembly drawing of a tool with its parts list. info: number, title, up, block
    (overall X, Y, Z intent), bom [(name, drawing no / standard, material, qty, remark)],
    parts_count (instances in the assembly, compared with the parts list), notes,
    density-weighted mass_g."""
    sh = None
    for size, k in TRIES:
        cand = _assembly(shape, info, size, k)
        if cand is None:
            continue
        sh = cand
        if _fits(sh):
            return sh
    if sh is None:
        raise ValueError(f"{info['number']}: views do not fit A2 at 1:10")
    return sh


def _assembly(shape, info, size, k):
    sh = Sheet(size)
    bb = shape.bounding_box()
    cx = (bb.min.X + bb.max.X) / 2
    front, top, right = _views(shape, info["up"], k, cx)
    if not sh.place_third_angle(front, top, right=right, gap=34, edge=16):
        return None
    rows = [("図名", info["title"]), ("図番", info["number"]), ("材料", "部品欄による"),
            ("質量 (計算)", f"{info['mass_g'] / 1000:.1f} kg"), ("尺度", _scale_text(k)), ("投影法", "第三角法"),
            ("単位", "mm"), ("普通公差", f"JIS B 0405-{TOL_CLASS}")]
    x0, ytop = sh.frame_and_title(rows)
    sh.projection_symbol((x0 + 140, ytop + 4))
    sh.meta["part"] = info["number"]
    sh.meta["finish"] = "部品図による"
    sh.meta["roughness_general"] = "部品図による"
    sh.meta["drawing_kind"] = "tool_assembly"
    _block_dims(sh, info.get("dim_shape", shape), front, right, info["block"], info["up"])
    tb = top.bbox()
    xs = top.to_sheet((cx, 0, 0))[0]
    sh.section_line((xs, tb[0][1] - 6), (xs, tb[1][1] + 6), "A", look=(1, 0))
    rb = right.bbox()
    sh.text(((rb[0][0] + rb[1][0]) / 2, rb[1][1] + 10), "A-A", 5, 0, "center", "bottom")
    notes = [f"指示なき寸法許容差は JIS B 0405-{TOL_CLASS} による。", "部品の形状・寸法は各部品図および3Dデータ (STEP AP242) による。"] \
        + info.get("notes", [])
    ybot = _notes(sh, notes, (16, 112))
    bom = info["bom"]
    cols = [("品番", 10), ("品名", 44), ("図番・規格", 40), ("材料", 34), ("数量", 10), ("備考", 34)]
    rows = [[str(i + 1), *r[:3], str(r[3]), r[4]] for i, r in enumerate(bom)]
    sh.text((16, ybot - 4), "部品欄", 3.2, 0, "left", "bottom")
    table(sh, (16, ybot - 5), cols, rows)
    sh.meta["bom"] = rows
    _need(sh, "bom.instances", sum(r[3] for r in bom), info["parts_count"])
    return sh
