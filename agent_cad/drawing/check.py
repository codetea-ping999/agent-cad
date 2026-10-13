"""Automatic checks on a generated drawing sheet.

- dims:     every dimension measured from the B-rep equals the design intent
- fits:     toleranced sizes carry the JIS B 0401 deviations of their fit code
- features: every required feature (holes, taps, sections) was found on the model
- content:  title block, general tolerance, roughness (general and local) and projection method present
- layout:   no two texts overlap and nothing leaves the drawing frame
"""
import math, re
import numpy as np
from .sheet import text_box, boxes_overlap
from ..standards import jis

DIM_TOL = 0.005      # mm / deg: drawing value vs design intent
REQUIRED_TITLE = ["図名", "図番", "材料", "質量 (計算)", "尺度", "投影法", "単位", "普通公差"]


def _text_box(it):
    _, x, y, s, size, rot, ha, va = it
    return text_box((x, y), s, size, rot, ha, va)


_overlap = boxes_overlap


def check_sheet(sheet):
    res = []
    add = lambda name, ok, **kw: res.append({"check": name, "pass": bool(ok), **kw})
    for d in sheet.meta.get("dims", []):
        add(f"dim.{d['name']}", abs(d["measured"] - d["intent"]) <= DIM_TOL,
            measured=d["measured"], intent=d["intent"], text=d["text"])
        if d.get("tol"):
            m = re.search(r"φ([\d.]+)\s+([A-Za-z]+\d+)", d["text"])
            up, lo = jis.fit_deviations(float(m.group(1)), m.group(2))
            ok = abs(float(d["tol"][0]) - up) < 1e-9 and abs(float(d["tol"][1]) - lo) < 1e-9
            add(f"fit.{d['name']}", ok, code=m.group(2), deviations=d["tol"])
    for f in sheet.meta.get("features", []):
        add(f"feature.{f['name']}", f["found"] == f["expected"], found=f["found"], expected=f["expected"])
    tb = sheet.meta.get("title_block", {})
    add("content.title_block", all(tb.get(k) for k in REQUIRED_TITLE), missing=[k for k in REQUIRED_TITLE if not tb.get(k)])
    add("content.projection", tb.get("投影法") == "第三角法")
    add("content.general_tolerance", "JIS B 0405" in tb.get("普通公差", "") and
        any("JIS B 0405" in n for n in sheet.meta.get("notes", [])))
    add("content.general_roughness", bool(sheet.meta.get("roughness_general")))
    add("content.finish", bool(sheet.meta.get("finish")), finish=sheet.meta.get("finish"))
    for r in sheet.meta.get("roughness_local", []):    # local surface texture shown where the design asks for it
        add(f"content.roughness.{r['name']}", r["drawn"] == r["required"], drawn=r["drawn"], required=r["required"])
    add("content.views", len(sheet.views) >= 2, views=[v.name for v in sheet.views])
    # layout
    texts = [it for it in sheet.items if it[0] == "text"]
    boxes = [_text_box(t) for t in texts]
    clashes = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i], boxes[j]):
                clashes.append((texts[i][3], texts[j][3]))
    add("layout.text_overlap", not clashes, clashes=clashes[:10])
    # text on top of part outlines (sampled every 0.5 mm)
    pts = []
    for it in sheet.items:
        if it[0] == "line" and it[2] == "outline":
            P = it[1]
            for a, b in zip(P[:-1], P[1:]):
                k = max(1, int(np.linalg.norm(b - a) / 0.5))
                pts.append(a + np.outer(np.linspace(0, 1, k + 1), b - a))
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    hit = []
    for t, (b0, b1) in zip(texts, boxes):
        inside = (pts[:, 0] > b0[0] + 0.3) & (pts[:, 0] < b1[0] - 0.3) & (pts[:, 1] > b0[1] + 0.3) & (pts[:, 1] < b1[1] - 0.3)
        if inside.any():
            hit.append(t[3])
    add("layout.text_on_geometry", not hit, texts=hit[:10])
    W, H = sheet.size
    out = [t[3] for t, (b0, b1) in zip(texts, boxes) if b0[0] < 10 or b0[1] < 10 or b1[0] > W - 10 or b1[1] > H - 10]
    lines_out = sum(1 for it in sheet.items if it[0] in ("line", "fill") and it[2:3] != ("frame",)
                    and ((it[1] < 9.99).any() or (it[1][:, 0] > W - 9.99).any() or (it[1][:, 1] > H - 9.99).any()))
    add("layout.inside_frame", not out and lines_out == 0, texts=out[:5], lines=lines_out)
    return {"part": sheet.meta.get("part"), "pass": all(r["pass"] for r in res), "checks": res}
