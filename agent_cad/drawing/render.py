"""Render a Sheet display list to PDF/PNG (matplotlib) and DXF (ezdxf)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from .sheet import STYLES

_JP_FONTS = ["IPAGothic", "IPAexGothic", "Noto Sans CJK JP", "TakaoGothic"]
PT_PER_MM = 72 / 25.4


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in _JP_FONTS:
        if n in names:
            return n
    raise RuntimeError("No Japanese font found for drawing PDFs (tried: " + ", ".join(_JP_FONTS) +
                       "). Install one, e.g. `apt-get install fonts-ipafont-gothic` or `fonts-noto-cjk`.")


def _font_file():
    """File name of the selected Japanese font, for the DXF text style."""
    import os
    return os.path.basename(font_manager.findfont(font_manager.FontProperties(family=_font()), fallback_to_default=False))


def to_pdf(sheet, path, dpi=200):
    W, H = sheet.size
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    font = _font()
    for it in sheet.items:
        k = it[0]
        if k == "line":
            lw, ls = STYLES[it[2]]
            dash = {"solid": "-", "dashed": (0, (6, 3)), "dashdot": (0, (12, 2, 1.5, 2))}[ls]
            ax.plot(it[1][:, 0], it[1][:, 1], color="black", lw=lw * PT_PER_MM, ls=dash, solid_capstyle="round")
        elif k == "fill":
            ax.add_patch(Polygon(it[1], closed=True, fc="black", ec="none"))
        elif k == "hatch":
            verts, codes = [], []
            for loop in it[1]:
                verts += list(loop) + [loop[0]]
                codes += [Path.MOVETO] + [Path.LINETO] * (len(loop) - 1) + [Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec="black", lw=0.13 * PT_PER_MM, hatch="////"))
        elif k == "text":
            _, x, y, s, size, rot, ha, va = it
            ax.text(x, y, s, fontsize=size * PT_PER_MM * 1.25, rotation=rot, ha=ha, va=va,
                    fontfamily=font, rotation_mode="anchor")
    matplotlib.rcParams["hatch.linewidth"] = 0.3
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def to_dxf(sheet, path):
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    doc = ezdxf.new("R2018", setup=True, units=4)  # mm
    doc.styles.add("JP", font=_font_file())   # the same Japanese font the PDF uses
    lt = {"solid": "CONTINUOUS", "dashed": "HIDDEN", "dashdot": "CENTER"}
    for name, (lw, ls) in STYLES.items():
        doc.layers.add(name.upper(), linetype=lt[ls], lineweight=int(round(lw * 100)))
    doc.layers.add("TEXT"); doc.layers.add("FILL")
    msp = doc.modelspace()
    al = {("left", "bottom"): TextEntityAlignment.BOTTOM_LEFT, ("center", "bottom"): TextEntityAlignment.BOTTOM_CENTER,
          ("center", "center"): TextEntityAlignment.MIDDLE_CENTER, ("left", "center"): TextEntityAlignment.MIDDLE_LEFT,
          ("left", "top"): TextEntityAlignment.TOP_LEFT, ("center", "top"): TextEntityAlignment.TOP_CENTER}
    for it in sheet.items:
        k = it[0]
        if k == "line":
            msp.add_lwpolyline(it[1].tolist(), dxfattribs={"layer": it[2].upper(), "ltscale": 0.5})
        elif k == "fill":
            h = msp.add_hatch(color=7, dxfattribs={"layer": "FILL"}); h.paths.add_polyline_path(it[1].tolist())
        elif k == "hatch":
            h = msp.add_hatch(dxfattribs={"layer": "HATCH"})
            h.set_pattern_fill("ANSI31", scale=0.5)
            for i, loop in enumerate(it[1]):
                h.paths.add_polyline_path(np.asarray(loop).tolist(), flags=1 if i == 0 else 0)
        elif k == "text":
            _, x, y, s, size, rot, ha, va = it
            t = msp.add_text(s, height=size, rotation=rot, dxfattribs={"layer": "TEXT", "style": "JP"})
            t.set_placement((x, y), align=al.get((ha, va), TextEntityAlignment.BOTTOM_LEFT))
    doc.saveas(path)
