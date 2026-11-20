"""Plain A4 calculation reports (計算書) as PDF: headings, paragraphs, tables and figures
flowed onto pages with matplotlib, in the Japanese font the drawings use.

    doc = Report("締結計算書", "AC-1001 / AC-1002 ステム", today)
    doc.h1("1. 概要"); doc.para("..."); doc.table(["項目", "値"], rows); doc.figure(fig)
    doc.save("out/fastener/締結計算書.pdf")
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .drawing.render import _font

A4 = (210.0, 297.0)
MARGIN = 16.0
PT = 72 / 25.4


def _wrap(text, width_mm, size_pt):
    """Wrap by character count: CJK glyphs are ~1 em wide, Latin ~0.55 em."""
    em = size_pt / PT
    out = []
    for raw in str(text).split("\n"):
        line, w = "", 0.0
        for ch in raw:
            cw = em * (1.0 if ord(ch) > 0x2E80 else 0.56)
            if w + cw > width_mm and line:
                out.append(line); line, w = "", 0.0
            line += ch; w += cw
        out.append(line)
    return out


class Report:
    def __init__(self, title, subject, date, doc_no=""):
        self.title, self.subject, self.date, self.doc_no = title, subject, date, doc_no
        self.items = []

    def h1(self, t):
        self.items.append(("h", t, 12.5))

    def h2(self, t):
        self.items.append(("h", t, 10.5))

    def para(self, t, size=8.5):
        self.items.append(("p", t, size))

    def bullets(self, items, size=8.0):
        for t in items:
            self.items.append(("p", "・" + t, size))

    def table(self, header, rows, widths=None, size=7.2, highlight=None):
        """highlight(row) -> colour or None, per row."""
        self.items.append(("t", header, rows, widths, size, highlight))

    def figure(self, fig, height_mm=90):
        self.items.append(("f", fig, height_mm))

    def pagebreak(self):
        self.items.append(("br",))

    # ------------------------------------------------------------------ layout
    def save(self, path):
        font = _font()
        W, H = A4
        pages, state = [], {}

        def new_page():
            fig = plt.figure(figsize=(W / 25.4, H / 25.4))
            ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
            pages.append(fig)
            n = len(pages)
            ax.plot([MARGIN, W - MARGIN], [H - 14, H - 14], lw=0.5, color="k")
            ax.text(MARGIN, H - 12.5, f"{self.title}  {self.subject}", fontsize=7, family=font, va="bottom")
            ax.text(W - MARGIN, H - 12.5, f"{self.doc_no}  {self.date}", fontsize=7, family=font, va="bottom", ha="right")
            ax.text(W / 2, 8, f"- {n} -", fontsize=7, family=font, ha="center")
            state.update(ax=ax, y=H - 20, fig=fig)

        def need(h):
            if state["y"] - h < 14:
                new_page()

        new_page()
        ax = state["ax"]
        ax.text(W / 2, H - 34, self.title, fontsize=20, family=font, ha="center", va="center")
        ax.text(W / 2, H - 44, self.subject, fontsize=11, family=font, ha="center", va="center")
        state["y"] = H - 54
        for it in self.items:
            k = it[0]
            if k == "br":
                new_page(); continue
            if k in ("h", "p"):
                lines = _wrap(it[1], W - 2 * MARGIN - (0 if k == "h" else 2), it[2])
                lh = it[2] / PT * 1.45
                need(lh * len(lines) + (3 if k == "h" else 1))
                if k == "h":
                    state["y"] -= 2.5
                for ln in lines:
                    state["ax"].text(MARGIN + (0 if k == "h" else 2), state["y"], ln, fontsize=it[2], family=font, va="top",
                                     weight="bold" if k == "h" else "normal")
                    state["y"] -= lh
                state["y"] -= 1.2
            elif k == "t":
                _, header, rows, widths, size, hl = it
                tw = W - 2 * MARGIN
                widths = widths or [1] * len(header)
                cw = [tw * w / sum(widths) for w in widths]
                lh = size / PT * 1.35

                def row_lines(r):
                    return [_wrap(str(c), cw[i] - 2, size) for i, c in enumerate(r)]

                def draw_row(cells, bold=False, fill=None):
                    n = max(len(c) for c in cells)
                    h = n * lh + 1.6
                    need(h)
                    a, y = state["ax"], state["y"]
                    x = MARGIN
                    if fill:
                        a.add_patch(plt.Rectangle((MARGIN, y - h), tw, h, color=fill, lw=0))
                    for i, c in enumerate(cells):
                        for j, ln in enumerate(c):
                            a.text(x + 1, y - 0.8 - j * lh, ln, fontsize=size, family=font, va="top",
                                   weight="bold" if bold else "normal")
                        x += cw[i]
                    a.plot([MARGIN, MARGIN + tw], [y - h, y - h], lw=0.3, color="0.4")
                    state["y"] = y - h
                need(lh * 3)
                state["ax"].plot([MARGIN, MARGIN + tw], [state["y"], state["y"]], lw=0.6, color="k")
                draw_row(row_lines(header), True, "0.9")
                for r in rows:
                    draw_row(row_lines(r), False, hl(r) if hl else None)
                state["y"] -= 3
            elif k == "f":
                _, fig, hmm = it
                need(hmm + 2)
                w_in, h_in = fig.get_size_inches()
                wmm = min(W - 2 * MARGIN, hmm * w_in / h_in)
                hh = wmm * h_in / w_in
                fig.canvas.draw()
                img = fig.canvas.buffer_rgba()
                y = state["y"]
                iax = state["fig"].add_axes([(W - wmm) / 2 / W, (y - hh) / H, wmm / W, hh / H])
                iax.imshow(img); iax.axis("off")
                plt.close(fig)
                state["y"] = y - hh - 3
        with PdfPages(path) as pdf:
            for f in pages:
                pdf.savefig(f)
                plt.close(f)
        return len(pages)
