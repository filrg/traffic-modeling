import html, sys

W, H = 1060, 320

INK, MUTED, LINE, GRID = "#1F2430", "#5F6875", "#8A94A3", "#D7DBE0"
TR_F, TR_S, TR_P = "#FFFFFF", "#7A838F", "#F4F6F8"
GN_F, GN_S, GN_A, GN_B = "#EAF5EA", "#4E8A52", "#3C7A41", "#BCD9BE"
OR_F, OR_S, OR_A, OR_B = "#FDF2E4", "#C7862F", "#B0731F", "#E8CBA4"
PU_F, PU_S, PU_A = "#F3ECF8", "#7E579B", "#6E4A89"
BL_F, BL_S, BL_A = "#E9F1FB", "#3E6FA3", "#31598A"
DK_F, DK_A = "#2F3B52", "#9CC0EC"

shapes = []


def rect(x, y, w, h, fill, stroke, rx=0, sw=1, dash=False):
    shapes.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                       rx=rx, sw=sw, dash=dash))


def ellipse(x, y, w, h, fill, stroke, sw=1):
    shapes.append(dict(k="ell", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw))


def text(x, y, w, h, lines, size=9, bold=False, color=INK, align="center", lh=1.3):
    if isinstance(lines, str):
        lines = [lines]
    shapes.append(dict(k="text", x=x, y=y, w=w, h=h, lines=lines, size=size,
                       bold=bold, color=color, align=align, lh=lh))


def poly(pts, stroke, sw=1.4, fill="none", dash=False):
    shapes.append(dict(k="poly", pts=pts, stroke=stroke, sw=sw, fill=fill, dash=dash))


def arrow(pts, stroke=LINE, sw=1.7, dash=False):
    shapes.append(dict(k="arrow", pts=pts, stroke=stroke, sw=sw, dash=dash))


# ------------------------------------------------------------------- icons ---
def ic_gate(x, y, c):                                   # slow on/off step
    poly([(x, y + 12), (x + 4, y + 12), (x + 4, y + 3), (x + 9, y + 3),
          (x + 9, y + 12), (x + 13, y + 12), (x + 13, y + 5), (x + 16, y + 5)],
         c, 1.5)


def ic_noise(x, y, c):                                  # fast fluctuation
    poly([(x, y + 9), (x + 2.5, y + 4), (x + 5, y + 11), (x + 7.5, y + 6),
          (x + 10, y + 12), (x + 12.5, y + 5), (x + 16, y + 9)], c, 1.5)


def ic_counts(x, y, c):                                 # irregular count bars
    for i, hgt in enumerate((5, 11, 7, 13)):
        rect(x + i * 4.4, y + 13 - hgt, 2.8, hgt, c, c)


def ic_chip(x, y, c):                                   # gpu / hardware
    rect(x + 3, y + 2, 11, 11, "none", c, rx=2, sw=1.4)
    for i in range(3):
        poly([(x, y + 4.5 + i * 3.2), (x + 3, y + 4.5 + i * 3.2)], c, 1.2)
        poly([(x + 14, y + 4.5 + i * 3.2), (x + 17, y + 4.5 + i * 3.2)], c, 1.2)


def ic_mixture(x, y, c):                                # 3 unequal components
    ellipse(x, y + 6, 7, 7, "none", c, 1.3)
    ellipse(x + 5, y + 2, 11, 11, "none", c, 1.3)
    ellipse(x + 12, y + 7, 5, 5, "none", c, 1.3)


def ic_weights(x, y, c):                                # mixing weights
    for i, f in enumerate((0.7, 0.35, 0.55)):
        yy = y + 3 + i * 4.5
        poly([(x, yy), (x + 16, yy)], c, 1.2)
        ellipse(x + 16 * f - 2, yy - 2, 4, 4, c, "none")


def ic_dice(x, y, c):
    rect(x, y, 17, 17, "none", c, rx=3, sw=1.4)
    for ox, oy in ((4.5, 4.5), (12.5, 4.5), (8.5, 8.5), (4.5, 12.5), (12.5, 12.5)):
        ellipse(x + ox - 1.6, y + oy - 1.6, 3.2, 3.2, c, "none")


def ic_rack(x, y, c):
    for i in range(3):
        rect(x, y + i * 6, 17, 4.6, "none", c, rx=1, sw=1.1)
        ellipse(x + 2, y + i * 6 + 1.6, 1.6, 1.6, c, "none")


def ic_ramp(x, y, c):                                   # load: rising bars
    for i, hgt in enumerate((4, 8, 12)):
        rect(x + i * 5.5, y + 13 - hgt, 3.6, hgt, c, c)


def ic_span(x, y, c):                                   # horizon: extending time
    poly([(x, y + 3), (x, y + 13)], c, 1.3)
    poly([(x, y + 8), (x + 13, y + 8)], c, 1.3)
    poly([(x + 11, y + 5), (x + 16, y + 8), (x + 11, y + 11)], c, 1.3)


def ic_db(x, y, c):
    ellipse(x, y + 9, 26, 8, "#FFFFFF", c, 1.2)
    poly([(x, y + 4), (x, y + 13)], c, 1.2)
    poly([(x + 26, y + 4), (x + 26, y + 13)], c, 1.2)
    ellipse(x, y, 26, 8, "#FFFFFF", c, 1.2)

def ic_place(x, y, c):                                  # events placed in a bin
    poly([(x + 1, y + 3), (x + 1, y + 12), (x + 16, y + 12), (x + 16, y + 3)], c, 1.2)
    for ox in (4.5, 8.0, 12.5):
        ellipse(x + ox - 1.7, y + 5.5, 3.4, 3.4, c, "none")


# ------------------------------------------------------------------ layout ---
LB_X, LB_W = 20, 150
MID_X, MID_W = 225, 550
RT_X, RT_W = 835, 205

GN_Y, GN_H = 16, 128
OR_Y, OR_H = 176, 128
LB_Y, LB_H = 56, 208
UH, UGAP = 88, 44
SPAN = MID_W - 28
BW4 = (SPAN - 3 * UGAP) / 4.0
BW3 = (SPAN - 2 * UGAP) / 3.0
GX = [MID_X + 14 + i * (BW4 + UGAP) for i in range(4)]
OX = [MID_X + 14 + i * (BW3 + UGAP) for i in range(3)]

PU_Y, PU_H = 16, 66
BL_Y, BL_H = 104, 96
DK_Y, DK_H = 222, 82

GN_MID, OR_MID = GN_Y + 64, OR_Y + 64
TRUNK = 822


def block(x, y, w, h, fill, stroke, title, tcolor=INK, dash=False):
    rect(x, y, w, h, fill, stroke, rx=9, sw=1.4, dash=dash)
    text(x, y + 6, w, 18, title, size=13, bold=True, color=tcolor)


def unit(x, y, w, title, method, icon, accent, border):
    rect(x, y, w, UH, "#FFFFFF", border, rx=6, sw=1.1)
    icon(x + w / 2 - 8, y + 9, accent)
    text(x + 5, y + 28, w - 10, 26, title, size=10, bold=True, color=accent, lh=1.2)
    text(x + 5, y + 56, w - 10, 24, method, size=8.2, color=MUTED, lh=1.3)


def chain(xs, w, y, labels, stroke):
    for i, lab in enumerate(labels):
        x0, x1 = xs[i] + w + 5, xs[i + 1] - 5
        arrow([(x0, y + UH / 2), (x1, y + UH / 2)], stroke=stroke, sw=1.6)
        if lab:
            n = len(lab) if isinstance(lab, list) else 1
            text(x0 - 5, y + UH / 2 - 12 - 5 * n, x1 - x0 + 10, 11 * n, lab,
                 size=8, color=MUTED, lh=1.25)


# --------------------------------------------------------- production trace --
block(LB_X, LB_Y, LB_W, LB_H, TR_F, TR_S, "Production Trace")
ic_db(LB_X + 14, LB_Y + 40, TR_S)
text(LB_X + 48, LB_Y + 36, LB_W - 58, 28, ["Seren SFT subset", "of AcmeTrace"],
     size=8.8, align="left", lh=1.35)

px, py, pw = LB_X + 10, LB_Y + 78, LB_W - 20
rect(px, py, pw, LB_H - 92, TR_P, "#E1E5EA", rx=6)
text(px + 10, py + 10, pw - 20, 12, "Arrivals", size=8.8, bold=True, align="left")
text(px + 10, py + 24, pw - 20, 26, ["submit time  T", "→ 15-min bins"],
     size=8.4, align="left", lh=1.45)
poly([(px + 10, py + 52), (px + pw - 10, py + 52)], "#DDE2E7", 1.0)
text(px + 10, py + 60, pw - 20, 12, "Jobs", size=8.8, bold=True, align="left")
text(px + 10, py + 74, pw - 20, 36, ["duration  D", "GPU count  n", "clock  f"],
     size=8.4, align="left", lh=1.45)

# ------------------------------------------------------------ arrival model --
block(MID_X, GN_Y, MID_W, GN_H, GN_F, GN_S, "GLI-NB Arrival Model")
uy = GN_Y + 34
unit(GX[0], uy, BW4, "Gate Process", ["two-scale AR(1),", "slow quiet ↔ busy"],
     ic_gate, GN_A, GN_B)
unit(GX[1], uy, BW4, "Fluctuation", ["fast AR(1) latent,", "sets intensity λ(t)"],
     ic_noise, GN_A, GN_B)
unit(GX[2], uy, BW4, "Batch Counts", ["N ~ NegBin(λ, κ),", "per 15-min bin"],
     ic_counts, GN_A, GN_B)
unit(GX[3], uy, BW4, ["Event", "Reconstruction"],
     ["batch times in bin,", "+ jobs per batch"], ic_place, GN_A, GN_B)
chain(GX, BW4, uy, [None, ["intensity", "λ(t)"], ["batch", "counts N"]], GN_S)

# ----------------------------------------------------------- job size model --
block(MID_X, OR_Y, MID_W, OR_H, OR_F, OR_S, "Job Size Model")
vy = OR_Y + 34
unit(OX[0], vy, BW3, "Demand Inverter", ["w = D ÷ s(n, f),",
     "hardware-independent work units"], ic_chip, OR_A, OR_B)
unit(OX[1], vy, BW3, "Size Mixture", ["3-component lognormal",
     "over work units"], ic_mixture, OR_A, OR_B)
unit(OX[2], vy, BW3, "Activity Coupling", ["mixture weights follow log N,",
     "busy bins draw smaller jobs"], ic_weights, OR_A, OR_B)
chain(OX, BW3, vy, [["work", "units"], None], OR_S)

# ------------------------------------------------------------- right column --
block(RT_X, PU_Y, RT_W, PU_H, PU_F, PU_S, "Generate Workload")
ic_dice(RT_X + 16, PU_Y + 38, PU_A)
text(RT_X + 42, PU_Y + 36, RT_W - 54, 24,
     ["each job:  arrival time", "+ computational demand"], size=8.8, align="left",
     lh=1.35)

block(RT_X, BL_Y, RT_W, BL_H, BL_F, BL_S, "Load & Horizon Scaling", dash=True)
text(RT_X + 10, BL_Y + 25, RT_W - 20, 11, "two adjustable parameters", size=8.4,
     color=MUTED)
for i, (icon, label, cap) in enumerate(
        [(ic_ramp, "α   Load", "arrival intensity,  ×0.5 … ×16"),
         (ic_span, "T   Horizon", "trace length,  1 … 180 days")]):
    ry = BL_Y + 42 + i * 28
    icon(RT_X + 16, ry + 1, BL_A)
    text(RT_X + 40, ry, 78, 13, label, size=9.4, bold=True, color=BL_A, align="left")
    text(RT_X + 40, ry + 13, RT_W - 52, 11, cap, size=8, color=MUTED, align="left")

block(RT_X, DK_Y, RT_W, DK_H, DK_F, DK_F, "Data Center Evaluation", tcolor="#FFFFFF")
ic_rack(RT_X + 16, DK_Y + 36, DK_A)
text(RT_X + 42, DK_Y + 32, RT_W - 54, 40,
     ["scheduling policy", "energy consumption", "QoS & performance"],
     size=8.8, color="#E7EDF6", align="left", lh=1.5)

# ---------------------------------------------------------------- dataflow ---
GAP_L, GAP_R = MID_X - LB_X - LB_W, TRUNK - MID_X - MID_W
arrow([(LB_X + LB_W, GN_MID), (MID_X, GN_MID)])
text(LB_X + LB_W, GN_MID - 15, GAP_L, 11, "15-min counts", size=7.8, color=MUTED)
arrow([(LB_X + LB_W, OR_MID), (MID_X, OR_MID)])
text(LB_X + LB_W, OR_MID - 15, GAP_L, 11, "D,  n,  f", size=7.8, color=MUTED)

poly([(MID_X + MID_W, GN_MID), (TRUNK, GN_MID)], LINE, 1.7)
poly([(MID_X + MID_W, OR_MID), (TRUNK, OR_MID)], LINE, 1.7)
poly([(TRUNK, PU_Y + PU_H / 2), (TRUNK, OR_MID)], LINE, 1.7)
arrow([(TRUNK, PU_Y + PU_H / 2), (RT_X, PU_Y + PU_H / 2)])
text(MID_X + MID_W, GN_MID - 24, GAP_R, 20, ["job", "timestamps"], size=7.8,
     color=MUTED, lh=1.25)
text(MID_X + MID_W, OR_MID - 24, GAP_R, 20, ["work", "per job"], size=7.8,
     color=MUTED, lh=1.25)

arrow([(RT_X + RT_W / 2, PU_Y + PU_H), (RT_X + RT_W / 2, BL_Y)])
arrow([(RT_X + RT_W / 2, BL_Y + BL_H), (RT_X + RT_W / 2, DK_Y)])

cxx = GX[2] + BW4 / 2
arrow([(cxx, GN_Y + GN_H), (cxx, OR_Y)], stroke=OR_S, sw=1.5, dash=True)
text(cxx + 7, GN_Y + GN_H + 8, 96, 12, "bin activity  N", size=8.2, color=OR_A,
     align="left")


# -------------------------------------------------------------------- emit ---
def esc(s):
    return html.escape(s, quote=True)


def svg_out(path):
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H)]
    for s in shapes:
        d = ' stroke-dasharray="5 3"' if s.get("dash") else ''
        if s["k"] == "rect":
            o.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="%s" '
                     'fill="%s" stroke="%s" stroke-width="%s"%s/>'
                     % (s["x"], s["y"], s["w"], s["h"], s["rx"], s["fill"],
                        s["stroke"], s["sw"], d))
        elif s["k"] == "ell":
            o.append('<ellipse cx="%.2f" cy="%.2f" rx="%.2f" ry="%.2f" fill="%s" '
                     'stroke="%s" stroke-width="%s"/>'
                     % (s["x"] + s["w"] / 2, s["y"] + s["h"] / 2, s["w"] / 2,
                        s["h"] / 2, s["fill"], s["stroke"], s["sw"]))
        elif s["k"] == "poly":
            pts = " ".join("%.2f,%.2f" % p for p in s["pts"])
            o.append('<polyline points="%s" fill="%s" stroke="%s" stroke-width="%s" '
                     'stroke-linejoin="round" stroke-linecap="round"%s/>'
                     % (pts, s["fill"], s["stroke"], s["sw"], d))
        elif s["k"] == "arrow":
            import math
            pts = s["pts"]
            o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" '
                     'stroke-linejoin="round"%s/>'
                     % (" ".join("%.2f,%.2f" % p for p in pts), s["stroke"], s["sw"], d))
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            a = math.atan2(y2 - y1, x2 - x1)
            L, Wd = 8.0, 3.6
            tri = [(x2, y2),
                   (x2 - L * math.cos(a) + Wd * math.sin(a),
                    y2 - L * math.sin(a) - Wd * math.cos(a)),
                   (x2 - L * math.cos(a) - Wd * math.sin(a),
                    y2 - L * math.sin(a) + Wd * math.cos(a))]
            o.append('<polygon points="%s" fill="%s"/>'
                     % (" ".join("%.2f,%.2f" % p for p in tri), s["stroke"]))
        elif s["k"] == "text":
            nl = len(s["lines"])
            lh = s["size"] * s["lh"]
            cy0 = s["y"] + s["h"] / 2 - (nl - 1) * lh / 2
            if s["align"] == "left":
                xa, anc = s["x"], "start"
            elif s["align"] == "right":
                xa, anc = s["x"] + s["w"], "end"
            else:
                xa, anc = s["x"] + s["w"] / 2, "middle"
            fw = ' font-weight="bold"' if s["bold"] else ''
            for i, ln in enumerate(s["lines"]):
                o.append('<text x="%.2f" y="%.2f" fill="%s" font-size="%s" '
                         'text-anchor="%s" dominant-baseline="central"%s>%s</text>'
                         % (xa, cy0 + i * lh, s["color"], s["size"], anc, fw, esc(ln)))
    o.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(o))


def drawio_out(path):
    cells = []
    ctr = [0]

    def nid():
        ctr[0] += 1
        return "n%d" % ctr[0]

    for s in shapes:
        d = "dashed=1;" if s.get("dash") else "dashed=0;"
        if s["k"] in ("rect", "ell"):
            if s["k"] == "ell":
                base = "ellipse;"
            elif s.get("rx"):
                base = "rounded=1;absoluteArcSize=1;arcSize=%d;" % (2 * s["rx"])
            else:
                base = "rounded=0;"
            st = ("%swhiteSpace=wrap;html=1;%sfillColor=%s;strokeColor=%s;strokeWidth=%s;"
                  % (base, d, s["fill"], s["stroke"], s["sw"]))
            cells.append('<mxCell id="%s" value="" style="%s" vertex="1" parent="1">'
                         '<mxGeometry x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                         'as="geometry"/></mxCell>'
                         % (nid(), esc(st), s["x"], s["y"], s["w"], s["h"]))
        elif s["k"] == "text":
            val = "&lt;br&gt;".join(esc(l) for l in s["lines"])
            st = ("text;html=1;whiteSpace=nowrap;strokeColor=none;fillColor=none;align=%s;"
                  "verticalAlign=middle;fontSize=%g;fontColor=%s;spacing=0;spacingLeft=0;"
                  "spacingRight=0;%s" % (s["align"], s["size"], s["color"],
                                         "fontStyle=1;" if s["bold"] else ""))
            cells.append('<mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">'
                         '<mxGeometry x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                         'as="geometry"/></mxCell>'
                         % (nid(), val, esc(st), s["x"], s["y"], s["w"], s["h"]))
        else:
            pts = s["pts"]
            if len(pts) < 2:
                continue
            end = "blockThin;endFill=1;endSize=4" if s["k"] == "arrow" else "none"
            st = ("edgeStyle=none;html=1;rounded=0;%sendArrow=%s;startArrow=none;"
                  "strokeColor=%s;strokeWidth=%g;" % (d, end, s["stroke"], max(1, s["sw"])))
            mid = "".join('<mxPoint x="%.2f" y="%.2f"/>' % p for p in pts[1:-1])
            cells.append('<mxCell id="%s" value="" style="%s" edge="1" parent="1">'
                         '<mxGeometry relative="1" as="geometry">'
                         '<mxPoint x="%.2f" y="%.2f" as="sourcePoint"/>'
                         '<mxPoint x="%.2f" y="%.2f" as="targetPoint"/>%s'
                         '</mxGeometry></mxCell>'
                         % (nid(), esc(st), pts[0][0], pts[0][1], pts[-1][0], pts[-1][1],
                            ('<Array as="points">%s</Array>' % mid) if mid else ''))
    body = "\n        ".join(cells)
    xml = ('<mxfile host="app.diagrams.net">\n  <diagram name="system-overview" id="sysov">\n'
           '    <mxGraphModel dx="1400" dy="800" grid="0" gridSize="10" guides="1" '
           'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
           'pageWidth="%d" pageHeight="%d" math="0" shadow="0">\n      <root>\n'
           '        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
           '        %s\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n'
           % (W, H, body))
    open(path, "w", encoding="utf-8").write(xml)


out = sys.argv[1]
svg_out(out + ".svg")
drawio_out(out + ".drawio")
print("shapes:", len(shapes))
