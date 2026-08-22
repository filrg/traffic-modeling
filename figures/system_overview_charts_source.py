import math, html, random, sys

W, H = 1060, 400

INK, MUTED, LINE, GRID = "#1F2430", "#6B7280", "#98A2B3", "#D7DBE0"
TR_F, TR_S = "#FFFFFF", "#9AA3AE"
PANEL_F = "#FFFFFF"
GN_F, GN_S, GN_A, GN_T = "#EFF8EF", "#7BAE7F", "#4C8C50", "#DDEEDD"
OR_F, OR_S, OR_A, OR_T = "#FDF4E9", "#D9A45C", "#C07C2C", "#F6E4CB"
PU_F, PU_S, PU_A = "#F3ECF8", "#9673A6", "#7E579B"
BL_F, BL_S, BL_A = "#EAF1FA", "#7699C4", "#3E6FA3"
DK_F, DK_S, DK_A = "#2F3B52", "#2F3B52", "#9CC0EC"

shapes = []


def rect(x, y, w, h, fill, stroke, rx=0, sw=1, dash=False):
    shapes.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                       rx=rx, sw=sw, dash=dash))


def ellipse(x, y, w, h, fill, stroke, sw=1):
    shapes.append(dict(k="ell", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw))


def text(x, y, w, h, lines, size=9, bold=False, color=INK, align="center", lh=1.25):
    if isinstance(lines, str):
        lines = [lines]
    shapes.append(dict(k="text", x=x, y=y, w=w, h=h, lines=lines, size=size,
                       bold=bold, color=color, align=align, lh=lh))


def poly(pts, stroke, sw=1.4, fill="none", dash=False):
    shapes.append(dict(k="poly", pts=pts, stroke=stroke, sw=sw, fill=fill, dash=dash))


def arrow(pts, stroke=LINE, sw=1.6, dash=False):
    shapes.append(dict(k="arrow", pts=pts, stroke=stroke, sw=sw, dash=dash))


def axes(x, y, w, h):
    poly([(x, y), (x, y + h), (x + w, y + h)], GRID, 1.0)


def bars(x, y, w, h, vals, color, gap=0.30):
    n = len(vals)
    step = w / n
    bw = step * (1 - gap)
    for i, v in enumerate(vals):
        bh = max(0.8, v * h)
        rect(x + i * step + (step - bw) / 2, y + h - bh, bw, bh, color, color)


def curve(x, y, w, h, f, color, sw=1.4, n=44, fill="none"):
    pts = [(x + w * i / n, y + h - h * f(i / n)) for i in range(n + 1)]
    if fill != "none":
        pts = pts + [(x + w, y + h), (x, y + h)]
    poly(pts, color, sw, fill=fill)


# ---------------------------------------------------------------- layout ----
M = 20
LB_X, LB_W = M, 185
MID_X, MID_W = 232, 520
RT_X, RT_W = 782, 258
GN_Y, GN_H = 18, 160
OR_Y, OR_H = 208, 170
LB_Y, LB_H = 40, 316
P_W = 158
PX = [244, 412, 580]
TRUNK_L, TRUNK_R = 218, 767
GN_MID, OR_MID = GN_Y + 80, OR_Y + 85
PU_Y, PU_H = 18, 84
BL_Y, BL_H = 124, 128
DK_Y, DK_H = 274, 104
RT_CX = RT_X + RT_W / 2


def block(x, y, w, h, fill, stroke, title, tcolor=INK):
    rect(x, y, w, h, fill, stroke, rx=10, sw=1.3)
    text(x, y + 6, w, 18, title, size=13, bold=True, color=tcolor)


def panel(px, py, pw, ph, title, sub, accent):
    rect(px, py, pw, ph, PANEL_F, "#E3E7EC", rx=7, sw=1.0)
    text(px + 6, py + 5, pw - 12, 14, title, size=10.5, bold=True, color=accent)
    text(px + 6, py + 20, pw - 12, 22, sub, size=8.2, color=MUTED, lh=1.22)


# --------------------------------------------------- left: production trace --
block(LB_X, LB_Y, LB_W, LB_H, TR_F, TR_S, "Production Trace")

gx, gy = LB_X + 16, LB_Y + 34
for dy in (16, 8):
    ellipse(gx, gy + dy, 40, 13, "#EDEFF3", TR_S)
rect(gx, gy + 6.5, 40, 16, "#EDEFF3", "none")
poly([(gx, gy + 6.5), (gx, gy + 22.5)], TR_S, 1.0)
poly([(gx + 40, gy + 6.5), (gx + 40, gy + 22.5)], TR_S, 1.0)
ellipse(gx, gy, 40, 13, "#FFFFFF", TR_S)
text(LB_X + 62, LB_Y + 36, LB_W - 74, 30, ["Seren SFT subset", "of AcmeTrace"],
     size=9, color=INK, align="left", lh=1.3)

m1x, m1y, m1w, m1h = LB_X + 14, LB_Y + 92, LB_W - 28, 100
rect(m1x, m1y, m1w, m1h, "#FBFCFD", "#E3E7EC", rx=6)
text(m1x + 8, m1y + 4, m1w - 16, 13, "Batch arrivals", size=9.5, bold=True, align="left")
text(m1x + 8, m1y + 17, m1w - 16, 10, "batch = jobs at one timestamp", size=7.2,
     color=MUTED, align="left")
cx, cy, cw, ch = m1x + 36, m1y + 32, m1w - 46, 44
axes(cx, cy, cw, ch)
v1 = [0.14, 0.05, 0.09, 0.31, 0.11, 0.06, 0.22, 0.58, 0.16, 0.08, 0.04, 0.27,
      0.12, 0.07, 0.95, 0.34, 0.10, 0.05, 0.19, 0.63, 0.24, 0.09, 0.13, 0.06,
      0.41, 0.15, 0.07, 0.11]
bars(cx + 2, cy, cw - 4, ch, v1, "#8E99A8", gap=0.62)
text(m1x + 2, cy + 10, 33, 12, "# batches", size=6.9, color=MUTED)
text(cx, cy + ch + 3, cw, 10, "time", size=7.4, color=MUTED)

m2x, m2y, m2w, m2h = LB_X + 14, LB_Y + 202, LB_W - 28, 100
rect(m2x, m2y, m2w, m2h, "#FBFCFD", "#E3E7EC", rx=6)
text(m2x + 8, m2y + 4, m2w - 16, 13, "Job records", size=9.5, bold=True, align="left")
text(m2x + 8, m2y + 17, m2w - 16, 10, "one row per job", size=7.2,
     color=MUTED, align="left")
tbx, tby, tbw, trh = m2x + 12, m2y + 30, m2w - 24, 13
rect(tbx, tby, tbw, trh * 4, "#FFFFFF", "#E3E7EC", rx=3)
rect(tbx, tby, tbw, trh, "#EFF1F4", "none", rx=3)
rect(tbx, tby + trh - 3, tbw, 3, "#EFF1F4", "none")
poly([(tbx + tbw * 0.5, tby), (tbx + tbw * 0.5, tby + trh * 4)], "#E3E7EC", 1.0)
for i in range(1, 4):
    poly([(tbx, tby + trh * i), (tbx + tbw, tby + trh * i)], "#E3E7EC", 1.0)
rows = [("dur (s)", "n GPUs"), ("253", "8"), ("198", "1"), ("33", "2")]
for i, (a, b) in enumerate(rows):
    ry = tby + trh * i
    text(tbx, ry, tbw * 0.5, trh, a, size=7.2, bold=(i == 0),
         color=INK if i else MUTED)
    text(tbx + tbw * 0.5, ry, tbw * 0.5, trh, b, size=7.2, bold=(i == 0),
         color=INK if i else MUTED)
text(m2x + 12, tby + trh * 4 + 2, tbw, 10, "clock f = 1.0 GHz (max)", size=7.0,
     color=MUTED, align="left")

# -------------------------------------------------- middle: arrival model ----
block(MID_X, GN_Y, MID_W, GN_H, GN_F, GN_S, "GLI-NB Arrival Model")
gpy, gph = GN_Y + 28, GN_H - 38

panel(PX[0], gpy, P_W, gph, "Gate process (slow)", "quiet ↔ busy periods", GN_A)
ax, ay, aw, ah = PX[0] + 24, gpy + 50, P_W - 34, 46
axes(ax, ay, aw, ah)
lo, hi = ay + ah - 4, ay + 6
segs = [(0.00, 0.13, 0), (0.13, 0.36, 1), (0.36, 0.50, 0), (0.50, 0.78, 1),
        (0.78, 0.88, 0), (0.88, 1.00, 1)]
for a, b, s in segs:
    if s:
        rect(ax + aw * a, hi, aw * (b - a), lo - hi, GN_T, "none")
pts = []
for a, b, s in segs:
    yv = hi if s else lo
    pts += [(ax + aw * a, yv), (ax + aw * b, yv)]
poly(pts, GN_A, 1.6)
text(PX[0] + 3, hi - 6, 20, 12, "on", size=7.2, color=MUTED)
text(PX[0] + 3, lo - 6, 20, 12, "off", size=7.2, color=MUTED)
text(ax, ay + ah + 3, aw, 10, "time", size=7.4, color=MUTED)

panel(PX[1], gpy, P_W, gph, "Fluctuation (fast)", "burstiness inside busy periods", GN_A)
bx, by, bw2, bh2 = PX[1] + 24, gpy + 50, P_W - 34, 46
axes(bx, by, bw2, bh2)
rng = random.Random(11)
n = 40
ys, cur = [], 0.5
for i in range(n + 1):
    cur = 0.62 * cur + 0.38 * 0.5 + rng.uniform(-0.19, 0.19)
    ys.append(min(0.94, max(0.06, cur)))
poly([(bx, by + bh2 * 0.5), (bx + bw2, by + bh2 * 0.5)], GRID, 0.9, dash=True)
poly([(bx + bw2 * i / n, by + bh2 - bh2 * ys[i]) for i in range(n + 1)], GN_A, 1.3)
text(bx, by + bh2 + 3, bw2, 10, "time", size=7.4, color=MUTED)

panel(PX[2], gpy, P_W, gph, "Arrival rate λ(t)", "batch counts per 15-min bin", GN_A)
dx, dy, dw, dh = PX[2] + 24, gpy + 50, P_W - 34, 46
axes(dx, dy, dw, dh)
v3 = [0.12, 0.05, 0.08, 0.42, 0.68, 0.55, 0.88, 0.60, 0.30, 0.10, 0.06, 0.22,
      0.50, 0.34, 0.14, 0.07]
bars(dx + 2, dy, dw - 4, dh, v3, GN_A)
text(PX[2] + 1, dy + 12, 22, 12, "N", size=7.6, color=MUTED)
text(dx, dy + dh + 3, dw, 10, "time", size=7.4, color=MUTED)

# ------------------------------------------------- middle: job size model ----
block(MID_X, OR_Y, MID_W, OR_H, OR_F, OR_S, "Job Size Model")
opy, oph = OR_Y + 28, OR_H - 38

panel(PX[0], opy, P_W, oph, "Trace allocation", "duration on n GPUs at clock f", OR_A)
ex, ey, ew = PX[0] + 30, opy + 50, P_W - 44
for i in range(3):
    ly = ey + i * 14
    rect(ex, ly, ew, 10, "#F4F6F8", "#E3E7EC", rx=2)
    text(PX[0] + 3, ly, 26, 10, "gpu %d" % (i + 1), size=6.6, color=MUTED)
    rect(ex + 8, ly + 2, ew * 0.62, 6, OR_A, "none", rx=2)
poly([(ex + 8, ey + 44), (ex + 8, ey + 48), (ex + 8 + ew * 0.62, ey + 48),
      (ex + 8 + ew * 0.62, ey + 44)], MUTED, 0.9)
text(ex, ey + 49, ew, 10, "duration", size=7.4, color=MUTED)

panel(PX[1], opy, P_W, oph, "Computational demand",
      ["hardware-independent:", "work = duration ÷ s(n, f)"], OR_A)
fx, fy, fw, fh = PX[1] + 30, opy + 54, P_W - 42, 42
axes(fx, fy, fw, fh)
v4 = [0.26, 0.70, 1.00, 0.78, 0.48, 0.30, 0.22, 0.26, 0.30, 0.24, 0.14, 0.08,
      0.05, 0.03]
bars(fx + 2, fy, fw - 4, fh, v4, OR_A)
text(PX[1] + 2, fy + 10, 30, 12, "# jobs", size=6.9, color=MUTED)
text(fx, fy + fh + 3, fw, 10, "work units", size=7.4, color=MUTED)

panel(PX[2], opy, P_W, oph, "Size drawn on activity", "busier bins draw smaller jobs", OR_A)
hx, hy, hw, hh = PX[2] + 30, opy + 54, P_W - 42, 42
axes(hx, hy, hw, hh)
for off in (0.19, -0.19):
    poly([(hx, hy + hh - hh * (0.75 - 0.42 * 0.0 + off)),
          (hx + hw, hy + hh - hh * (0.75 - 0.42 * 1.0 + off))], OR_S, 0.9, dash=True)
curve(hx, hy, hw, hh, lambda t: 0.75 - 0.42 * t, OR_A, 1.5, n=2)
rng = random.Random(3)
for i in range(11):
    t = 0.05 + 0.09 * i
    yv = (0.75 - 0.42 * t) + rng.uniform(-0.16, 0.16)
    ellipse(hx + hw * t - 1.7, hy + hh - hh * yv - 1.7, 3.4, 3.4, OR_A, "none")
text(PX[2] + 2, hy + 8, 30, 14, ["work", "units"], size=6.9, color=MUTED, lh=1.15)
text(hx, hy + hh + 3, hw, 10, "bin activity  N", size=7.4, color=MUTED)

# ------------------------------------------------------------ right column ---
block(RT_X, PU_Y, RT_W, PU_H, PU_F, PU_S, "Generate Stochastic Workload")
dxx, dyy = RT_X + 26, PU_Y + 34
rect(dxx, dyy, 38, 38, "#FFFFFF", PU_A, rx=7, sw=1.3)
for ox, oy in ((10, 10), (28, 10), (19, 19), (10, 28), (28, 28)):
    ellipse(dxx + ox - 2.6, dyy + oy - 2.6, 5.2, 5.2, PU_A, "none")
text(RT_X + 78, PU_Y + 34, RT_W - 92, 40, ["arrivals × job sizes,", "sampled together"],
     size=9.3, color=INK, align="left", lh=1.35)

block(RT_X, BL_Y, RT_W, BL_H, BL_F, BL_S, "Load & Horizon Scaling")
text(RT_X + 10, BL_Y + 22, RT_W - 20, 11, "two adjustable parameters", size=8.2, color=MUTED)
IX, IW = RT_X + 84, RT_W - 96

# load: same window, taller bars
r1 = BL_Y + 36
text(RT_X + 12, r1 + 8, 68, 14, "Load  α", size=9.6, bold=True, align="left")
gw, gg = 56, 32
g0 = IX + 2
ib = r1 + 26
for gx0, vals, col in ((g0, [0.26, 0.14, 0.36, 0.20, 0.30], "#AFC0D6"),
                       (g0 + gw + gg, [0.66, 0.40, 0.92, 0.54, 0.76], BL_A)):
    bars(gx0, ib - 20, gw, 20, vals, col, gap=0.34)
    poly([(gx0 - 1, ib), (gx0 + gw + 1, ib)], GRID, 1.0)
am = g0 + gw + gg / 2
arrow([(am - 10, ib - 9), (am + 10, ib - 9)], stroke=BL_A, sw=1.3)
text(am - 14, ib - 24, 28, 10, "× α", size=8.4, bold=True, color=BL_A)
text(IX, r1 + 30, IW, 10, "arrivals per bin,  ×0.5 … ×16", size=7.4, color=MUTED,
     align="left")

# horizon: same intensity, longer span
r2 = BL_Y + 82
text(RT_X + 12, r2 + 8, 68, 14, "Horizon  T", size=9.6, bold=True, align="left")
tx0, tx1, ty = IX + 2, IX + IW - 6, r2 + 22
poly([(tx0, ty), (tx1, ty)], GRID, 1.0)
for i in range(7):
    tk = tx0 + (tx1 - tx0) * i / 6
    poly([(tk, ty), (tk, ty + 4)], GRID, 1.0)
tsp = tx0 + (tx1 - tx0) * 0.40
rect(tx0, ty - 9, tsp - tx0, 7, BL_A, "none", rx=2)
arrow([(tsp + 6, ty - 5.5), (tx1, ty - 5.5)], stroke=BL_A, sw=1.2, dash=True)
text(tx0, ty - 21, tsp - tx0, 10, "T", size=8.4, bold=True, color=BL_A)
text(IX, r2 + 30, IW, 10, "days simulated,  1 … 180 d", size=7.4, color=MUTED,
     align="left")

block(RT_X, DK_Y, RT_W, DK_H, DK_F, DK_S, "Data Center Evaluation", tcolor="#FFFFFF")
for i, b in enumerate(["Scheduling policy", "Energy consumption", "QoS & performance"]):
    by2 = DK_Y + 34 + i * 24
    ellipse(RT_X + 26, by2 + 3.5, 6, 6, DK_A, "none")
    text(RT_X + 42, by2, RT_W - 58, 13, b, size=9.6, color="#EDF1F7", align="left")

# ----------------------------------------------------------------- routing ---
poly([(LB_X + LB_W, 198), (TRUNK_L, 198)], LINE, 1.6)
poly([(TRUNK_L, GN_MID), (TRUNK_L, OR_MID)], LINE, 1.6)
arrow([(TRUNK_L, GN_MID), (MID_X, GN_MID)])
arrow([(TRUNK_L, OR_MID), (MID_X, OR_MID)])

poly([(MID_X + MID_W, GN_MID), (TRUNK_R, GN_MID)], LINE, 1.6)
poly([(MID_X + MID_W, OR_MID), (TRUNK_R, OR_MID)], LINE, 1.6)
poly([(TRUNK_R, OR_MID), (TRUNK_R, PU_Y + PU_H / 2)], LINE, 1.6)
arrow([(TRUNK_R, PU_Y + PU_H / 2), (RT_X, PU_Y + PU_H / 2)])

arrow([(RT_CX, PU_Y + PU_H), (RT_CX, BL_Y)])
arrow([(RT_CX, BL_Y + BL_H), (RT_CX, DK_Y)])

cxx = PX[2] + P_W / 2
arrow([(cxx, GN_Y + GN_H), (cxx, OR_Y)], stroke=OR_A, sw=1.4, dash=True)
text(cxx + 6, GN_Y + GN_H + 6, 96, 12, "bin activity  N", size=8, color=OR_A, align="left")


# -------------------------------------------------------------------- emit ---
def esc(s):
    return html.escape(s, quote=True)


def svg_out(path):
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#FFFFFF"/>' % (W, H)]
    for s in shapes:
        d = ' stroke-dasharray="4 3"' if s.get("dash") else ''
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
            pts = s["pts"]
            o.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" '
                     'stroke-linejoin="round"%s/>'
                     % (" ".join("%.2f,%.2f" % p for p in pts), s["stroke"], s["sw"], d))
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            ang = math.atan2(y2 - y1, x2 - x1)
            L, Wd = 7.5, 3.4
            tri = [(x2, y2),
                   (x2 - L * math.cos(ang) + Wd * math.sin(ang),
                    y2 - L * math.sin(ang) - Wd * math.cos(ang)),
                   (x2 - L * math.cos(ang) - Wd * math.sin(ang),
                    y2 - L * math.sin(ang) + Wd * math.cos(ang))]
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
            fill = s.get("fill", "none")
            st = ("edgeStyle=none;html=1;rounded=0;%sendArrow=%s;startArrow=none;"
                  "strokeColor=%s;strokeWidth=%g;%s"
                  % (d, end, s["stroke"], max(1, s["sw"]),
                     ("fillColor=%s;" % fill) if fill != "none" else ""))
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
