"""PIL-based figure rendering for the directed co-detection network.

Matplotlib's Agg renderer aborts the process in this environment, so all
figures are composed directly with PIL at a high raster scale and exported to
PNG (1000 dpi) plus a vector PDF rendered from the recorded drawing primitives.

Typography is Arial throughout, enforced rather than requested: font() raises
if no Arial-metric face is installed instead of letting PIL substitute a
default face, because a silent substitution would produce figures that look
finished but are not in the specified face.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from vector_pdf import VectorDraw, save_vector_pdf

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_INK = "#c3c2b7"
WHITE = "#ffffff"

GROUP_COLORS = {
    # Group-level fills, kept for the centrality figure's bars.
    "Short-chain PFAS focus": "#c0392b",
    "Other PFAS cluster": "#c0392b",
    "Long-chain PFAS cluster": "#c0392b",
    "Non-PFAS cluster": "#8c9199",
    "Focus PFAS compound": "#c0392b",
    "Other PFAS compound": "#c0392b",
    "Non-PFAS compound": "#8c9199",
}

# Per-category node fills.  Ten pollutant categories reach a network ring, so
# every node carries its own hue instead of one recessive slate.  All ten are
# light enough to take the node label *inside* the disc in dark ink (>= 4.6:1
# against #0b0b0b), and the set was optimized for all-pairs separation, then
# validated with the dataviz palette validator on surface #fcfcfb:
#   lightness band PASS, chroma floor PASS,
#   CVD all-pairs worst ΔE 8.7 (target >= 8) — #9696ff vs #00c5dd,
#   normal-vision all-pairs worst ΔE 15.3 (floor >= 15) — #00af70 vs #afba00.
# Five sit below 3:1 against the surface, so the relief rule applies: every node
# is directly labeled inside its own disc, and the legend repeats the mapping.
# PFAS keeps the reserved deep red — it is the story color (focus hubs, the
# asymmetry ramp), not "category N", and is the only fill that takes white ink.
CATEGORY_COLORS = {
    "PFAS": "#c0392b",
    "Pest.": "#ff80a5",
    "Pharm.": "#c17d00",
    "VDs": "#afba00",
    "OPEs": "#00af70",
    "PAEs": "#00c5dd",
    "DBPs": "#0084b0",
    "SHDs": "#9696ff",
    "Others": "#b049ff",
    "BPs": "#f300bc",
    # centrality-figure only (never reach a ring); adjacent-pairs validated
    "FAs": "#00a9df",
    "PCPs": "#db00e5",
}
CATEGORY_FALLBACK = "#8c9199"
# fills that need white rather than dark ink for an in-disc label
WHITE_INK_FILLS = {"#c0392b"}


def category_color(category: str) -> str:
    """Node fill for a pollutant category (grey fallback for unknown ones)."""
    return CATEGORY_COLORS.get(str(category), CATEGORY_FALLBACK)


def label_ink(fill: str) -> str:
    """Text colour that stays legible on the given node fill."""
    return WHITE if fill in WHITE_INK_FILLS else INK_PRIMARY


# Red sequential ramp — light (low A) → saturated dark (high A)
# Monotonic lightness 0.906 → 0.253, all steps pass HLS monotonicity check
SEQ_RED = ["#fad4d4", "#f5b8b8", "#f09090", "#ea6868",
           "#e34948", "#d03b3b", "#b83333", "#9e2a2a", "#822121", "#681919"]

DPI = 1000

# Raster scale: every layout dimension below is written in nominal units and
# multiplied by SCALE to get pixels, so SCALE alone sets the canvas pixel size.
# DPI is only metadata PIL stamps into the file -- it does not resample. At a
# fixed DPI, SCALE is therefore the only real fidelity knob: every glyph is
# rasterized at (point size x SCALE) pixels, so raising SCALE is what actually
# sharpens type and curves rather than just relabeling the same pixels.
#
# SCALE must stay an integer -- Image.new dimensions and ImageDraw width= are
# both strictly int -- so the ladder is 3 (old 800 dpi) -> 4 -> 5. At 5 the
# network panel prints ~7.8 in wide at 1000 dpi, wider than a journal page, but
# the layout is resolution-independent and gets scaled to the column width at
# submission; oversampling then lands *more* than 1000 dpi on the printed page,
# which is the safe direction. Keeping it integer also keeps every glyph an
# exact integer multiple of its design size.
SCALE = 5      # paired with DPI = 1000


def hex_rgb(value: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b)."""
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def blend(fg: str, bg: str, alpha: float) -> tuple[int, int, int]:
    """Alpha-blend two hex colours (alpha = weight of fg)."""
    f, b = hex_rgb(fg), hex_rgb(bg)
    return tuple(int(round(f[i] * alpha + b[i] * (1 - alpha))) for i in range(3))


def font(size: int, bold: bool = False):
    """Arial only -- Nature journal figures require a single sans face.

    Falls back to Liberation Sans (metric-compatible with Arial) and only then
    to Helvetica; a non-Arial fallback would break the typographic spec, so the
    substitution order never reaches Calibri or Segoe UI.

    If none of the three is installed this raises instead of returning
    ImageFont.load_default(): the default is a small bitmap face that ignores
    `size`, so it would silently render the whole figure in the wrong face at
    the wrong scale. Failing loudly is the only way the all-Arial requirement
    stays checkable.
    """
    names = ["arialbd.ttf" if bold else "arial.ttf",
             "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
             "helvetikab.ttf" if bold else "helvetica.ttf"]
    searched = []
    for name in names:
        for directory in (Path("C:/Windows/Fonts"),
                          Path.home() / "AppData/Local/Microsoft/Windows/Fonts"):
            path = directory / name
            searched.append(path)
            if path.exists():
                return ImageFont.truetype(str(path), size=int(size))
    raise RuntimeError(
        "no Arial-metric font found; figures would not be Arial. Looked for: "
        + ", ".join(str(p) for p in searched))


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt) -> tuple[int, int]:
    """Width and height of the rendered text bounding box in pixels."""
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def a_color(value: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Map the asymmetry index A onto the discrete SEQ_RED ramp."""
    if not np.isfinite(value):
        return GRIDLINE
    span = vmax - vmin
    t = (min(max(value, vmin), vmax) - vmin) / span if span > 0 else 0.0
    return SEQ_RED[min(len(SEQ_RED) - 1, int(t * len(SEQ_RED)))]


def save(img: Image.Image, png: Path, pdf: Path,
         vector_operations: list[tuple] | None = None) -> None:
    """Save the raster PNG at DPI and the matching vector PDF."""
    png.parent.mkdir(parents=True, exist_ok=True)
    img.save(png, dpi=(DPI, DPI))
    if vector_operations is None:
        raise ValueError("vector drawing operations are required for PDF export")
    save_vector_pdf(img.size, vector_operations, pdf, dpi=DPI)


def draw_arrow(draw: ImageDraw.ImageDraw, p0: tuple[float, float],
               p1: tuple[float, float], color, width: int, rad: float = 0.13,
               head: int = 0) -> None:
    """Quadratic-bezier arc with an arrowhead at p1."""
    x0, y0 = p0
    x1, y1 = p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * rad, my + dx * rad
    pts = []
    steps = 48
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        pts.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                    u * u * y0 + 2 * u * t * cy + t * t * y1))
    draw.line(pts, fill=color, width=width, joint="curve")
    if head <= 0:
        return
    # tangent from a point set back along the curve by roughly the head length,
    # so the head points along the path rather than along a fixed chord
    ax, ay = pts[-1]
    back = ax, ay
    for px, py in reversed(pts):
        if math.hypot(ax - px, ay - py) >= head * 0.9:
            back = (px, py)
            break
    ang = math.atan2(ay - back[1], ax - back[0])
    spread = math.radians(27)
    # filled triangle plus a short notch so the head reads as an arrow, not a blob
    draw.polygon(
        [
            (ax, ay),
            (ax - head * math.cos(ang - spread), ay - head * math.sin(ang - spread)),
            (ax - head * 0.62 * math.cos(ang), ay - head * 0.62 * math.sin(ang)),
            (ax - head * math.cos(ang + spread), ay - head * math.sin(ang + spread)),
        ],
        fill=color,
    )


# Descending ladder of in-disc label sizes. fit_node_label walks it from the
# top and takes the first size whose wrapped text block fits the disc, so the
# head of the ladder sets how large a short label (PFAS-1, PFBA) is allowed to
# get and the tail sets how small a long one (2-Ethylhexyl diphenyl phosphate)
# is willing to go before it just wraps further. The head was 9.5, which capped
# every short cluster label well below what the enlarged discs can hold.
NODE_FONT_STEPS = (17.0, 15.5, 14.0, 12.8, 11.6, 10.5, 9.5, 8.6, 7.8, 7.0, 6.3)
LABEL_MAX_LINES = 3

# Hub-label ladder, walked the same way but against the hub disc's own chord.
# The hub is the largest disc in the panel, so it gets the largest type.
HUB_FONT_STEPS = (26.0, 23.0, 20.0, 18.0, 16.0, 14.0, 12.0)


def _break_word(d: ImageDraw.ImageDraw, word: str, fnt, max_w: float) -> list[str]:
    """Hard-break a single word too wide for the disc, hyphenating the split."""
    out, cur = [], ""
    for ch in word:
        if text_size(d, cur + ch + "-", fnt)[0] > max_w and cur:
            out.append(cur + "-")
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _wrap_to_width(d: ImageDraw.ImageDraw, name: str, fnt, max_w: float) -> list[str]:
    """Greedy word wrap to max_w pixels, hard-breaking over-long words."""
    lines, cur = [], ""
    for word in str(name).split():
        if text_size(d, word, fnt)[0] > max_w:
            if cur:
                lines.append(cur)
                cur = ""
            lines.extend(_break_word(d, word, fnt, max_w))
            continue
        trial = f"{cur} {word}".strip()
        if text_size(d, trial, fnt)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def fit_node_label(d: ImageDraw.ImageDraw, name: str, scale: int,
                   cap_r: float) -> tuple[list[str], object, object, float]:
    """Largest label that fits inside a disc of radius `cap_r`.

    Returns (lines, label font, value font, required radius).  The radius is the
    circumradius of the text block plus a margin, so wrapped multi-line labels
    keep their corners inside the circle.  Long compound names wrap and, only if
    wrapping is not enough, step down through the font ladder.
    """
    fallback = None
    for pt in NODE_FONT_STEPS:
        f_lab = font(pt * scale, bold=True)
        f_val = font(max(4.4, pt * 0.84) * scale)
        best = None
        for n_lines in range(1, LABEL_MAX_LINES + 1):
            # allow the block to be as wide as a chord of the target circle
            max_w = 2 * cap_r * 0.80 if n_lines > 1 else 2 * cap_r * 0.86
            lines = _wrap_to_width(d, name, f_lab, max_w)
            if len(lines) > n_lines:
                continue
            lw = max(text_size(d, ln, f_lab)[0] for ln in lines)
            lh = max(text_size(d, ln, f_lab)[1] for ln in lines)
            vw, vh = text_size(d, "0.00", f_val)
            block_h = len(lines) * lh + (len(lines) - 1) * 2 * scale + vh + 3 * scale
            half_w, half_h = max(lw, vw) / 2, block_h / 2
            r = math.hypot(half_w, half_h) + 5 * scale
            if best is None or r < best[3]:
                best = (lines, f_lab, f_val, r)
            break
        if best is None:
            continue
        if fallback is None or best[3] < fallback[3]:
            fallback = best
        if best[3] <= cap_r:
            return best
    return fallback


def resolve_focus(nodes_df: pd.DataFrame, focus: tuple[str, ...],
                  n_fallback: int = 2) -> list[str]:
    """Focus nodes for the radial centre.

    At category resolution the named clusters (PFAS-1 / PFAS-2) exist as nodes.
    At compound resolution they do not, so fall back to the highest out-degree
    hubs, preferring members of the focus clusters when they qualify.
    """
    present = [f for f in focus if f in set(nodes_df["node"])]
    if present:
        return present
    ranked = nodes_df.sort_values(
        ["out_degree", "hub_score", "n_detected"], ascending=[False, False, False]
    )
    hubs = ranked[ranked["out_degree"] > 0]
    if hubs.empty:
        hubs = ranked
    return list(hubs["node"].head(n_fallback))

def select_targets(spokes: pd.DataFrame, detected: pd.Series, max_targets: int,
                   always_keep_prefix: str = "PFAS") -> pd.DataFrame:
    """Choose which targets to draw when a ring is overfull.

    Ranking by coverage alone is misleading: a cluster detected in only 8
    specimens trivially reaches coverage 1.00 against a hub present in 353/355
    specimens, so rare clusters crowd out the long-chain PFAS clusters
    (n = 282-346) that carry the real signal.  Targets are therefore ranked by
    detection breadth, and every PFAS cluster is pinned in regardless of rank.
    """
    if spokes.empty:
        return spokes
    work = spokes.copy()
    work["_n"] = work["target"].map(detected).fillna(0).astype(int)
    is_pfas = work["target"].str.startswith(always_keep_prefix)
    pinned = work[is_pfas]
    rest = work[~is_pfas].sort_values(
        ["_n", "coverage_P_source_given_target"], ascending=[False, False]
    ).head(max(0, max_targets - len(pinned)))
    return pd.concat([pinned, rest]).sort_values(
        ["_n", "coverage_P_source_given_target"], ascending=[False, False]
    )


# Radial factors for the staggered double ring, and the node count above which
# staggering switches on. A single ring of 27 nodes puts neighbour centres only
# 0.23*R apart, which is what forced the discs to stay small; alternating two
# radii moves same-ring neighbours to twice the angular pitch and pushes the
# closest approach to ~0.36*R, so each disc can be ~1.5x wider before touching.
STAGGER_MIN_K = 12
RING_INNER = 0.70
RING_OUTER = 1.00

# Hard ceiling on a ring disc, as a fraction of R. The pairwise neighbour cap is
# the binding constraint at the ring sizes this figure actually draws (25-27
# targets); this only stops a very small ring from inflating its few discs into
# balloons that swamp the hub. Set just above the staggered geometry's own limit
# (~0.156*R at k=26) so it never becomes the active constraint by accident.
NODE_R_MAX_FRAC = 0.16


def _star_positions(targets: list[str], nodes_df: pd.DataFrame,
                    inner: float = RING_INNER,
                    start: float = -math.pi / 2) -> dict:
    """Targets around a single hub, ordered by detection count.

    A single hub with its own ring means every spoke is a clean radius with no
    crossings -- the reason the combined two-hub panel looked tangled is that 35
    of 44 targets received a spoke from both hubs.

    Returns each target's centre as a vector in units of the ring radius R, so a
    length of 1.0 sits on the outer ring. Above STAGGER_MIN_K targets the ring
    alternates between RING_INNER and RING_OUTER: the spokes stay radial (still
    no crossings) but neighbouring discs are no longer competing for the same arc
    length, which is what allows the nodes and their in-disc labels to grow.
    """
    info = nodes_df.set_index("node")
    ordered = sorted(targets, key=lambda n: (-int(info.loc[n, "n_detected"]), n))
    k = len(ordered)
    stagger = k >= STAGGER_MIN_K
    # An odd node count would wrap the alternation onto itself -- slot k-1 and
    # slot 0 would both land on the outer ring one pitch apart, recreating the
    # single-ring spacing at exactly one pair and dragging the whole ring's cap
    # down to it. Allocating an even number of slots and leaving the last one
    # empty keeps the alternation consistent all the way around; the empty slot
    # reads as a small break in the ring, not as a missing node.
    slots = k if (not stagger or k % 2 == 0) else k + 1
    pos = {}
    for i, n in enumerate(ordered):
        ang = start + 2 * math.pi * i / max(1, slots)
        # Highest-detection targets land on the outer ring (even i), so the nodes
        # carrying the most weight get the most room and read first.
        rf = RING_OUTER if (not stagger or i % 2 == 0) else inner
        pos[n] = (math.cos(ang) * rf, math.sin(ang) * rf)
    return pos


def _neighbour_caps(pos: dict, R: float, min_gap: float,
                    hub_clear: float, max_r: float) -> dict:
    """Largest disc radius each node can take without touching a neighbour.

    Half the distance to the nearest other centre, less half the required gap,
    guarantees two adjacent discs never overlap no matter which one is larger.
    Deriving this from the realised positions rather than from a fixed fraction
    of R means it stays correct for the staggered ring, for odd node counts
    (where the alternation wraps two same-radius nodes together) and for small
    rings that are not staggered at all.
    """
    caps = {}
    for name, (vx, vy) in pos.items():
        x, y = vx * R, vy * R
        best = max_r * 2 + min_gap
        for other, (ox, oy) in pos.items():
            if other == name:
                continue
            best = min(best, math.hypot(x - ox * R, y - oy * R))
        # also keep clear of the hub disc at the centre
        radial = math.hypot(x, y) - hub_clear
        caps[name] = max(1.0, min((best - min_gap) / 2, radial - min_gap, max_r))
    return caps


HUB_R_FACTOR = 2.45
NODE_MIN_GAP = 7.0
NODE_FILL_FRAC = 0.86

# Spoke curvature, tried in descending order. A quadratic bezier whose control
# point is offset by rad*chord bows about rad/2*chord away from the chord, so on a
# long outer spoke rad=0.22 swings ~19 units sideways -- straight into the
# staggered inner-ring node that sits one slot over. The panel takes the largest
# value from this ladder that keeps *every* spoke clear of every disc it is not
# aimed at, so curvature is spent only where the geometry has room for it.
SPOKE_RAD_STEPS = (0.22, 0.19, 0.16, 0.13, 0.10, 0.08, 0.06, 0.04, 0.02, 0.0)
# Required gap between a spoke's stroke edge and any disc it passes, in units.
SPOKE_CLEAR = 5.0

# Candidate inner-ring radii, tried in this order. Pushing the inner ring outward
# moves the staggered node out of the arc swept by its outer neighbour's spoke,
# which is what buys back curvature; the cost is that the two rings converge, so
# the pairwise neighbour cap shrinks the discs. The order below therefore walks
# outward only as far as needed: the first candidate that supports at least
# SPOKE_RAD_MIN curvature wins, so node size is given up only in exchange for
# actually-visible edges.
RING_INNER_STEPS = (0.70, 0.74, 0.78, 0.82)
SPOKE_RAD_MIN = 0.13


def _bezier_points(p0, p1, rad: float, steps: int = 64):
    """Sample the same quadratic arc draw_arrow renders, for clearance testing."""
    x0, y0 = p0
    x1, y1 = p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * rad, my + dx * rad
    out = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        out.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                    u * u * y0 + 2 * u * t * cy + t * t * y1))
    return out


def _panel_spoke_rad(shafts: dict, discs: dict, scale: int,
                     half_stroke: float = 0.0) -> float:
    """Largest curvature all spokes in a panel can share without hiding behind a disc.

    One value for the whole panel rather than one per spoke: the star's readability
    depends on every arc sweeping the same way by the same amount, so a mixed set
    of curvatures would trade one artefact for a worse one. `shafts` maps a target
    name to its (start, end) points; `discs` maps every node name to (centre,
    radius). A spoke is allowed to enter its own target's disc, and every spoke
    necessarily starts inside the hub, so both are skipped for that spoke.
    """
    need = SPOKE_CLEAR * scale + half_stroke
    for rad in SPOKE_RAD_STEPS:
        ok = True
        for name, (p0, p1) in shafts.items():
            for px, py in _bezier_points(p0, p1, rad):
                for other, ((ox, oy), orad) in discs.items():
                    if other == name:
                        continue
                    if math.hypot(px - ox, py - oy) - orad < need:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return rad
    return 0.0


def _ring_radii(d: ImageDraw.ImageDraw, hub: str, targets: list[str], pos: dict,
                nodes_df: pd.DataFrame, R: float, scale: int):
    """Disc radius for every ring node, plus the hub's, and the fitted labels.

    Single source of truth for node sizing: the panel renderer and the geometry
    self-check both call this, so a spacing check can never validate different
    numbers than the ones that get drawn.

    Ring discs hold their own label, so each radius has to clear its text.
    Detection count drives the growth term and the fitted label sets the floor;
    the per-node neighbour cap sets the ceiling. That cap used to be a flat
    0.085*R, which bound ~25% below what the ring geometry actually allowed and
    so held every disc (and every in-disc label) small. It is now measured from
    the realised layout, letting each disc expand until it is one gap away from
    its nearest neighbour -- "as large as possible without crowding", enforced
    pairwise rather than by a global guess.
    """
    info = nodes_df.set_index("node")
    size_ref = max(1, int(info["n_detected"].max()))

    def base_r(name: str) -> float:
        """Unconstrained disc radius, growing with sqrt(detection count)."""
        return 17 * scale + 20 * scale * math.sqrt(info.loc[name, "n_detected"] / size_ref)

    # Hub first: the ring caps have to clear the real hub disc, not an estimate.
    # The factor is generous because the annulus between the hub and the inner
    # ring is otherwise dead space -- the arrows need only a fraction of it -- and
    # the hub should stay unambiguously the largest disc now that the ring discs
    # have grown by ~70%.
    hub_r = base_r(hub) * HUB_R_FACTOR
    caps = _neighbour_caps(pos, R, min_gap=NODE_MIN_GAP * scale, hub_clear=hub_r,
                           max_r=NODE_R_MAX_FRAC * R)
    fitted = {n: fit_node_label(d, n, scale, caps[n]) for n in targets}
    radii = {}
    for name in targets:
        # grow to the cap: the larger of the detection-driven radius and the
        # label's required radius, then let it fill the space that is free
        want = max(base_r(name), fitted[name][3])
        radii[name] = min(max(want, caps[name] * NODE_FILL_FRAC), caps[name])
    return radii, fitted, hub_r


def _spoke_endpoints(name: str, pos: dict, radii: dict, R: float, hub_r: float,
                     scale: int, is_context: bool) -> tuple:
    """Start/end point of a spoke's visible shaft, plus its arrowhead length.

    Shared by the renderer and the layout solver so the arcs that get tested for
    occlusion are the arcs that get drawn.
    """
    vx, vy = pos[name]
    # pos vectors are scaled by the ring factor, so re-normalise before using them
    # as a direction: on the inner ring |v| is < 1, and treating that as a unit
    # vector would pull both arrow ends short of their discs.
    mag = math.hypot(vx, vy) or 1.0
    ux, uy = vx / mag, vy / mag
    tr = radii[name]
    # Arrowheads are sized from the target disc, which is now much larger, so they
    # are additionally capped at a share of the *visible* shaft. Inner-ring spokes
    # are the short ones and an uncapped head would consume the whole shaft and
    # read as a wedge rather than an arrow.
    shaft = max(1.0, mag * R - hub_r - tr)
    head_cap = 0.42 * shaft
    if is_context:
        head = min(max(tr * 0.42, 8 * scale), head_cap)
    else:
        head = min(max(tr * 0.52, 11 * scale), head_cap)
    p0 = (ux * hub_r * 1.03, uy * hub_r * 1.03)
    p1 = (vx * R - ux * (tr + head * 0.58), vy * R - uy * (tr + head * 0.58))
    return p0, p1, head


def _solve_layout(d: ImageDraw.ImageDraw, hub: str, targets: list[str],
                  nodes_df: pd.DataFrame, R: float, scale: int,
                  ctx_names: set) -> tuple:
    """Pick the inner-ring radius and the shared spoke curvature together.

    These two cannot be chosen independently. A node sitting one slot over from
    its neighbour on a tighter inner ring falls exactly inside the arc that
    neighbour's spoke sweeps, so the edge vanishes behind the disc; pushing the
    inner ring outward frees the arc but converges the two rings and shrinks the
    discs. Solving jointly keeps the star's sweep *and* keeps every edge visible,
    spending node size only when it is the only way to get both.

    Returns (pos, radii, fitted, hub_r, rad, inner).
    """
    widest_stroke = max(1, int(round((0.9 + 2.0) * scale)))
    best = None
    for inner in RING_INNER_STEPS:
        pos = _star_positions(targets, nodes_df, inner=inner)
        radii, fitted, hub_r = _ring_radii(d, hub, targets, pos, nodes_df, R, scale)
        shafts, discs = {}, {}
        for n in targets:
            p0, p1, _ = _spoke_endpoints(n, pos, radii, R, hub_r, scale,
                                         n in ctx_names)
            shafts[n] = (p0, p1)
            discs[n] = ((pos[n][0] * R, pos[n][1] * R), radii[n])
        rad = _panel_spoke_rad(shafts, discs, scale,
                               half_stroke=widest_stroke / 2)
        if best is None or rad > best[4]:
            best = (pos, radii, fitted, hub_r, rad, inner)
        if rad >= SPOKE_RAD_MIN:
            return pos, radii, fitted, hub_r, rad, inner
    # nothing reached the target sweep: keep whichever candidate curved the most
    return best


def _draw_star_panel(d: ImageDraw.ImageDraw, hub: str, spokes: pd.DataFrame,
                     nodes_df: pd.DataFrame, cx: float, cy: float, R: float,
                     scale: int, coverage_min: float, fonts: dict,
                     context_spokes: pd.DataFrame | None = None) -> None:
    """Radial star: hub at the centre, main targets on the ring.

    `context_spokes` holds PFAS-cluster edges that pass a softer gate than
    the main statistical filter; they are drawn as thin gray arrows so the
    inter-PFAS relationships are always visible regardless of bootstrap CI.
    """
    info = nodes_df.set_index("node")
    # merge context targets into the ring so they get a position
    all_spokes = pd.concat([spokes, context_spokes]) if context_spokes is not None else spokes
    all_spokes = all_spokes.drop_duplicates("target")
    targets = list(all_spokes["target"])
    cov = all_spokes.set_index("target")["coverage_P_source_given_target"].to_dict()
    asym = all_spokes.set_index("target")["asymmetry_index_A"].to_dict()
    enr = all_spokes.set_index("target")["is_enriched_directed_edge"].to_dict()
    is_ctx = set() if context_spokes is None else set(context_spokes["target"])

    # Ring placement and spoke curvature are solved together -- see _solve_layout.
    pos, radii, fitted, hub_r, rad, _inner = _solve_layout(
        d, hub, targets, nodes_df, R, scale, is_ctx)

    def node_r(name: str) -> float:
        """Final (collision-capped) radius of a ring node."""
        return radii[name]

    # Spoke endpoints, in panel coordinates. Resolved before anything is drawn so
    # the arcs tested by the solver are exactly the arcs rendered here.
    geom = {}
    for name in targets:
        p0, p1, head = _spoke_endpoints(name, pos, radii, R, hub_r, scale,
                                        name in is_ctx)
        geom[name] = ((cx + p0[0], cy + p0[1]), (cx + p1[0], cy + p1[1]), head)

    # ── spokes ──────────────────────────────────────────────────────────────
    # All spokes curve in the same rotational direction (positive rad), giving
    # the star a consistent clockwise sweep — the "designed" quality rather
    # than a flat bicycle wheel. Context spokes are drawn first (background).
    for name in (list(is_ctx) + [t for t in pos if t not in is_ctx]):
        if name not in pos:
            continue
        p0, p1, head = geom[name]
        w = float(cov.get(name, coverage_min))
        if name in is_ctx:
            color = hex_rgb(BASELINE_INK)
            lw = max(1, int(0.7 * scale))
        else:
            norm = max(0.0, (w - coverage_min) / max(1e-9, 1 - coverage_min))
            lw = max(1, int(round((0.9 + 2.0 * norm) * scale)))
            color = hex_rgb(a_color(float(asym.get(name, 0.0))))
        draw_arrow(d, p0, p1, color, lw, rad=rad, head=int(head))

    # ── nodes ────────────────────────────────────────────────────────────────
    for name, (vx, vy) in pos.items():
        tx, ty = cx + vx * R, cy + vy * R
        tr = node_r(name)
        is_context = name in is_ctx
        node_color = category_color(info.loc[name, "category"])
        # Context nodes stay filled -- a hollow ring would drop the in-disc label
        # onto the surface, where several of the fills sit below 3:1.  They
        # recede through a dark outline plus a thinner arrow instead.
        d.ellipse([tx - tr - 1.8 * scale, ty - tr - 1.8 * scale,
                   tx + tr + 1.8 * scale, ty + tr + 1.8 * scale],
                  fill=hex_rgb(SURFACE))
        d.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], fill=hex_rgb(node_color))
        if is_context:
            d.ellipse([tx - tr, ty - tr, tx + tr, ty + tr],
                      outline=hex_rgb(BASELINE_INK), width=max(2, int(0.9 * scale)))
        elif enr.get(name):
            d.ellipse([tx - tr, ty - tr, tx + tr, ty + tr],
                      outline=hex_rgb(INK_PRIMARY), width=max(1, int(0.7 * scale)))
        # label inside the disc, like the hub -- with the coverage value under it
        lines, f_lab, f_val, _ = fitted[name]
        ink = label_ink(node_color)
        lh = max(text_size(d, ln, f_lab)[1] for ln in lines)
        val = f"{cov.get(name, 0.0):.2f}"
        vw, vh = text_size(d, val, f_val)
        block = len(lines) * lh + (len(lines) - 1) * 2 * scale + vh + 3 * scale
        yy = ty - block / 2
        for ln in lines:
            lw, _ = text_size(d, ln, f_lab)
            d.text((tx - lw / 2, yy), ln, fill=hex_rgb(ink), font=f_lab)
            yy += lh + 2 * scale
        d.text((tx - vw / 2, yy + 1 * scale), val, fill=hex_rgb(ink), font=f_val)

    # ── hub (drawn last so it sits on top) ───────────────────────────────────
    hub_fill = category_color(info.loc[hub, "category"])
    hub_ink = label_ink(hub_fill)
    hcol = hex_rgb(hub_fill)
    d.ellipse([cx - hub_r - 2.6 * scale, cy - hub_r - 2.6 * scale,
               cx + hub_r + 2.6 * scale, cy + hub_r + 2.6 * scale],
              fill=hex_rgb(SURFACE))
    d.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r], fill=hcol)
    # The hub disc grew with the ring, so its label is sized to the disc rather
    # than held at one point size -- a fixed size would leave the largest disc in
    # the figure carrying visibly undersized type. 1.55*hub_r is the usable chord
    # at the height where the name sits.
    f_hub = fonts["hub"]
    f_hub_sub = fonts["small"]
    for pt in HUB_FONT_STEPS:
        trial = font(pt * scale, bold=True)
        if text_size(d, hub, trial)[0] <= 1.55 * hub_r:
            f_hub = trial
            f_hub_sub = font(pt * 0.62 * scale)
            break
    tw, th = text_size(d, hub, f_hub)
    sub = f"n = {len(targets)}"
    tw2, th2 = text_size(d, sub, f_hub_sub)
    gap = 3 * scale
    block = th + gap + th2
    d.text((cx - tw / 2, cy - block / 2), hub, fill=hex_rgb(hub_ink), font=f_hub)
    d.text((cx - tw2 / 2, cy - block / 2 + th + gap), sub,
           fill=hex_rgb(hub_ink), font=f_hub_sub)


def draw_network_figure(edges: pd.DataFrame, nodes_df: pd.DataFrame, png: Path,
                        pdf: Path,
                        edge_flag: str = "is_directed_edge",
                        coverage_min: float = 0.90,
                        focus: tuple[str, ...] = ("PFAS-1", "PFAS-2"),
                        max_targets: int = 25,
                        edges_all: pd.DataFrame | None = None) -> None:
    """One radial star per focus hub, side by side (Nature-style panels a/b).

    No in-figure title or subtitle: the panel letters plus the legend carry the
    figure, and the wording lives in the manuscript caption.

    Parameters
    ----------
    edges:       retained directed edges (strict statistical gate).
    edges_all:   full all-pairs table; used to add inter-PFAS context edges
                 (coverage ≥ 0.85, A > 0, q_coverage < 0.05) that pass a softer
                 gate so long-chain PFAS clusters always appear in the figure.
    max_targets: ring size cap for the strict-gate spokes; PFAS clusters are
                 always pinned in regardless of this cap.
    """
    sel = edges[edges[edge_flag]].copy()
    hubs = resolve_focus(nodes_df, focus)
    detected = nodes_df.set_index("node")["n_detected"]
    scale = SCALE
    R = 310 * scale
    # Tight margins: no figure title or subtitle (the caption carries them), so
    # the header only has to clear the panel letter, and the ring padding only
    # has to clear the widest outer disc.
    # The ring padding has to clear the widest outer disc, which is now capped at
    # 0.145*R plus its surface halo -- so pad is derived from that cap instead of
    # being a constant that the enlarged discs could overrun.
    pad = int(0.145 * R + 22 * scale)
    head_h = 34 * scale
    legend_h = 92 * scale
    panel = 2 * R + 2 * pad
    W = panel * len(hubs)
    H = head_h + panel + legend_h
    img = Image.new("RGB", (W, H), hex_rgb(SURFACE))
    d = VectorDraw(img)

    fonts = {
        "hub": font(20 * scale, bold=True),      # fallback; sized to the disc
        "node": font(11 * scale, bold=True),     # sits inside the disc
        "val": font(9.0 * scale),
        "small": font(13 * scale),               # legend body
        "bold": font(13 * scale, bold=True),
        "panel": font(17 * scale, bold=True),
        "panel_note": font(13 * scale),
    }

    shown = 0
    for i, hub in enumerate(hubs):
        spokes = sel[sel["source"] == hub].copy()
        total_hub = len(spokes)
        if total_hub > max_targets:
            spokes = select_targets(spokes, detected, max_targets)

        # Context edges: other PFAS clusters not in the strict-gate set but with
        # coverage ≥ 0.85, A > 0, q_coverage < 0.05 -- so long-chain PFAS is
        # always visible even when the bootstrap CI barely includes 0
        ctx = None
        if edges_all is not None:
            pfas_ctx = edges_all[
                (edges_all["source"] == hub)
                & (~edges_all["target"].isin(set(spokes["target"])))
                & (edges_all["target"].str.startswith("PFAS"))
                & (edges_all["coverage_P_source_given_target"] >= 0.85)
                & (edges_all["asymmetry_index_A"] > 0)
                & (edges_all["q_coverage"] < 0.05)
            ]
            if not pfas_ctx.empty:
                ctx = pfas_ctx.copy()

        shown += len(spokes) + (len(ctx) if ctx is not None else 0)
        cx = panel * i + panel / 2
        cy = head_h + panel / 2
        px = panel * i + pad * 0.30

        d.text((px, 4 * scale), chr(ord("a") + i),
               fill=hex_rgb(INK_PRIMARY), font=fonts["panel"])
        note = f"{hub}: {total_hub} significant cluster edges"
        d.text((px + 30 * scale, 9 * scale), note,
               fill=hex_rgb(INK_SECONDARY), font=fonts["panel_note"])
        if len(spokes):
            _draw_star_panel(d, hub, spokes, nodes_df, cx, cy, R, scale,
                             coverage_min, fonts, context_spokes=ctx)

    _legend(d, sel, nodes_df, pad * 0.30, head_h + panel + 4 * scale,
            scale, fonts["small"], fonts["bold"], coverage_min, shown, len(sel))
    save(img, png, pdf, d.operations)


def _wrap(d: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    """Greedy word wrap of a caption to max_w pixels."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if text_size(d, trial, fnt)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _legend(d, sel, nodes_df, x0, y0, scale, f_small, f_bold, coverage_min,
            shown: int = 0, total: int = 0) -> None:
    """Category swatches (three rows) plus the width and asymmetry keys.

    Node fills now encode pollutant category, so the legend has to name each one.
    Several fills sit below 3:1 on the surface, so the swatch is paired with its
    label here and every node repeats its name inside the disc.
    """
    drawn = [c for c in dict.fromkeys(nodes_df["category"]) if c in CATEGORY_COLORS]
    rows = 3
    per_row = math.ceil(len(drawn) / rows)
    # Swatch and row pitch follow the legend text metrics rather than fixed units,
    # so enlarging the legend font does not leave undersized dots behind.
    cap_h = text_size(d, "Hg", f_small)[1]
    r = cap_h * 0.62
    row_h = cap_h + 9 * scale
    col_w = max(text_size(d, c, f_small)[0] for c in drawn) + 2 * r + 20 * scale
    for i, cat in enumerate(drawn):
        cxx = x0 + (i // rows) * col_w
        cyy = y0 + 10 * scale + (i % rows) * row_h
        d.ellipse([cxx, cyy, cxx + 2 * r, cyy + 2 * r],
                  fill=hex_rgb(CATEGORY_COLORS[cat]))
        d.text((cxx + 2 * r + 6 * scale, cyy - 1.5 * scale), cat,
               fill=hex_rgb(INK_SECONDARY), font=f_small)

    x2 = x0 + per_row * col_w + 16 * scale
    yy = y0 + 10 * scale
    d.text((x2, yy), "coverage P(S|T)", fill=hex_rgb(INK_PRIMARY), font=f_bold)
    yy += row_h
    for label, w in ((f"{coverage_min:.2f}", coverage_min), ("1.00", 1.0)):
        # same width ramp as the spokes in _draw_star_panel, so the key measures
        # the strokes it is standing in for
        norm = max(0.0, (w - coverage_min) / max(1e-9, 1 - coverage_min))
        lw = max(1, int(round((0.9 + 2.0 * norm) * scale)))
        d.line([(x2, yy + 5 * scale), (x2 + 52 * scale, yy + 5 * scale)],
               fill=hex_rgb(INK_MUTED), width=lw)
        d.text((x2 + 60 * scale, yy), label, fill=hex_rgb(INK_SECONDARY), font=f_small)
        yy += row_h

    x3 = x2 + 150 * scale
    bar_w, bar_h = 170 * scale, 11 * scale
    d.text((x3, y0 + 10 * scale), "asymmetry A", fill=hex_rgb(INK_PRIMARY), font=f_bold)
    by = y0 + 10 * scale + row_h
    for i in range(bar_w):
        col = a_color(i / (bar_w - 1))
        d.line([(x3 + i, by), (x3 + i, by + bar_h)], fill=hex_rgb(col))
    d.rectangle([x3, by, x3 + bar_w, by + bar_h], outline=hex_rgb(BASELINE_INK),
                width=max(1, scale // 2))
    for frac, label in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        tw, _ = text_size(d, label, f_small)
        d.text((x3 + frac * bar_w - tw / 2, by + bar_h + 3 * scale), label,
               fill=hex_rgb(INK_SECONDARY), font=f_small)


def draw_centrality_figure(nodes_df: pd.DataFrame, png: Path, pdf: Path,
                           title: str) -> None:
    """Diverging out/in degree bars plus hub, authority and betweenness panels."""
    d0 = nodes_df.sort_values(
        ["net_degree_out_minus_in", "out_degree", "n_detected"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    n = len(d0)
    scale = SCALE
    # Row pitch and the label gutter grow with the enlarged row type: at 11pt a
    # 16u row would clip the ascenders of the node names, so the pitch is set from
    # the font's own height with a fixed leading on top.
    f_title = font(19 * scale, bold=True)
    f_head = font(12 * scale, bold=True)
    f_lab = font(11 * scale)
    f_val = font(10 * scale)

    row_h = (24 if n > 20 else 32) * scale
    label_w = 190 * scale
    panel_w = 235 * scale
    gap = 32 * scale
    pad_l, pad_t = 26 * scale, 108 * scale
    pad_b = 78 * scale
    W = pad_l + label_w + panel_w * 4 + gap * 3 + 12 * scale
    H = pad_t + n * row_h + pad_b
    img = Image.new("RGB", (W, H), hex_rgb(SURFACE))
    d = VectorDraw(img)

    d.text((pad_l, 28 * scale), title, fill=hex_rgb(INK_PRIMARY), font=f_title)
    d.text((pad_l, 60 * scale),
           "Bars colored by node group. Out-degree = how many nodes this node covers; "
           "in-degree = how many cover it.",
           fill=hex_rgb(INK_SECONDARY), font=f_lab)

    panels = [
        ("Directed degree", None),
        ("Hub score (coverer)", "hub_score"),
        ("Authority score (covered)", "authority_score"),
        ("Betweenness", "betweenness"),
    ]
    x_starts = [pad_l + label_w + i * (panel_w + gap) for i in range(4)]

    for (head, col), x0 in zip(panels, x_starts):
        d.text((x0, pad_t - 30 * scale), head, fill=hex_rgb(INK_PRIMARY), font=f_head)

    # Truncate the row label by measured width, not by a character count: the old
    # 24-char rule was tuned to 7pt type in a narrower gutter and would either
    # overflow or cut short now that the type is larger.
    name_max_w = label_w - 12 * scale

    def fit_name(raw: str) -> str:
        """Truncate a row label (with a trailing '.') to the gutter width."""
        if text_size(d, raw, f_lab)[0] <= name_max_w:
            return raw
        cut = raw
        while cut and text_size(d, cut + ".", f_lab)[0] > name_max_w:
            cut = cut[:-1]
        return cut + "."

    max_deg = max(1, int(max(d0["out_degree"].max(), d0["in_degree"].max())))
    for i, r in d0.iterrows():
        cy = pad_t + i * row_h + row_h / 2
        color = hex_rgb(category_color(r["category"]))
        name = fit_name(str(r["node"]))
        tw, th = text_size(d, name, f_lab)
        d.text((pad_l + label_w - tw - 8 * scale, cy - th / 2), name,
               fill=hex_rgb(INK_PRIMARY), font=f_lab)

        # panel 1: diverging degree
        x0 = x_starts[0]
        mid = x0 + panel_w / 2
        half = panel_w / 2 - 16 * scale
        bh = row_h * 0.46
        d.line([(mid, cy - row_h / 2), (mid, cy + row_h / 2)],
               fill=hex_rgb(GRIDLINE), width=max(1, scale // 2))
        # Bar-end numerals are vertically centred from their own measured height,
        # so they stay on the bar's axis at any font size.
        def val_y(text: str) -> float:
            """Top y that vertically centres a numeral on the current row."""
            return cy - text_size(d, text, f_val)[1] / 2

        out_len = half * r["out_degree"] / max_deg
        in_len = half * r["in_degree"] / max_deg
        if out_len > 0:
            d.rectangle([mid, cy - bh / 2, mid + out_len, cy + bh / 2], fill=color)
            lab = str(int(r["out_degree"]))
            d.text((mid + out_len + 5 * scale, val_y(lab)), lab,
                   fill=hex_rgb(INK_SECONDARY), font=f_val)
        if in_len > 0:
            d.rectangle([mid - in_len, cy - bh / 2, mid, cy + bh / 2],
                        fill=blend(category_color(r["category"]), SURFACE, 0.42))
            lab = str(int(r["in_degree"]))
            tw2, _ = text_size(d, lab, f_val)
            d.text((mid - in_len - tw2 - 5 * scale, val_y(lab)), lab,
                   fill=hex_rgb(INK_SECONDARY), font=f_val)

        # panels 2-4: magnitude bars
        for (head, col), xs in zip(panels[1:], x_starts[1:]):
            top = float(d0[col].max())
            # reserve room for the numeral that sits past the bar end
            val_w = text_size(d, "0.000", f_val)[0] + 12 * scale
            length = (panel_w - val_w) * (float(r[col]) / top) if top > 0 else 0.0
            if length > 0:
                d.rectangle([xs, cy - bh / 2, xs + length, cy + bh / 2], fill=color)
            if top > 0:
                lab = f"{float(r[col]):.3f}"
                d.text((xs + length + 5 * scale, val_y(lab)), lab,
                       fill=hex_rgb(INK_SECONDARY), font=f_val)

    for (head, col), x0 in zip(panels[1:], x_starts[1:]):
        if float(d0[col].max()) <= 0:
            d.text((x0, pad_t + n * row_h / 2),
                   "all zero — every source node points\n"
                   "directly at all its targets, so no\n"
                   "2-hop shortest path exists",
                   fill=hex_rgb(INK_MUTED), font=f_lab, spacing=5 * scale)

    y_axis = pad_t + n * row_h + 14 * scale
    mid = x_starts[0] + panel_w / 2
    # right-align the in-degree caption on the centre line by measuring it, rather
    # than by a fixed offset that only held at the old font size
    in_cap = "in-degree ←"
    d.text((mid - text_size(d, in_cap, f_lab)[0] - 10 * scale, y_axis), in_cap,
           fill=hex_rgb(INK_MUTED), font=f_lab)
    d.text((mid + 12 * scale, y_axis), "→ out-degree", fill=hex_rgb(INK_MUTED), font=f_lab)

    cap_h = text_size(d, "Hg", f_lab)[1]
    ly = y_axis + cap_h + 16 * scale
    lx = pad_l
    sw = cap_h  # square swatch matched to the cap height
    for cat in dict.fromkeys(d0["category"]):
        if cat not in CATEGORY_COLORS:
            continue
        d.rectangle([lx, ly, lx + sw, ly + sw], fill=hex_rgb(CATEGORY_COLORS[cat]))
        d.text((lx + sw + 7 * scale, ly - 1 * scale), cat,
               fill=hex_rgb(INK_SECONDARY), font=f_lab)
        lx += sw + text_size(d, cat, f_lab)[0] + 30 * scale
    save(img, png, pdf, d.operations)
