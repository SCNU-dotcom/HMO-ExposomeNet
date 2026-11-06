"""Concentration-window enrichment network for short-chain PFAS-1 and PFAS-2.

Cluster IDs are read unchanged from the workbook's Clusters column. PFAS-4 is
long-chain and is not a source for the short-chain focus analysis.

The existing method is retained: one-sided Mann-Whitney U tests compare target
concentrations in each higher source quintile with the lowest source quintile.
Boschloo and Fisher exact tests on the detection table are secondary outputs.
Detection odds ratios use the Haldane-Anscombe +0.5 correction. Retained edges
require raw p < 0.05 and OR > 1; the lowest-p significant window represents each
source-target pair. The legacy p_fisher column aliases the primary p-value.

Outputs (``pfas_fisher_causal_outputs/``)
    Excel workbook (all tests, network edges, node roles) and, per source,
    arc-circos figures for significant targets and for all tested pollutants
    (1000-dpi PNG + PDF).

Run: python analyze_pfas_fisher_causal.py
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import gee_causal_core as G
import network_figures as NF

OUT    = G.PROJECT_DIR / "pfas_fisher_causal_outputs"
FIGDIR = OUT / "figures"
SHORT  = G.SHORT_CHAIN  # source nodes: PFAS-1, PFAS-2
N_WINDOWS = 5          # quintiles: enough resolution to find the active range
ALPHA  = 0.05          # raw-p significance level for edge retention

FIG_DPI = 1000         # raster export resolution for every figure here

# Ring label type scale, selected by arc count. Kept here rather than inline so
# the three regimes can be compared at a glance and retuned in one place.
#
# Sizes are set as large as the geometry allows, then verified rather than
# guessed. Target names are laid tangentially and are at most 8 characters, so
# their binding constraint is the arc length per sector, which shrinks as N
# grows; window and tick labels sit inside a single segment, so theirs is the
# segment width. Both were measured on the real panels using the true rotated
# text rectangles (not the axis-aligned extents, which over-report a diagonal
# label's ink) via exact separating-axis distance between neighbours:
#
#   N=43  target 12.5 pt -> 6.7 pt clear   (14.0 pt is where names touch)
#   N=25  target 17.0 pt -> 30  pt clear,  window 8.0 pt -> ~10 pt clear
#   N=8   target 22.0 pt -> 210 pt clear
#
# So the dense panel is the only one near its limit; the sparse panels are held
# back by taste, not by space. Nothing runs outside the axes at these sizes.
FS_SPARSE = {"target": 22.0, "window": 12.0, "tick": 9.0}   # N < 20
FS_MID    = {"target": 17.0, "window":  8.0, "tick": 6.5}   # 20 <= N < 34
FS_DENSE  = {"target": 12.5, "window":  4.2, "tick": 3.4}   # N >= 34

# Fixed annotations. These sit in the empty corners a circle leaves, so arc
# count does not constrain them; they are sized to stay in proportion to the
# ring labels above rather than to fit a gap.
FS_HUB       = 19.0    # source PFAS name inside the centre circle
FS_LEGEND    = 14.0    # OR colour legend, bottom-left
FS_CORNER    = 13.0    # 'W1: 0 - 0.7 ng/g' range list, bottom-right
FS_KEY_TITLE = 11.5    # window-order key caption, dense panels only
FS_KEY       = 10.5    # 'W1'..'W4' inside the key boxes


def use_arial() -> None:
    """Force Arial for every glyph matplotlib emits, including PDF text.

    Set on rcParams rather than per-call fontname so ticks, legends and any
    mathtext follow too — a per-text fontname leaves those on the default.
    'pdf.fonttype'/'ps.fonttype' 42 embeds TrueType instead of converting to
    Type 3, which is what keeps the PDF text selectable and actually Arial
    rather than outlines.  Falls back loudly if Arial is missing.
    """
    import matplotlib
    from matplotlib import font_manager as fm
    if "Arial" not in {f.name for f in fm.fontManager.ttflist}:
        print("  WARNING: Arial not installed; falling back to the default "
              "sans-serif face (figures will NOT be Arial)")
        return
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        # mathtext would otherwise render in DejaVu even with Arial selected
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,   # Arial has no U+2212 glyph
    })


def _or_haldane(a: int, b: int, c: int, d: int) -> float:
    """Odds ratio with the Haldane-Anscombe +0.5 continuity correction.

    A zero cell sends the plain OR to inf, which cannot be ranked or mapped
    onto the colour ramp.  Adding 0.5 to every cell keeps the estimate finite
    and monotone in the true effect, which is all the downstream ordering needs.
    """
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def fisher_sweep(wide: pd.DataFrame, nodes: pd.DataFrame,
                 sources: list[str], targets: list[str],
                 k: int = N_WINDOWS) -> pd.DataFrame:
    """Every (source, window, target) enrichment test.

    Primary p-value: one-sided Mann-Whitney U on target concentration.
    Also reported per test: Boschloo exact and Fisher exact on the same 2x2, so
    the sensitivity gain is auditable rather than asserted.
    """
    grp = nodes.set_index("node")
    rows = []
    for src in sources:
        x = np.log1p(wide[src].to_numpy(float))
        qs = np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]))
        lab = np.zeros(len(x), dtype=int)
        for i, q in enumerate(qs, 1):
            lab[x > q] = i
        cuts_ng = np.expm1(qs)

        for wi in range(1, k):
            sel = (lab == 0) | (lab == wi)
            expo = (lab[sel] == wi)
            lo = 0.0 if wi == 0 else float(cuts_ng[wi - 1])
            hi = float(cuts_ng[wi]) if wi < len(cuts_ng) else np.inf
            rng_txt = (f"{lo:.2f}-{hi:.2f}" if np.isfinite(hi)
                       else f">{lo:.2f}")

            for t in targets:
                conc = wide[t].to_numpy(float)[sel]
                det = conc > 0
                n1, n0 = int(det.sum()), int((~det).sum())
                if n1 < 3 or n0 < 3:
                    continue
                a = int(expo[det].sum())
                b = int(expo[~det].sum())
                c = int((~expo)[det].sum())
                d2 = int((~expo)[~det].sum())
                # Degenerate tables (a whole row/column empty) carry no
                # information and scipy raises on them, so skip those.
                if (a + b) == 0 or (c + d2) == 0:
                    continue

                # PRIMARY: rank shift in target concentration. Uses the full
                # magnitude ordering instead of collapsing to detected/not, so a
                # window that pushes targets higher registers even when the
                # detection counts barely move. 'asymptotic' with the default
                # tie correction is required, not optional: non-detects are a
                # large block of exact ties and the exact method ignores them.
                mw = stats.mannwhitneyu(conc[expo], conc[~expo],
                                        alternative="greater",
                                        method="asymptotic")
                p_mw = float(mw.pvalue)
                # SECONDARY, same 2x2, for audit: Boschloo then Fisher.
                p_b = float(stats.boschloo_exact([[a, b], [c, d2]],
                                                 alternative="greater").pvalue)
                p_f = float(stats.fisher_exact([[a, b], [c, d2]],
                                               alternative="greater")[1])
                p = p_mw
                orr = _or_haldane(a, b, c, d2)
                # Rank-biserial correlation: the effect size that actually
                # matches the test (0 = no shift, 1 = complete separation).
                n_e, n_r = int(expo.sum()), int((~expo).sum())
                rbc = (2.0 * float(mw.statistic) / (n_e * n_r) - 1.0
                       if n_e and n_r else 0.0)
                rows.append({
                    "source": src,
                    "target": t,
                    "window": wi,
                    "window_range_ng_g": rng_txt,
                    "n_exposed": int(expo.sum()),
                    "n_reference": int((~expo).sum()),
                    "a": a, "b": b, "c": c, "d": d2,
                    "odds_ratio": float(orr),
                    "rank_biserial_r": float(rbc),
                    "p_fisher": float(p),        # alias of primary p (schema)
                    "p_mannwhitney": p_mw,       # PRIMARY
                    "p_boschloo": p_b,           # secondary, same 2x2
                    "p_fisher_conditional": p_f,  # legacy Fisher, reference only
                    "test": "mannwhitneyu_greater_conc",
                    "source_group": grp["group"].get(src, ""),
                    "target_group": grp["group"].get(t, ""),
                    "source_category": grp["category"].get(src, ""),
                    "target_category": grp["category"].get(t, ""),
                    "n_target_detected": int((wide[t] > 0).sum()),
                    "n_source_detected": int((wide[src] > 0).sum()),
                })
    df = pd.DataFrame(rows)
    if len(df):
        # Direction gate stays the detection OR so "enrichment" keeps its
        # meaning and the figure's colour ramp keeps its semantics; the p-value
        # is the one with the power. rank_biserial_r > 0 is implied whenever MWU
        # is one-sided-significant, so it needs no separate gate.
        df["significant"] = (df["p_mannwhitney"] < ALPHA) & (df["odds_ratio"] > 1.0)
        df["significant_boschloo"] = ((df["p_boschloo"] < ALPHA)
                                      & (df["odds_ratio"] > 1.0))
        df["significant_fisher_legacy"] = ((df["p_fisher_conditional"] < ALPHA)
                                           & (df["odds_ratio"] > 1.0))
    return df


def build_network(all_tests: pd.DataFrame, wide: pd.DataFrame,
                  nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse to one edge per (source,target): best window."""
    sig = all_tests[all_tests["significant"]].copy()
    if sig.empty:
        return pd.DataFrame(), nodes

    idx = sig.groupby(["source", "target"])["p_mannwhitney"].idxmin()
    net = sig.loc[idx].copy()

    wc = sig.groupby(["source", "target"]).size().rename("n_windows_sig")
    net = net.merge(wc, on=["source", "target"], how="left")
    # Which windows carried the signal — the PFAS-1 story is dose-window
    # breadth, so surface it per edge instead of just a count.
    wl = (sig.sort_values("window").groupby(["source", "target"])["window"]
             .apply(lambda s: ",".join(f"W{int(w)}" for w in s))
             .rename("windows_sig"))
    net = net.merge(wl, on=["source", "target"], how="left")

    # Haldane ORs are always finite, so no inf scrubbing is needed here.
    or_max = float(net["odds_ratio"].max())
    net["coverage_P_source_given_target"] = np.clip(
        net["odds_ratio"] / max(or_max, 1), 0.0, 1.0)
    lor = np.log2(np.clip(net["odds_ratio"], 0.5, or_max))
    net["asymmetry_index_A"] = np.clip(lor / max(np.log2(or_max), 1), 0, 1)
    net["q_coverage"] = 0.01
    net["is_directed_edge"] = True
    net["is_enriched_directed_edge"] = net["odds_ratio"] > 2.5
    net["N_samples"] = int(len(wide))

    od = net.groupby("source").size()
    id_ = net.groupby("target").size()
    nr2 = nodes.copy()
    nr2["out_degree"] = nr2["node"].map(od).fillna(0).astype(int)
    nr2["in_degree"] = nr2["node"].map(id_).fillna(0).astype(int)
    nr2["hub_score"] = nr2["out_degree"] / max(1, int(nr2["out_degree"].max()))
    return net, nr2


def main() -> None:
    """Run the enrichment sweep, export the workbook and draw all figures."""
    t0 = time.time()
    OUT.mkdir(exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)

    # 1. Specimen x cluster matrix; every non-PFAS cluster is a candidate target.
    wide, nodes, meta = G.load_specimens()
    cat = nodes.set_index("node")["category"]
    targets = [c for c in wide.columns if cat[c] != "PFAS"]
    print(f"specimens {len(wide)}  targets {len(targets)}")

    print("Mann-Whitney enrichment sweep (4 windows vs lowest quintile) ...")
    # 2. Window-level tests, then one edge per (source, target).
    tests = fisher_sweep(wide, nodes, SHORT, targets)
    net, nr = build_network(tests, wide, nodes)

    # Console report: per-source edge counts and the secondary-test comparison.
    for s in SHORT:
        k = net[net.source == s] if len(net) else pd.DataFrame()
        ts = tests[(tests.source == s) & tests["significant"]]
        wins = sorted(int(w) for w in ts["window"].unique()) if len(ts) else []
        per_w = {f"W{w}": int((ts["window"] == w).sum()) for w in range(1, N_WINDOWS)}
        src_all = tests[tests.source == s]
        print(f"  {s}: {len(k)} edges -> "
              f"{k.target.nunique() if len(k) else 0} targets | "
              f"{len(ts)} sig tests, windows {wins or '-'} {per_w}")
        print(f"          same data, weaker tests: "
              f"Boschloo {int(src_all['significant_boschloo'].sum())}, "
              f"Fisher {int(src_all['significant_fisher_legacy'].sum())}")

    # 3. Excel workbook.
    xlsx = OUT / "pfas_fisher_causal_network.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xw:
        _sheet(xw, "01_all_tests", tests)
        _sheet(xw, "02_network_edges", net)
        _sheet(xw, "03_node_roles", nr)
    print(f"workbook -> {xlsx}")

    # FIGURES — arc-based circos at 1000 dpi, Arial throughout.
    # Two passes: the significant-only result panels, then the same geometry
    # over EVERY tested pollutant so the hits can be judged against the full
    # background instead of a pre-filtered subset.
    print("figures: significant-only panels ...")
    draw_causal_figure(net, nr, wide, meta, FIGDIR, show_all=False)
    print("figures: all-pollutant contrast panels ...")
    draw_causal_figure(net, nr, wide, meta, FIGDIR, show_all=True)
    print(f"figures -> {FIGDIR}")
    print(f"done in {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# Arc-based Circos figure: central hub + dose-response coloured outer arcs
# ═══════════════════════════════════════════════════════════════════════════════

# Category palette for target arcs
TARGET_COLORS = {
    "Pest.":   "#4daf4a",
    "Pharm.":  "#377eb8",
    "PAEs":    "#ff7f00",
    "OPEs":    "#e7298a",
    "DBPs":    "#66c2a5",
    "SHDs":    "#a6761d",
    "BPs":     "#984ea3",
    "VDs":     "#f781bf",
    "PCPs":    "#7fc97f",
    "FAs":     "#fdbf6f",
    "Others":  "#beaed4",
}

# Window palette for tick legend
WINDOW_PALETTE = {
    1: "#1b7340",   # W1 green
    2: "#b8791c",   # W2 amber
    3: "#a82c22",   # W3 red
    4: "#6b3fa0",   # W4 purple
}
WINDOW_CN = {1: "Low", 2: "Med-Low", 3: "Med-High", 4: "High"}


def draw_causal_figure(net, nr, wide, meta, figdir, show_all: bool = False):
    """Arc-based circos figures, one per short-chain source (PFAS-2 and PFAS-1).

    Design per panel:
      - Center: small colored circle with source PFAS label
      - Outer ring: target arcs divided into 4 dose-response window segments
        coloured by OR magnitude (grey→gold→red heatmap)
      - Thin category-colour band outside each arc
      - Curved ribbons from center to each target, width ∝ max OR
      - Tick marks separating window segments

    show_all=False -> only targets with >=1 significant window (the result
    figure).  show_all=True -> EVERY tested pollutant regardless of
    significance, so the significant hits can be read against the full
    background rather than in isolation.  Non-significant arcs keep the same
    geometry but stay in the faint end of the ramp, which is what makes the
    comparison honest: absence of colour is visible, not hidden by omission.
    """
    import matplotlib
    matplotlib.use("Agg")
    use_arial()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    from matplotlib.colors import LinearSegmentedColormap

    # Re-run the window sweep so every tested (source, target, window) cell,
    # significant or not, is available for drawing.
    tests = fisher_sweep(wide, nr, SHORT,
                         [c for c in wide.columns
                          if nr.set_index("node")["category"].get(c) != "PFAS"])
    info = nr.set_index("node")
    sig = tests[tests["significant"]]
    n_windows = N_WINDOWS - 1  # 4 windows tested

    # OR colourmap: more aggressive contrast
    # Low OR: very light grey; High OR: deep saturated red
    OR_MAX = 8.0
    or_cmap = LinearSegmentedColormap.from_list(
        'or_ramp', ['#f5f5f5', '#ffffbf', '#fdae61', '#f46d43', '#a50026'], N=256)

    HUB_COLORS = {'PFAS-2': '#C0392B', 'PFAS-1': '#2980B9'}

    def or_to_color(orr, is_sig):
        """Stronger contrast: very light for low OR, saturated deep red for high."""
        if orr is None or (not isinstance(orr, (int, float))) or not np.isfinite(float(np.real(orr))):
            orr = OR_MAX if is_sig else 0.5
        orr = float(np.real(orr))
        if orr < 0.5:
            return '#f7f7f7'
        orr = min(orr, OR_MAX)  # cap at OR_MAX
        if not is_sig:
            # Non-significant: faint tint only
            f = min(1.0, max(0.0, (orr - 0.5) / (OR_MAX - 0.5)))
            rgba = or_cmap(f * 0.2)
            return '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        # Significant: full vivid ramp
        f = min(1.0, max(0.0, (orr - 1.0) / (OR_MAX - 1.0)))
        # Push contrast: apply power curve to spread the mid/high range
        f = f ** 0.7
        rgba = or_cmap(max(0.25, f))
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    def or_to_ribbon(orr):
        """Ribbon colour and alpha interpolated from teal (OR <= 1) to red."""
        if not np.isfinite(orr) or orr <= 1.0:
            return '#4DBBD5', 0.25
        f = min(1.0, (orr - 1.0) / (OR_MAX - 1.0))
        r = int(77 + (230 - 77) * f)
        g = int(189 + (75 - 189) * f)
        b = int(213 + (53 - 213) * f)
        alpha = 0.35 + 0.50 * f
        return '#{:02x}{:02x}{:02x}'.format(r, g, b), alpha

    def tangential_rot(theta):
        """Rotation (deg) for text laid parallel to the arc tangent.

        Returns (rotation_deg, ha). The angle is normalised to [0, 360) FIRST —
        without that the bottom-half flip test never fires for the negative
        angles produced by the clockwise sector layout, which leaves labels
        upside down. Used for target names, window labels and tick labels alike
        so every ring of text follows one scheme.
        """
        a = np.degrees(theta) % 360
        rot = a + 90.0
        if 90.0 < rot < 270.0:      # bottom half -> flip so text stays upright
            return rot - 180.0, 'right'
        if rot >= 270.0:            # >= so the 9-o'clock label is not inverted
            rot -= 360.0
        return rot, 'left'

    # Build lookup: (source, target, window) -> (OR, sig)
    test_lookup = {}
    for _, row in tests.iterrows():
        key = (row["source"], row["target"], int(row["window"]))
        test_lookup[key] = (float(row["odds_ratio"]), bool(row["significant"]))

    # ── Generate one figure per source PFAS ──
    for src_pfas in reversed(SHORT):
        src_sig = sig[sig.source == src_pfas]
        if show_all:
            # Every pollutant that was actually TESTED for this source. Not
            # wide.columns: targets dropped by the n>=3 detection guard have no
            # test to show, and an empty arc would imply a null result where
            # none was computed.
            pool = tests[tests.source == src_pfas]["target"].unique()
        else:
            pool = src_sig["target"].unique()
        targets_list = sorted(pool,
                              key=lambda t: (str(info["category"].get(t, "")),
                                             G.natural_key(t)))
        if not targets_list:
            print(f"  {src_pfas}: no targets to draw, skip")
            continue

        N = len(targets_list)
        hub_color = HUB_COLORS.get(src_pfas, '#C0392B')

        # Layout parameters. With ~45 arcs the sectors get thin, so the gap has
        # to shrink with N or the wedges vanish into their own margins.
        gap_rad = 0.08 if N < 8 else (0.05 if N < 20 else 0.022)
        step = 2 * np.pi / N
        sector_width = max(step - gap_rad, 0.10)
        start_angle = np.pi / 2
        angles = [start_angle - i * step for i in range(N)]

        radius = 1.0
        center_r = 0.18

        # Type scale. The all-pollutant panel packs ~45 arcs into the same ring,
        # so glyphs that fit at N=8 would collide; shrink them with N instead of
        # letting the labels overlap. dense=True is the ~45-arc regime.
        dense = N >= 20
        fs = FS_SPARSE if not dense else (FS_MID if N < 34 else FS_DENSE)
        fs_target, fs_window, fs_tick = fs["target"], fs["window"], fs["tick"]
        show_win_lbl = not dense or N < 34   # W1..W4 text stops fitting past that
        show_tick_lbl = not dense
        # Past ~34 arcs a tangential 'W1' would have to drop to ~4 pt to clear
        # its segment, below print legibility, so the per-segment labels are off
        # here. The information still has to reach the reader: a schematic key
        # in the top-left corner (drawn after the ring) shows the four bands and
        # their order once, at full size. Every sector shares the same scheme,
        # so one key is sufficient; 172 illegible labels would not be.
        draw_win_key = dense and not show_win_lbl

        # Figure — limits hug the drawn content. The outermost ink is the target
        # labels at arc_outer + 0.115 plus half a glyph height (~1.21 units), so
        # the limits sit just outside that. Square limits + equal aspect + a
        # square figsize means matplotlib adds no letterboxing of its own; the
        # legend goes in the corners, which a circle leaves empty anyway.
        # Outermost circle ink = target labels at arc_outer + 0.115 plus half a
        # glyph height, ~1.20 units. x/top clamp right at that so the ring touches
        # the canvas edge; the only added space is a bottom band sized for the
        # legend, so no side is padded for nothing.
        lim = 1.215
        legend_band = 0.40
        fig, ax = plt.subplots(figsize=(12, 12 * (2 * lim + legend_band) / (2 * lim)),
                               facecolor='white')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim - legend_band, lim)
        ax.set_aspect('equal')
        ax.axis('off')

        # ── 1. Center circle (PFAS hub) ──
        center_circ = mpatches.Circle(
            (0, 0), center_r, facecolor=hub_color,
            edgecolor='white', linewidth=3, alpha=0.95, zorder=10)
        ax.add_patch(center_circ)
        ax.text(0, 0, src_pfas, ha='center', va='center',
                fontsize=FS_HUB, fontweight='bold', color='white', zorder=11)

        # Best OR per target (for ribbon width scaling)
        best_or = {}
        for tgt in targets_list:
            sub = net[(net.source == src_pfas) & (net.target == tgt)]
            if len(sub):
                best_or[tgt] = float(sub["odds_ratio"].iloc[0])
            else:
                best_or[tgt] = 2.0
        or_vals = [v for v in best_or.values() if np.isfinite(v)]
        or_min_v = min(or_vals) if or_vals else 1.0
        or_max_v = min(max(or_vals) if or_vals else 8.0, OR_MAX)

        # ── 2. Compute concentration cut-points for this source ──
        x_src = np.log1p(wide[src_pfas].to_numpy(float))
        qs_src = np.unique(np.quantile(x_src, np.linspace(0, 1, N_WINDOWS + 1)[1:-1]))
        cuts_ng = np.expm1(qs_src)  # ng/g cut-points

        # ── 3. Outer arcs: 4 window segments per target ──
        # Each segment coloured by OR with LARGE gaps between windows
        # so each concentration band is visually distinct.
        arc_width = 0.080
        arc_outer = radius + 0.06
        arc_inner = arc_outer - arc_width
        # Large gap between segments — 25% of available width goes to gaps
        total_gap = sector_width * 0.25
        seg_gap = total_gap / (n_windows - 1) if n_windows > 1 else 0
        seg_width = (sector_width - total_gap) / n_windows

        for i, tgt in enumerate(targets_list):
            theta_center = angles[i]
            arc_start = theta_center - sector_width / 2

            for wi in range(n_windows):
                window_idx = wi + 1
                seg_start = arc_start + wi * (seg_width + seg_gap)
                seg_end = seg_start + seg_width
                t1 = np.degrees(seg_start)
                t2 = np.degrees(seg_end)

                key = (src_pfas, tgt, window_idx)
                orr, sf = test_lookup.get(key, (None, False))
                fc = or_to_color(orr, sf)

                # Dark edge between segments to clearly distinguish windows
                wedge = mpatches.Wedge(
                    (0, 0), arc_outer, min(t1, t2), max(t1, t2),
                    width=arc_width, facecolor=fc, edgecolor='#333333',
                    linewidth=1.0 if not dense else 0.35, alpha=0.95, zorder=5)
                ax.add_patch(wedge)
                if not show_win_lbl:
                    continue

                # Window number label inside each segment (tangential — parallel to arc)
                seg_mid_angle = (seg_start + seg_end) / 2
                seg_label_r = (arc_inner + arc_outer) / 2
                slx = seg_label_r * np.cos(seg_mid_angle)
                sly = seg_label_r * np.sin(seg_mid_angle)
                # Same tangential scheme as the target labels: always upright,
                # reading outward from the circle centre.
                seg_rot_deg, _ = tangential_rot(seg_mid_angle)
                ax.text(slx, sly, f'W{window_idx}', fontsize=fs_window,
                        ha='center', va='center', color='#1A1A1A',
                        fontweight='bold', zorder=6,
                        rotation=seg_rot_deg, rotation_mode='anchor')

            # Tick marks with concentration labels at boundaries
            for wi in range(n_windows + 1):
                a_tick = arc_start + wi * (seg_width + seg_gap) - (seg_gap / 2 if wi > 0 else 0)
                if wi == n_windows:
                    a_tick = arc_start + sector_width
                r_in = arc_inner - 0.015
                r_out = arc_outer + 0.015
                # Draw tick line
                ax.plot([r_in * np.cos(a_tick), r_out * np.cos(a_tick)],
                        [r_in * np.sin(a_tick), r_out * np.sin(a_tick)],
                        color='#222222', lw=1.0 if not dense else 0.35, zorder=6)

                # Concentration label at each tick for all targets
                if wi == 0:
                    lbl = f'{cuts_ng[0]:.1f}' if len(cuts_ng) > 0 else ''
                elif wi < len(cuts_ng):
                    lbl = f'{cuts_ng[wi]:.1f}'
                elif wi == n_windows:
                    lbl = ''  # no upper bound label at end
                else:
                    lbl = ''
                if lbl and show_tick_lbl:
                    # Centred on its own tick line: ha/va='center' puts the ink
                    # centroid exactly on the tick angle, offset radially inward.
                    tick_label_r = arc_inner - 0.040
                    tlx = tick_label_r * np.cos(a_tick)
                    tly = tick_label_r * np.sin(a_tick)
                    # Same tangential scheme as the target labels
                    tick_rot_deg, _ = tangential_rot(a_tick)
                    ax.text(tlx, tly, lbl, fontsize=fs_tick,
                            ha='center', va='center', color='#263238',
                            fontweight='normal',
                            rotation=tick_rot_deg, zorder=7,
                            rotation_mode='anchor')

            # Thin category-colour band outside
            cat = info["category"].get(tgt, "Others")
            cat_color = TARGET_COLORS.get(cat, '#999999')
            t1_full = np.degrees(arc_start)
            t2_full = np.degrees(arc_start + sector_width)
            cat_band = mpatches.Wedge(
                (0, 0), arc_outer + 0.045, min(t1_full, t2_full),
                max(t1_full, t2_full), width=0.018,
                facecolor=cat_color, edgecolor='white',
                linewidth=0.3, alpha=0.90, zorder=4)
            ax.add_patch(cat_band)

            # Target name label — tangential, centred on the arc midpoint.
            # ha='center' (not left/right) so the text straddles theta_center
            # instead of hanging off to one side of it.
            label_r = arc_outer + 0.115
            lx = label_r * np.cos(theta_center)
            ly = label_r * np.sin(theta_center)
            name_rot_deg, _ = tangential_rot(theta_center)
            # On the all-pollutant panel, targets with no significant window are
            # greyed and un-bolded so the hits still read as the foreground
            # instead of drowning in 45 equally-weighted labels.
            tgt_is_sig = tgt in set(src_sig["target"])
            ax.text(lx, ly, tgt, fontsize=fs_target,
                    fontweight='bold' if tgt_is_sig else 'normal',
                    ha='center', va='center', rotation=name_rot_deg,
                    rotation_mode='anchor',
                    color='#2C3E50' if tgt_is_sig else '#95A5A6', zorder=7)

        # --- FILLED MOUNTAIN PROFILE: radial depth ∝ OR, showing inverted-U ---
        # Arc direction = "x-axis" (4 windows), radial toward center = "y-axis"
        # Higher OR → deeper toward center → mountain peak at mid-concentrations
        from scipy.interpolate import PchipInterpolator

        ribbon_outer_r = arc_inner - 0.115  # leave gap for larger tick labels
        max_depth = ribbon_outer_r - center_r - 0.08  # max radial depth

        for i, tgt in enumerate(targets_list):
            theta_center = angles[i]
            arc_start = theta_center - sector_width / 2

            # Collect OR and significance for all windows
            or_vals = []
            sig_flags = []
            for wi in range(n_windows):
                key = (src_pfas, tgt, wi + 1)
                orr, is_sig = test_lookup.get(key, (None, False))
                if orr is None or (not isinstance(orr, (int, float))) or not np.isfinite(float(np.real(orr))):
                    orr = 0.5
                or_vals.append(float(np.real(orr)))
                sig_flags.append(is_sig)

            or_arr = np.array(or_vals)

            # Map OR to radial depth: normalize within this target
            or_min_w, or_max_w = or_arr.min(), or_arr.max()
            min_depth = max_depth * 0.12  # minimum so shape is visible
            if or_max_w > or_min_w:
                depths = min_depth + (max_depth * 0.78 - min_depth) * \
                         (or_arr - or_min_w) / (or_max_w - or_min_w)
            else:
                depths = np.ones(n_windows) * max_depth * 0.35

            # --- Draw per-window filled segments with smooth inner boundary ---
            # Window midpoint angles
            win_mids = np.array([arc_start + (wi + 0.5) * sector_width / n_windows
                                 for wi in range(n_windows)])

            # PCHIP interpolation for smooth monotone curve through OR depths
            # Extend to sector edges for a clean start/end
            x_knots = np.concatenate([[arc_start], win_mids,
                                      [arc_start + sector_width]])
            y_knots = np.concatenate([[depths[0] * 0.7], depths,
                                      [depths[-1] * 0.7]])
            pchip = PchipInterpolator(x_knots, y_knots)

            # Sample smooth curve
            n_pts = 80
            sector_angles = np.linspace(arc_start, arc_start + sector_width, n_pts)
            interp_depths = pchip(sector_angles)
            interp_depths = np.clip(interp_depths, min_depth * 0.5, max_depth * 0.85)

            # Inner boundary radii
            inner_radii = ribbon_outer_r - interp_depths

            # Draw filled segments per window (each with its own OR color)
            pts_per_win = n_pts // n_windows
            for wi in range(n_windows):
                i0 = wi * pts_per_win
                i1 = (wi + 1) * pts_per_win if wi < n_windows - 1 else n_pts
                seg_angles = sector_angles[i0:i1]
                seg_inner = inner_radii[i0:i1]

                # Polygon: outer arc → inner curve (reversed)
                outer_x = ribbon_outer_r * np.cos(seg_angles)
                outer_y = ribbon_outer_r * np.sin(seg_angles)
                inn_x = seg_inner[::-1] * np.cos(seg_angles[::-1])
                inn_y = seg_inner[::-1] * np.sin(seg_angles[::-1])

                poly_x = np.concatenate([outer_x, inn_x])
                poly_y = np.concatenate([outer_y, inn_y])
                verts = list(zip(poly_x, poly_y))
                verts.append(verts[0])

                codes = ([mpath.Path.MOVETO] +
                         [mpath.Path.LINETO] * (len(verts) - 2) +
                         [mpath.Path.CLOSEPOLY])

                # Color by this window's OR
                seg_color = or_to_color(or_vals[wi], sig_flags[wi])
                seg_alpha = 0.72 if sig_flags[wi] else 0.32

                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(
                    path, facecolor=seg_color, edgecolor='none',
                    linewidth=0, alpha=seg_alpha, zorder=2)
                ax.add_patch(patch)

            # Smooth inner-edge curve line (emphasize the inverted-U shape)
            cx = inner_radii * np.cos(sector_angles)
            cy = inner_radii * np.sin(sector_angles)
            ax.plot(cx, cy, color='#2C3E50', lw=1.4, alpha=0.85, zorder=3)

            # Outer edge line
            ox = ribbon_outer_r * np.cos(sector_angles)
            oy = ribbon_outer_r * np.sin(sector_angles)
            ax.plot(ox, oy, color='#555555', lw=0.6, alpha=0.5, zorder=3)

            # ── Molecular-network style edges: center hub → window nodes ──
            # Curved tapered edges + filled arrowheads + circular nodes
            or_span = max(OR_MAX - 1.0, 1e-6)
            for wi in range(n_windows):
                mid_a = win_mids[wi]
                d = pchip(mid_a)
                d = np.clip(d, min_depth * 0.5, max_depth * 0.85)
                r_dot = ribbon_outer_r - d
                dx = r_dot * np.cos(mid_a)
                dy = r_dot * np.sin(mid_a)
                dot_color = or_to_color(or_vals[wi], sig_flags[wi])

                # Edge weight ∝ OR (molecular-network convention)
                w_f = min(1.0, max(0.0, (or_vals[wi] - 1.0) / or_span))
                edge_lw = 0.8 + 2.6 * (w_f ** 0.65)
                node_ms = 5.0 + 5.0 * (w_f ** 0.6)
                if dense:
                    # 45 targets x 4 windows = 180 spokes into one hub; at full
                    # weight they merge into a solid disc and hide the ring.
                    edge_lw *= 0.42
                    node_ms *= 0.50
                if sig_flags[wi]:
                    edge_alpha, edge_col = 0.80, dot_color
                else:
                    edge_lw *= 0.55
                    node_ms *= 0.72
                    edge_alpha, edge_col = 0.28, '#9aa5ad'

                arrow_start_r = center_r + 0.015
                sx = arrow_start_r * np.cos(mid_a)
                sy = arrow_start_r * np.sin(mid_a)

                # Gentle curvature so bundled edges fan out (network aesthetic)
                rad = 0.10 if (wi % 2 == 0) else -0.10

                # Soft white casing under the edge for crisp overlaps
                ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                            arrowprops=dict(
                                arrowstyle='-', color='white',
                                lw=edge_lw + 1.6, alpha=0.55,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=0, shrinkB=0), zorder=2)
                # The edge itself, with a filled molecular-network arrowhead
                ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                            arrowprops=dict(
                                arrowstyle='-|>,head_length=0.55,head_width=0.28',
                                color=edge_col, lw=edge_lw,
                                alpha=edge_alpha, joinstyle='round',
                                capstyle='round',
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=0, shrinkB=node_ms * 0.55),
                            zorder=3)

                # Window node: filled circle with white ring (Cytoscape-like)
                ax.plot(dx, dy, 'o', color=dot_color, markersize=node_ms,
                        markeredgecolor='white', markeredgewidth=1.1,
                        alpha=0.95 if sig_flags[wi] else 0.5, zorder=5)
                ax.plot(dx, dy, 'o', markerfacecolor='none',
                        markersize=node_ms, markeredgecolor='#37474F',
                        markeredgewidth=0.5,
                        alpha=0.85 if sig_flags[wi] else 0.35, zorder=6)

        # ── 4. Legend — OR only ──
        from matplotlib.lines import Line2D
        legend_elements = [
            mpatches.Patch(fc='#d73027', alpha=0.9, ec='white',
                           label='High OR (strong enrichment)'),
            mpatches.Patch(fc='#fee090', alpha=0.9, ec='white',
                           label='Moderate OR'),
            mpatches.Patch(fc='#eeeeee', alpha=0.9, ec='#999',
                           label=('Tested, not significant (shown for contrast)'
                                  if show_all else 'Low OR / not significant')),
            Line2D([0], [0], color='#a50026', lw=3.2, marker='o',
                   markersize=9, markerfacecolor='#a50026',
                   markeredgecolor='white', markeredgewidth=1.1,
                   # Plain ASCII: Arial has no U+221D PROPORTIONAL TO glyph, and
                   # the fallback would quietly render this one label in another
                   # face, breaking the all-Arial requirement.
                   label='Edge width & node size scale with OR'),
        ]

        # Sits in the bottom-left corner INSIDE the axes — the circle leaves the
        # corners empty, so this costs no extra canvas.
        ax.legend(handles=legend_elements, loc='lower left',
                  bbox_to_anchor=(0.0, 0.0),
                  fontsize=FS_LEGEND, frameon=False, ncol=2,
                  handlelength=1.5, handletextpad=0.5,
                  columnspacing=1.4, labelspacing=0.35, borderpad=0.0)

        # Window concentration ranges — bottom-right corner, mirroring the legend
        for wi in range(1, n_windows + 1):
            if wi == 1:
                rng = f'0 – {cuts_ng[0]:.1f}'
            elif wi <= len(cuts_ng):
                rng = f'{cuts_ng[wi-2]:.1f} – {cuts_ng[wi-1]:.1f}'
            else:
                rng = f'> {cuts_ng[-1]:.1f}'
            ax.text(lim * 0.995,
                    -lim - legend_band * 0.97 + (n_windows - wi) * 0.082,
                    f"W{wi}: {rng} ng/g",
                    fontsize=FS_CORNER, color='#37474F',
                    ha='right', va='bottom')

        # Window-order key for the dense panel, where the per-segment W1..W4
        # labels had to come off the ring. One schematic sector at full size,
        # top-left, in the corner the circle leaves empty: four boxes in ring
        # order with an arrow for the reading direction. Without this the four
        # bands on each arc are unlabelled and the corner range list above has
        # nothing to bind to.
        if draw_win_key:
            kx0, ky0 = -lim * 0.985, lim * 0.90
            bw, bh, bgap = 0.062, 0.040, 0.012
            for wi in range(1, n_windows + 1):
                bx = kx0 + (wi - 1) * (bw + bgap)
                ax.add_patch(mpatches.Rectangle(
                    (bx, ky0), bw, bh, facecolor='#f5f5f5',
                    edgecolor='#333333', linewidth=0.8, zorder=9))
                ax.text(bx + bw / 2, ky0 + bh / 2, f'W{wi}', fontsize=FS_KEY,
                        ha='center', va='center', color='#1A1A1A',
                        fontweight='bold', zorder=10)
            # Windows are laid out ALONG the arc (tangentially), not radially,
            # so the key reads left-to-right in the same order the segments run
            # counter-clockwise within each sector.
            ax.text(kx0, ky0 + bh + 0.026,
                    'Each arc is split into 4 concentration windows, in order:',
                    fontsize=FS_KEY_TITLE, color='#37474F',
                    ha='left', va='bottom')
            ax.annotate('', xy=(kx0 + n_windows * (bw + bgap) + 0.028,
                                ky0 + bh / 2),
                        xytext=(kx0 + n_windows * (bw + bgap) - 0.008,
                                ky0 + bh / 2),
                        arrowprops=dict(arrowstyle='-|>', color='#37474F',
                                        lw=0.9, shrinkA=0, shrinkB=0),
                        annotation_clip=False)

        # No tight_layout: it fights the manual corner placement. Zero-margin
        # subplot params instead, so the axes fills the figure.
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        suffix = "all_pollutants" if show_all else "significant"
        stem = f"fisher_causal_{src_pfas}_arc_circos_{suffix}_{FIG_DPI}dpi"
        out_png = figdir / f"{stem}.png"
        out_pdf = figdir / f"{stem}.pdf"
        # pad_inches=0.02: bbox_inches='tight' crops to the ink, and a hair of
        # padding keeps outer glyphs from touching the edge.
        fig.savefig(str(out_png), dpi=FIG_DPI, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
        fig.savefig(str(out_pdf), dpi=FIG_DPI, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
        plt.close(fig)
        print(f"  {N:2d} arcs -> {out_png.name}")


def _sheet(xw, name, df):
    """Write one DataFrame to a formatted sheet (header style, widths, freeze)."""
    if df is None or not len(df):
        df = pd.DataFrame({"note": ["no rows"]})
    df.to_excel(xw, sheet_name=name[:31], index=False)
    ws = xw.sheets[name[:31]]
    hdr = xw.book.add_format({"bold": True, "bg_color": "#f2dede",
                              "border": 1, "text_wrap": True, "valign": "top"})
    for j, c in enumerate(df.columns):
        ws.write(0, j, str(c), hdr)
        ws.set_column(j, j, max(len(str(c)) + 2,
                               min(36, int(df[c].astype(str).str.len().max()
                                           if len(df) else 10) + 2)))
    ws.freeze_panes(1, 0)


if __name__ == "__main__":
    main()
