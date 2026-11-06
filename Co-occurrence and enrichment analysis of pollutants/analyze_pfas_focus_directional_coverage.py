"""Directional co-detection coverage matrix focused on short-chain PFAS.

Analysis unit: one unique (Organ, Sample) specimen.

Variables
    PFAS keeps its cluster resolution (PFAS-1 ... PFAS-5); every non-PFAS
    pollutant is summed to its Category (Pest., Pharm., OPEs, ...).

Metric
    C(i|j) = n(X_i > 0 and X_j > 0) / n(X_j > 0), i.e. the probability that the
    row variable is detected given that the column variable is detected.

Significance
    For the PFAS-1 / PFAS-2 focus rows, a one-sided exact binomial test of
    C(focus|other) > COVERAGE_BASELINE, Benjamini-Hochberg FDR within the focus
    comparisons.  A top-25% high-burden Fisher enrichment table is exported as
    a reference only.

Outputs (``pfas_focus_directional_outputs/``)
    Excel workbook, 1000-dpi PNG + vector PDF heatmap, Word methods (Chinese)
    and a Markdown summary (Chinese).

Run: python analyze_pfas_focus_directional_coverage.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from PIL import Image, ImageDraw, ImageFont
from vector_pdf import VectorDraw, save_vector_pdf
from gee_causal_core import LONG_CHAIN, PFAS_ORDER, PROJECT_DIR, SHORT_CHAIN, WORKBOOK


DPI = 1000                     # export resolution of the PNG / PDF figure
REQUIRED_COLUMNS = {"Sample", "Organ", "Category", "Clusters", "Name", "Concentration"}
FOCUS = SHORT_CHAIN.copy()     # focus rows of the matrix: PFAS-1, PFAS-2
COVERAGE_BASELINE = 0.5        # H0 of the one-sided binomial coverage test
OUT_DIR = PROJECT_DIR / "pfas_focus_directional_outputs"
FIG_DIR = OUT_DIR / "figures"


# --------------------------------------------------------------------------
# input and variable construction
# --------------------------------------------------------------------------
def natural_key(text: str) -> tuple:
    """Sort key for 'Prefix-N' labels with a numeric suffix."""
    head, _, tail = text.partition("-")
    try:
        n = int(tail)
    except ValueError:
        n = 10**9
    return head.lower(), n, text.lower()


def find_input_workbook() -> Path:
    """Return the shared input workbook path, failing early if it is missing."""
    if not WORKBOOK.is_file():
        raise FileNotFoundError(f"Analysis workbook not found: {WORKBOOK}")
    return WORKBOOK


def read_main_table(path: Path) -> pd.DataFrame:
    """Read the long-format table, coerce concentrations and add a specimen id."""
    df = pd.read_excel(path, sheet_name=0)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    # Non-detects were removed from the source table; missing values count as 0.
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce").fillna(0)
    df["Sample"] = df["Sample"].astype(str).str.strip()
    df["Organ"] = df["Organ"].astype(str).str.strip()
    df["Clusters"] = df["Clusters"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip().replace({"others": "Others"})
    df["Name"] = df["Name"].astype(str).str.strip()
    df["sample_id"] = df["Organ"] + " | " + df["Sample"]
    return df


def pfas_class(cluster: str, category: str) -> str:
    """Reporting group of a plot variable."""
    if category != "PFAS":
        return "Non-PFAS category total"
    if cluster in SHORT_CHAIN:
        return "Short-chain PFAS cluster"
    if cluster in LONG_CHAIN:
        return "Long-chain / precursor PFAS cluster"
    return "PFAS cluster"


def build_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the specimen x variable concentration matrix.

    PFAS rows keep their Clusters value; non-PFAS rows are pooled by Category.
    Returns (long grouped table, wide matrix, ordered variable list).
    """
    work = df.copy()
    work["Category_norm"] = work["Category"].replace({"others": "Others"})
    work["plot_variable"] = np.where(work["Category_norm"].eq("PFAS"), work["Clusters"], work["Category_norm"])
    grouped = (
        work.groupby(["Organ", "Sample", "sample_id", "plot_variable"], as_index=False)["Concentration"]
        .sum()
    )
    matrix = (
        grouped.pivot_table(
            index=["Organ", "Sample", "sample_id"],
            columns="plot_variable",
            values="Concentration",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    non_pfas = sorted(
        [c for c in work.loc[~work["Category_norm"].eq("PFAS"), "Category_norm"].dropna().unique()],
        key=str.lower,
    )
    variables = [c for c in PFAS_ORDER + non_pfas if c in matrix.columns]
    matrix = matrix[["Organ", "Sample", "sample_id"] + variables]
    return grouped, matrix, variables


def annotations(df: pd.DataFrame, matrix: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Per-variable composition and detection summary for the Excel export."""
    rows = []
    work = df.copy()
    work["Category_norm"] = work["Category"].replace({"others": "Others"})
    for var in variables:
        if var in PFAS_ORDER:
            sub = work[work["Clusters"].eq(var)]
            category = "PFAS"
            names = sorted(set(sub["Name"].dropna().astype(str)))
            group = pfas_class(var, "PFAS")
            composition = "; ".join(names)
            n_compounds = len(names)
            n_clusters = 1
        else:
            sub = work[work["Category_norm"].eq(var)]
            category = var
            clusters = sorted(set(sub["Clusters"].dropna().astype(str)), key=natural_key)
            names = sorted(set(sub["Name"].dropna().astype(str)))
            group = "Non-PFAS category total"
            composition = "; ".join(clusters)
            n_compounds = len(names)
            n_clusters = len(clusters)
        x = pd.to_numeric(matrix[var], errors="coerce").fillna(0)
        rows.append(
            {
                "Variable": var,
                "Category": category,
                "Variable_group": group,
                "n_clusters_merged": n_clusters,
                "n_compounds": n_compounds,
                "composition_or_clusters": composition,
                "n_samples_detected": int((x > 0).sum()),
                "detection_rate": float((x > 0).mean()),
                "total_concentration": float(x.sum()),
                "median_positive_concentration": float(x[x > 0].median()) if (x > 0).any() else 0.0,
                "q75_positive_concentration": float(x[x > 0].quantile(0.75)) if (x > 0).any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# exact tests (log-space, no scipy dependency)
# --------------------------------------------------------------------------
def logcomb(n: int, k: int) -> float:
    """log(n choose k); -inf outside the support."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k: int, n_a: int, n_b: int, n: int) -> float:
    """P[X >= k], X ~ Hypergeom(n, n_a, n_b): one-sided Fisher exact p-value."""
    upper = min(n_a, n_b)
    if k > upper:
        return 0.0
    terms = [
        logcomb(n_a, i) + logcomb(n - n_a, n_b - i) - logcomb(n, n_b)
        for i in range(k, upper + 1)
        if 0 <= n_b - i <= n - n_a
    ]
    if not terms:
        return 1.0
    m = max(terms)
    return min(1.0, float(math.exp(m) * sum(math.exp(t - m) for t in terms)))


def bh_fdr(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        true_rank = m - rank + 1
        val = min(prev, pvalues[idx] * m / true_rank)
        q[idx] = min(1.0, val)
        prev = val
    return q


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P[Binomial(n, p) >= k]."""
    if n <= 0:
        return math.nan
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    terms = [logcomb(n, i) + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, float(math.exp(m) * sum(math.exp(t - m) for t in terms)))


def binary_sets(values: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Binarize each variable; 'high_burden' = detected and >= Q75 of positives."""
    thresholds = []
    binary = pd.DataFrame(index=values.index)
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").fillna(0)
        if mode == "high_burden":
            positive = x[x > 0]
            threshold = float(positive.quantile(0.75)) if len(positive) else math.inf
            binary[col] = (x > 0) & (x >= threshold)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        thresholds.append(
            {
                "Cluster": col,
                "mode": mode,
                "threshold": threshold,
                "n_set_samples": int(binary[col].sum()),
                "set_rate": float(binary[col].mean()),
            }
        )
    return binary.astype(bool), pd.DataFrame(thresholds)


def enrichment_results(binary: pd.DataFrame, ann: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Pairwise one-sided Fisher enrichment of co-occurrence with BH-FDR."""
    ann_index = ann.rename(columns={"Variable": "Cluster", "Variable_group": "PFAS_group"}).set_index("Cluster")
    cols = list(binary.columns)
    n = len(binary)
    rows = []
    for i, a in enumerate(cols):
        xa = binary[a].to_numpy()
        n_a = int(xa.sum())
        for b in cols[i + 1 :]:
            xb = binary[b].to_numpy()
            n_b = int(xb.sum())
            both = int((xa & xb).sum())
            only_a = n_a - both
            only_b = n_b - both
            neither = n - both - only_a - only_b
            expected = n_a * n_b / n if n else 0.0
            fold = both / expected if expected > 0 else math.nan
            if only_a == 0 or only_b == 0:
                odds = math.inf if both * neither > 0 else math.nan
            else:
                odds = (both * neither) / (only_a * only_b)
            p = hypergeom_sf(both, n_a, n_b, n)
            rows.append(
                {
                    "mode": mode,
                    "Cluster_A": a,
                    "Cluster_B": b,
                    "Category_A": ann_index.loc[a, "Category"],
                    "Category_B": ann_index.loc[b, "Category"],
                    "PFAS_group_A": ann_index.loc[a, "PFAS_group"],
                    "PFAS_group_B": ann_index.loc[b, "PFAS_group"],
                    "N_background_samples": n,
                    "n_A": n_a,
                    "n_B": n_b,
                    "n_both_observed": both,
                    "n_both_expected": expected,
                    "fold_enrichment_observed_vs_expected": fold,
                    "odds_ratio": odds,
                    "fisher_p_greater": p,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_q"] = bh_fdr(out["fisher_p_greater"].fillna(1).tolist())
    out["significant_q_lt_0_05"] = out["fdr_q"] < 0.05
    out["log2_fold_enrichment"] = np.log2(out["fold_enrichment_observed_vs_expected"].replace(0, np.nan))
    return out.sort_values(["fdr_q", "fisher_p_greater", "fold_enrichment_observed_vs_expected"], ascending=[True, True, False])


def enrichment_matrix(enrich: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Symmetric log2 fold-enrichment matrix (clipped to +/-2.5) and its q<0.05 mask."""
    mat = pd.DataFrame(0.0, index=variables, columns=variables)
    sig = pd.DataFrame(False, index=variables, columns=variables)
    for _, row in enrich.iterrows():
        a, b = row["Cluster_A"], row["Cluster_B"]
        val = row["log2_fold_enrichment"]
        if not np.isfinite(val):
            continue
        mat.loc[a, b] = mat.loc[b, a] = float(max(-2.5, min(2.5, val)))
        sig.loc[a, b] = sig.loc[b, a] = bool(row["significant_q_lt_0_05"])
    return mat, sig


def presence_jaccard(x: pd.Series, y: pd.Series) -> float:
    """Jaccard index of the two detection sets."""
    xb = np.asarray(x) > 0
    yb = np.asarray(y) > 0
    denom = (xb | yb).sum()
    return float((xb & yb).sum() / denom) if denom else 0.0


def cosine_similarity(x: pd.Series, y: pd.Series) -> float:
    """Cosine similarity of two concentration vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else 0.0


def bray_curtis_similarity(x: pd.Series, y: pd.Series) -> float:
    """1 - Bray-Curtis dissimilarity of two concentration vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = (x + y).sum()
    return float(1 - np.abs(x - y).sum() / denom) if denom else 0.0


def ordered_variables(raw: pd.DataFrame, variables: list[str]) -> list[str]:
    """Focus PFAS first, then other variables by ascending detection count."""
    remaining = [v for v in variables if v not in FOCUS]
    counts = raw.gt(0).sum().to_dict()
    remaining = sorted(remaining, key=lambda v: (counts[v], v))
    return [v for v in FOCUS if v in variables] + remaining


def directional_coverage_matrix(raw: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Full square matrix of C(row|column) = P(row detected | column detected)."""
    present = raw[variables].gt(0)
    out = pd.DataFrame(1.0, index=variables, columns=variables)
    counts = present.sum()
    for row_var in variables:
        for col_var in variables:
            denom = counts[col_var]
            out.loc[row_var, col_var] = float((present[row_var] & present[col_var]).sum() / denom) if denom else np.nan
    return out


def pairwise_directional_table(raw: pd.DataFrame, variables: list[str], direction: pd.DataFrame) -> pd.DataFrame:
    """Upper-triangle long table holding both coverage directions of each pair."""
    rows = []
    present = raw[variables].gt(0)
    for i, a in enumerate(variables):
        for b in variables[i + 1 :]:
            rows.append(
                {
                    "row_variable": a,
                    "column_variable": b,
                    "n_row_present": int(present[a].sum()),
                    "n_column_present": int(present[b].sum()),
                    "n_both_present": int((present[a] & present[b]).sum()),
                    "P_row_present_given_column_present": float(direction.loc[a, b]),
                    "P_column_present_given_row_present": float(direction.loc[b, a]),
                    "is_focus_row_pair": a in FOCUS,
                }
            )
    return pd.DataFrame(rows)


def focus_coverage_tests(raw: pd.DataFrame, variables: list[str], baseline: float = COVERAGE_BASELINE) -> pd.DataFrame:
    """One-sided exact binomial test of P(focus | other) > baseline, BH-FDR."""
    rows = []
    present = raw[variables].gt(0)
    for focus in FOCUS:
        for other in variables:
            if other == focus:
                continue
            n = int(present[other].sum())
            k = int((present[focus] & present[other]).sum())
            coverage = k / n if n else np.nan
            fold = coverage / baseline if n else np.nan
            rows.append(
                {
                    "Focus": focus,
                    "Other": other,
                    "coverage_test": "one-sided exact binomial test",
                    "null_hypothesis": f"P({focus} present | {other} present) <= {baseline:g}",
                    "alternative": f"P({focus} present | {other} present) > {baseline:g}",
                    "coverage_baseline": baseline,
                    "n_other_present": n,
                    "n_focus_and_other_present": k,
                    "coverage": coverage,
                    "coverage_fold_over_baseline": fold,
                    "coverage_log2_fold_over_baseline": math.log2(fold) if fold and fold > 0 else np.nan,
                    "coverage_p_greater_baseline": binom_sf_ge(k, n, baseline),
                }
            )
    out = pd.DataFrame(rows)
    out["coverage_focus_fdr_q"] = bh_fdr(out["coverage_p_greater_baseline"].fillna(1).tolist())
    out["coverage_focus_fdr_q_lt_0_05"] = out["coverage_focus_fdr_q"] < 0.05
    return out.sort_values(["Focus", "coverage_focus_fdr_q", "coverage"], ascending=[True, True, False])


def focus_coverage_matrices(tests: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Place the focus-row test results into square matrices for plotting."""
    enrich = pd.DataFrame(0.0, index=variables, columns=variables)
    sig = pd.DataFrame(False, index=variables, columns=variables)
    for _, row in tests.iterrows():
        focus = row["Focus"]
        other = row["Other"]
        if focus in enrich.index and other in enrich.columns:
            val = row["coverage_log2_fold_over_baseline"]
            enrich.loc[focus, other] = float(max(-1, min(1, val))) if np.isfinite(val) else 0.0
            sig.loc[focus, other] = bool(row["coverage_focus_fdr_q_lt_0_05"])
    return enrich, sig


def focused_coverage_table(raw: pd.DataFrame, variables: list[str], high: pd.DataFrame) -> pd.DataFrame:
    """Focus-vs-other summary: coverage, similarity indices and top-25% reference."""
    rows = []
    log_values = np.log1p(raw[variables])
    high_lookup = {frozenset([row["Cluster_A"], row["Cluster_B"]]): row for _, row in high.iterrows()}
    for focus in FOCUS:
        f_present = raw[focus] > 0
        f_n = int(f_present.sum())
        for other in variables:
            if other == focus:
                continue
            o_present = raw[other] > 0
            o_n = int(o_present.sum())
            both_present = int((f_present & o_present).sum())
            h = high_lookup.get(frozenset([focus, other]))
            rows.append(
                {
                    "Focus": focus,
                    "Other": other,
                    "n_focus_present": f_n,
                    "n_other_present": o_n,
                    "n_both_present": both_present,
                    "P_focus_present_given_other_present": both_present / o_n if o_n else np.nan,
                    "P_other_present_given_focus_present": both_present / f_n if f_n else np.nan,
                    "presence_jaccard": presence_jaccard(raw[focus], raw[other]),
                    "bray_curtis_similarity_log1p": bray_curtis_similarity(log_values[focus], log_values[other]),
                    "cosine_similarity_log1p": cosine_similarity(log_values[focus], log_values[other]),
                    "reference_top25_high_observed": int(h["n_both_observed"]) if h is not None else np.nan,
                    "reference_top25_high_expected": float(h["n_both_expected"]) if h is not None else np.nan,
                    "reference_top25_fold_enrichment": float(h["fold_enrichment_observed_vs_expected"]) if h is not None else np.nan,
                    "reference_top25_fdr_q": float(h["fdr_q"]) if h is not None else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Focus", "P_focus_present_given_other_present", "presence_jaccard"],
        ascending=[True, False, False],
    )


# --------------------------------------------------------------------------
# figure (PIL raster + recorded vector PDF)
# --------------------------------------------------------------------------
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Arial if available, then Calibri / Segoe UI, then PIL's default."""
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def lerp(a: int, b: int, t: float) -> int:
    """Linear interpolation between two channel values."""
    return int(round(a + (b - a) * t))


def diverging(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Blue-white-red colour for a value in [vmin, vmax] (white at 0)."""
    if not np.isfinite(value):
        return (235, 235, 235)
    value = max(vmin, min(vmax, value))
    if value < 0:
        t = value / vmin if vmin else 0
        return (lerp(250, 42, t), lerp(250, 111, t), lerp(250, 174, t))
    t = value / vmax if vmax else 0
    return (lerp(250, 213, t), lerp(250, 62, t), lerp(250, 79, t))


def draw_rotated_label(base: Image.Image, text: str, xy: tuple[int, int], size: int = 12,
                       vector_draw: VectorDraw | None = None) -> None:
    """Paste a 90-degree rotated column label and record it for the PDF."""
    fnt = font(size)
    bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=fnt)
    w, h = bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4
    scale = vector_draw.scale if vector_draw is not None else 1
    tmp = Image.new("RGBA", (round(w * scale), round(h * scale)), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    d.text((2 * scale, 2 * scale), text, fill=(35, 35, 35), font=font(round(size * scale)))
    rot = tmp.rotate(90, expand=True)
    base.paste(rot, tuple(round(v * scale) for v in xy), rot)
    if vector_draw is not None:
        # A 90-degree PIL rotation maps (x, y) to (y, width - x).
        baseline = (xy[0] + 2 + fnt.getmetrics()[0], xy[1] + w - 2)
        vector_draw.text(baseline, text, fill=(35, 35, 35), font=fnt,
                         vector_only=True, rotation=90, anchor="ls")


def draw_matrix_figure(
    corr: pd.DataFrame,
    path: Path,
    title: str,
    pdf_path: Path,
    enrich_sig: pd.DataFrame,
    cell: int = 230,
    label_size: int = 50,
    value_size: int = 50,
) -> None:
    """Draw the annotated coverage heatmap; black dot marks focus FDR q < 0.05."""
    labels = list(corr.columns)
    n = len(labels)
    label_font = font(label_size)
    value_font = font(value_size, bold=True)
    max_label_w = max(
        ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=label_font)[2]
        for label in labels
    )
    margin_left = max(190, max_label_w + 45)
    margin_top = 310
    margin_right = 70
    margin_bottom = max(210, max_label_w + 70)
    w = margin_left + n * cell + margin_right
    h = margin_top + n * cell + margin_bottom
    # Preserve the original physical size and layout while increasing the
    # raster drawing resolution from the original 800 dpi to 1000 dpi.
    scale = DPI / 800
    img = Image.new("RGB", (round(w * scale), round(h * scale)), "white")
    d = VectorDraw(img, scale=scale)
    d.text((margin_left, 34), title, fill=(20, 20, 20), font=font(40, bold=True))
    subtitle = "Directional coverage matrix: P(row variable detected | column variable detected). Black dot: FDR q < 0.05."
    d.text((margin_left, 86), subtitle, fill=(65, 65, 65), font=font(24))
    legend_y = 150
    legend_font = font(45)
    d.text((margin_left, legend_y), "Coverage", fill=(45, 45, 45), font=legend_font)
    bar_x = margin_left + 210
    bar_w = 420
    bar_h = 44
    for x0 in range(bar_w):
        val = x0 / (bar_w - 1)
        d.line([(bar_x + x0, legend_y + 3), (bar_x + x0, legend_y + 3 + bar_h)], fill=diverging(val, 0, 1))
    d.text((bar_x - 3, legend_y + 62), "0", fill=(60, 60, 60), font=font(32))
    d.text((bar_x + 190, legend_y + 62), "0.5", fill=(60, 60, 60), font=font(32))
    d.text((bar_x + 400, legend_y + 62), "1", fill=(60, 60, 60), font=font(32))
    dot_x = bar_x + bar_w + 150
    d.ellipse([dot_x, legend_y + 13, dot_x + 42, legend_y + 55], fill=(20, 20, 20))
    d.text((dot_x + 60, legend_y), "FDR q < 0.05", fill=(45, 45, 45), font=legend_font)
    for i, row_label in enumerate(labels):
        y0 = margin_top + i * cell
        d.text((18, y0 + cell * 0.34), row_label, fill=(35, 35, 35), font=label_font)
        for j, col_label in enumerate(labels):
            x0 = margin_left + j * cell
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=(252, 252, 252), outline=(235, 235, 235), width=2)
            val = float(corr.loc[row_label, col_label])
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=diverging(val, 0, 1), outline=(255, 255, 255), width=2)
            if row_label in FOCUS:
                d.rectangle([x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2], outline=(210, 35, 60), width=3)
            if bool(enrich_sig.loc[row_label, col_label]):
                r = max(8, int(cell * 0.075))
                cx = x0 + cell - r - 12
                cy = y0 + r + 12
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20))
            txt = f"{val:.2f}"
            bbox = d.textbbox((0, 0), txt, font=value_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            fill = (255, 255, 255) if val > 0.58 else (35, 35, 35)
            d.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), txt, fill=fill, font=value_font)
    for j, label in enumerate(labels):
        label_xy = (margin_left + j * cell + 10, margin_top + n * cell + 20)
        draw_rotated_label(img, label, label_xy, size=label_size, vector_draw=d)
    img.save(path, dpi=(DPI, DPI))
    save_vector_pdf(img.size, d.operations, pdf_path, dpi=DPI)


# --------------------------------------------------------------------------
# text outputs (Word methods and Markdown summary are written in Chinese)
# --------------------------------------------------------------------------
def set_docx_font(run, size: float | None = None, bold: bool | None = None) -> None:
    """Apply Arial (Latin and East Asian) to a python-docx run."""
    run.font.name = "Arial"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def add_paragraph(doc: Document, text: str, size: float = 11, bold: bool = False) -> None:
    """Append a single-run paragraph with the standard font."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_docx_font(run, size=size, bold=bold)


def write_methods_docx(method_docx: Path, n_samples: int, variables: list[str], focus_table: pd.DataFrame) -> None:
    """Write the methods description (Chinese) to a Word document."""
    doc = Document()
    styles = doc.styles
    for style_name in ["Normal", "Heading 1", "Heading 2"]:
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    styles["Normal"].font.size = Pt(11)
    doc.add_heading("方向性共检出覆盖率矩阵分析方法", level=1)
    doc.add_heading("样本与变量汇总", level=2)
    add_paragraph(
        doc,
        "以 Organ 和 Sample 列完全相同的记录定义为同一个生物样本。首先在每个样本内按作图变量汇总 Concentration：PFAS 类污染物保留原 Clusters 分组（PFAS-1 至 PFAS-5），非 PFAS 新兴污染物不再按 Clusters 细分，而是按 Category 大类求和。所有低于检测限的浓度已在原始表中删除，因此浓度大于 0 被定义为该变量在该样本中检出。",
    )
    add_paragraph(
        doc,
        f"本分析共纳入 {n_samples} 个样本和 {len(variables)} 个作图变量。分组按输入 Excel 的 Clusters 原值，PFAS-1、PFAS-2 为短链，PFAS-3、PFAS-4、PFAS-5 为长链/前体，不互换编号。为便于比较不同污染物类别间的方向性共检出关系，矩阵的行和列使用同一套变量；PFAS 簇与非 PFAS 大类共同进入矩阵展示。",
    )
    doc.add_heading("方向性共检出覆盖率", level=2)
    add_paragraph(
        doc,
        "对任意两个变量 i 和 j，定义方向性共检出覆盖率为 C(i|j) = n(Xi > 0 且 Xj > 0) / n(Xj > 0)。其中 Xi 和 Xj 分别表示变量 i 和 j 在样本中的汇总浓度。该指标表示在变量 j 被检出的样本中，变量 i 同时被检出的比例，因此 C(i|j) 与 C(j|i) 通常不相等。矩阵图中行变量为被覆盖变量，列变量为条件变量，单元格颜色和数字均表示 C(i|j)。由于该指标具有方向性，完整方阵用于同时展示两个方向的覆盖关系。",
    )
    doc.add_heading("显著性检验", level=2)
    add_paragraph(
        doc,
        f"为评价短链 PFAS 相关簇与其它污染物变量之间的共检出覆盖是否高于背景水平，对相应方向性覆盖率进行单侧精确二项检验。对于变量 i 和 j，令 n 为变量 j 被检出的样本数，k 为变量 i 和 j 同时被检出的样本数。在零假设 H0: P(i 检出 | j 检出) <= {COVERAGE_BASELINE:g} 下，计算 P[Binomial(n, {COVERAGE_BASELINE:g}) >= k]。",
    )
    doc.add_heading("多重比较校正", level=2)
    add_paragraph(
        doc,
        "由于矩阵分析涉及多个方向性覆盖检验，原始 P 值进一步采用 Benjamini-Hochberg 方法进行 false discovery rate (FDR) 校正。具体而言，将同一检验集合内的 m 个 P 值由小到大排序为 p(1) <= p(2) <= ... <= p(m)，并按其秩次计算调整后的显著性水平；对应的 q 值可理解为在给定显著性阈值下，被判定为显著的结果中预期假阳性发现所占比例。与控制 family-wise error rate 的方法相比，FDR 校正在多重比较场景下更适合用于筛选具有统计学支持的共检出覆盖关系。本文以 q < 0.05 作为 FDR 校正后的显著性标准，达到该阈值的单元格在矩阵中以黑点标注。",
    )
    doc.add_heading("图形输出", level=2)
    add_paragraph(
        doc,
        "矩阵图使用 Arial 字体绘制，PNG 图像以 1000 dpi 导出，同时输出矢量 PDF。矩阵中红色越深表示方向性共检出覆盖率越高；单元格内显示对应覆盖率数值；黑点表示该方向性覆盖关系经 FDR 校正后达到 q < 0.05。",
    )
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.name != "Arial":
                set_docx_font(run)
    doc.save(method_docx)


def write_summary(
    focus_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    excel: Path,
    fig_png: Path,
    fig_pdf: Path,
    method_docx: Path,
) -> Path:
    """Write the Markdown result summary (Chinese) and return its path."""
    focus_pair = pair_table[pair_table["is_focus_row_pair"]]
    non_focus_pair = pair_table[~pair_table["is_focus_row_pair"]]
    lines = [
        "# PFAS-1/PFAS-2 directional coverage plot",
        "",
        "- 矩阵指标：P(行变量检出 | 列变量检出)。",
        "- 分组按输入 Excel 的 Clusters 原值：PFAS-1、PFAS-2 为短链，PFAS-3、PFAS-4、PFAS-5 为长链/前体；不互换编号。",
        "- 变量排序：PFAS-1、PFAS-2 放在前两行；其它变量按检出样本数从少到多排列。",
        "- 图形为完整方向性覆盖方阵热图，不再绘制散点图或对角线分布。",
        "- 这样读图时，PFAS-1/PFAS-2 与其它类别会显示为高覆盖；其它-其它因为行变量通常是较少检出的类别，整体会相对低。",
        f"- 显著性方法：one-sided exact binomial coverage test，检验覆盖率是否显著高于 {COVERAGE_BASELINE:g}；FDR 只在 PFAS-1/PFAS-2 焦点比较内校正。",
        "- 图形字体为 Arial，PNG 按 1000 dpi 输出，PDF 为矢量格式。",
        "- 注意：这是方向性共检出覆盖显著性，不是 top25 高负荷 Fisher 富集显著性；top25 高负荷结果仅作为参考保存在 Excel。",
        "",
        "## 图面数值概览",
        "",
        f"- PFAS-1/PFAS-2 行与其它变量的中位覆盖概率：{focus_pair['P_row_present_given_column_present'].median():.3f}。",
        f"- 其它-其它比较的中位覆盖概率：{non_focus_pair['P_row_present_given_column_present'].median():.3f}。",
        "",
        "## PFAS-1 与其它变量",
        "",
        "| Other | P(PFAS-1 present given Other present) | coverage focus-FDR q | Jaccard | reference top25 FDR q |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in focus_table[focus_table["Focus"].eq("PFAS-1")].head(12).iterrows():
        lines.append(
            f"| `{row['Other']}` | {row['P_focus_present_given_other_present']:.3f} | "
            f"{row['coverage_focus_fdr_q']:.3g} | {row['presence_jaccard']:.3f} | {row['reference_top25_fdr_q']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## PFAS-2 与其它变量",
            "",
            "| Other | P(PFAS-2 present given Other present) | coverage focus-FDR q | Jaccard | reference top25 FDR q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in focus_table[focus_table["Focus"].eq("PFAS-2")].head(12).iterrows():
        lines.append(
            f"| `{row['Other']}` | {row['P_focus_present_given_other_present']:.3f} | "
            f"{row['coverage_focus_fdr_q']:.3g} | {row['presence_jaccard']:.3f} | {row['reference_top25_fdr_q']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- Excel：`{excel.relative_to(OUT_DIR)}`",
            f"- PNG：`{fig_png.relative_to(OUT_DIR)}`",
            f"- PDF：`{fig_pdf.relative_to(OUT_DIR)}`",
            f"- Word 方法：`{method_docx.relative_to(OUT_DIR)}`",
        ]
    )
    report = OUT_DIR / "analysis_summary_cn.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    """Run the full pipeline and write every output file."""
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    # 1. Load data and build the specimen x variable matrix.
    df = read_main_table(find_input_workbook())
    grouped, matrix, variables = build_matrix(df)
    raw0 = matrix[variables].apply(pd.to_numeric, errors="coerce").fillna(0)
    variables = ordered_variables(raw0, variables)
    raw = raw0[variables]
    log_values = np.log1p(raw)
    ann = annotations(df, matrix, variables)

    # 2. Reference: top-25% high-burden Fisher enrichment (Excel only).
    high_binary, high_thresholds = binary_sets(raw, "high_burden")
    high = enrichment_results(high_binary, ann, "high_burden")

    # 3. Directional coverage matrix and focus-row binomial tests.
    direction = directional_coverage_matrix(raw, variables)
    pair_table = pairwise_directional_table(raw, variables, direction)
    focus_tests = focus_coverage_tests(raw, variables)
    _, focus_sig = focus_coverage_matrices(focus_tests, variables)
    focus_table = focused_coverage_table(raw, variables, high).merge(focus_tests, on=["Focus", "Other"], how="left")
    focus_table = focus_table.sort_values(
        ["Focus", "coverage_focus_fdr_q", "P_focus_present_given_other_present"],
        ascending=[True, True, False],
    )
    counts = pd.DataFrame(
        {
            "Variable": variables,
            "n_present": raw.gt(0).sum().reindex(variables).astype(int).to_numpy(),
            "present_rate": raw.gt(0).mean().reindex(variables).to_numpy(),
        }
    )
    excel = OUT_DIR / "pfas1_pfas2_directional_coverage_analysis.xlsx"
    fig_png = FIG_DIR / "pfas1_pfas2_directional_coverage_matrix_1000dpi.png"
    fig_pdf = FIG_DIR / "pfas1_pfas2_directional_coverage_matrix_1000dpi.pdf"
    method_docx = OUT_DIR / "methods_directional_codetection_coverage.docx"

    # 4. Excel workbook (README sheet first, then intermediate and result tables).
    with pd.ExcelWriter(excel, engine="xlsxwriter") as writer:
        pd.DataFrame(
            {
                "item": [
                    "sample_definition",
                    "variable_definition",
                    "plot_metric",
                    "plot_order",
                    "significance_method",
                    "figure_format",
                    "important_note",
                ],
                "description": [
                    "One sample is one unique (Organ, Sample) pair.",
                    "Clusters follow Excel unchanged. Short-chain: PFAS-1/PFAS-2; long-chain/precursor: PFAS-3/PFAS-4/PFAS-5. Non-PFAS is summed by Category.",
                    "Matrix cells use directional coverage: P(row variable detected | column variable detected).",
                    "PFAS-1 and PFAS-2 are first; all other variables are sorted by detected sample count ascending.",
                    "Black dots use one-sided exact binomial coverage test against baseline 0.5, with FDR correction within PFAS-1/PFAS-2 focus comparisons.",
                    "Full directional coverage heatmap matrix; Arial font; PNG exported at 1000 dpi; vector PDF.",
                    "This is a focused co-detection coverage display, not a symmetric correlation coefficient or proof of top25 high-burden enrichment.",
                ],
            }
        ).to_excel(writer, sheet_name="README", index=False)
        grouped.to_excel(writer, sheet_name="long_grouped_data", index=False)
        matrix[["Organ", "Sample", "sample_id"] + variables].to_excel(writer, sheet_name="sample_variable_matrix", index=False)
        pd.concat([matrix[["Organ", "Sample", "sample_id"]], log_values], axis=1).to_excel(writer, sheet_name="log1p_plot_values", index=False)
        counts.to_excel(writer, sheet_name="presence_counts_order", index=False)
        ann.to_excel(writer, sheet_name="variable_annotations", index=False)
        direction.to_excel(writer, sheet_name="directional_coverage_matrix")
        pair_table.to_excel(writer, sheet_name="upper_triangle_pair_values", index=False)
        focus_tests.to_excel(writer, sheet_name="coverage_focus_binomial_FDR", index=False)
        focus_table.to_excel(writer, sheet_name="PFAS1_PFAS2_with_others", index=False)
        high.to_excel(writer, sheet_name="reference_top25_high_burden", index=False)
        high_thresholds.to_excel(writer, sheet_name="reference_top25_thresholds", index=False)
        workbook = writer.book
        header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "font_name": "Arial"})
        num_fmt = workbook.add_format({"num_format": "0.0000", "font_name": "Arial"})
        text_fmt = workbook.add_format({"font_name": "Arial"})
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_row(0, None, header)
            ws.set_column(0, 2, 26, text_fmt)
            ws.set_column(3, 80, 16, num_fmt)
        writer.sheets["README"].set_column(1, 1, 125, text_fmt)

    # 5. Figure, Word methods and Markdown summary.
    draw_matrix_figure(
        direction.loc[variables, variables],
        fig_png,
        "PFAS-1/PFAS-2 directional coverage",
        fig_pdf,
        focus_sig.loc[variables, variables],
    )
    write_methods_docx(method_docx, len(raw), variables, focus_table)
    report = write_summary(focus_table, pair_table, excel, fig_png, fig_pdf, method_docx)
    print(f"Excel: {excel}")
    print(f"Report: {report}")
    print(f"Figure PNG: {fig_png}")
    print(f"Figure PDF: {fig_pdf}")
    print(f"Methods Word: {method_docx}")
    print(pair_table.groupby("is_focus_row_pair")["P_row_present_given_column_present"].describe().to_string())


if __name__ == "__main__":
    main()
