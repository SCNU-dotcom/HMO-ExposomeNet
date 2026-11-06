"""Directed co-detection network for PFAS-1 / PFAS-2 short-chain clusters.

Builds a statistically-inferred directed co-occurrence network from organ-level
detection data, following the five-step design:

  Step 1  conditional coverage probabilities P(S|T) and P(T|S)
  Step 2  hypergeometric / Fisher one-sided enrichment + BH-FDR
  Step 3  directional asymmetry index A = P(S|T) - P(T|S), with an exact
          McNemar test and a subject-level cluster bootstrap CI
  Step 4  directed edges weighted by coverage P(S|T), coloured by A
  Step 5  node out-degree / in-degree / HITS hub-authority / betweenness

Two node resolutions are produced: all pollutants at cluster resolution (main
text) and individual compounds (appendix).  A high-burden (>=Q75) discretization
is run as a sensitivity analysis.

Arrow convention
----------------
An edge S -> T means "S covers T": whenever T is detected, S is almost always
detected too.  Edge weight is the coverage P(S detected | T detected) and the
edge is retained only when A = P(S|T) - P(T|S) > 0.  Because
A = n_both * (1/n_T - 1/n_S), sign(A) is fixed by the prevalence ordering, so
the arrow direction encodes prevalence dominance and the *inference* lives in
the magnitude of A and in the coverage / enrichment tests.

Outputs (``pfas_directional_network_outputs/``)
    Excel workbook with every intermediate table, network and centrality
    figures (1000-dpi PNG + vector PDF), Word methods and Markdown summary
    (both in Chinese).

Run: python analyze_pfas_directional_cooccurrence_network.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from network_figures import draw_centrality_figure, draw_network_figure
from gee_causal_core import PFAS_ORDER, PROJECT_DIR, SHORT_CHAIN, WORKBOOK, pfas_group

REQUIRED_COLUMNS = {"Sample", "Organ", "Category", "Clusters", "Name", "Concentration"}
FOCUS = SHORT_CHAIN.copy()   # hub nodes drawn at the centre of the network panels

# --- statistical parameters ---
COVERAGE_BASELINE = 0.5      # H0 for the one-sided exact binomial coverage test
COVERAGE_MIN = 0.90          # primary edge retention threshold on P(S|T)
COVERAGE_GRID = [0.70, 0.80, 0.90, 0.95]  # sensitivity grid for COVERAGE_MIN
FDR_ALPHA = 0.05             # BH-FDR significance level
N_BOOTSTRAP = 2000           # cluster-bootstrap resamples
BOOTSTRAP_SEED = 20260728    # fixed seed -> reproducible confidence intervals
COMPOUND_MIN_DETECTED = 18   # ~5% of 355 organ samples

OUT_DIR = PROJECT_DIR / "pfas_directional_network_outputs"
FIG_DIR = OUT_DIR / "figures"

# dataviz reference palette -- validated all-pairs, light surface #fcfcfb
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_INK = "#c3c2b7"
GROUP_COLORS = {
    "Short-chain PFAS focus": "#c0392b",          # deep red -- focus hubs
    "Other PFAS cluster": "#c0392b",
    "Long-chain PFAS cluster": "#c0392b",
    "Non-PFAS cluster": "#8c9199",
    "Focus PFAS compound": "#c0392b",
    "Other PFAS compound": "#c0392b",
    "Non-PFAS compound": "#8c9199",
}
# Node fills are per pollutant category -- see CATEGORY_COLORS in
# network_figures, which is the single source of truth for the figures.
# red sequential ramp for the asymmetry index A (light = low, dark = high)
SEQ_RED = ["#fad4d4", "#f5b8b8", "#f09090", "#ea6868",
           "#e34948", "#d03b3b", "#b83333", "#9e2a2a", "#822121", "#681919"]


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------
def find_input_workbook() -> Path:
    """Return the shared input workbook path, failing early if it is missing."""
    if not WORKBOOK.is_file():
        raise FileNotFoundError(f"Analysis workbook not found: {WORKBOOK}")
    return WORKBOOK


def read_main_table(path: Path) -> pd.DataFrame:
    """Read the long-format table, coerce concentrations and add specimen ids."""
    df = pd.read_excel(path, sheet_name=0)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    # Non-detects were removed from the source table; missing values count as 0.
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce").fillna(0)
    for col in ("Sample", "Organ", "Clusters", "Category", "Name"):
        df[col] = df[col].astype(str).str.strip()
    df["Category"] = df["Category"].replace({"others": "Others"})
    # 'subject' is the cluster-bootstrap resampling unit (the Sample field).
    df["subject"] = df["Sample"]
    df["sample_id"] = df["Organ"] + " | " + df["Sample"]
    return df


def cluster_group(var: str, category: str) -> str:
    """Reporting group of a cluster-level node."""
    if category != "PFAS":
        return "Non-PFAS cluster"
    return pfas_group(var, category)


def natural_key(text: str) -> tuple:
    """Sort 'Pest.-10' after 'Pest.-9' rather than lexically."""
    head, _, tail = str(text).rpartition("-")
    try:
        n = int(tail)
    except ValueError:
        return (str(text).lower(), 10 ** 9)
    return (head.lower(), n)


def build_cluster_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analysis unit = one (Organ, Sample) organ specimen.

    Every pollutant is kept at its own cluster resolution (PFAS-1..PFAS-5,
    Pest.-1.., Pharm.-1.., etc.) -- categories are never aggregated.
    """
    work = df.copy()
    work["variable"] = work["Clusters"]
    grouped = work.groupby(
        ["sample_id", "subject", "Organ", "variable"], as_index=False
    )["Concentration"].sum()
    values = grouped.pivot_table(
        index=["sample_id", "subject", "Organ"],
        columns="variable",
        values="Concentration",
        fill_value=0.0,
        aggfunc="sum",
    )
    cluster_category = work.drop_duplicates("variable").set_index("variable")["Category"]
    non_pfas = sorted(
        (v for v in values.columns if v not in PFAS_ORDER),
        key=lambda v: (str(cluster_category.get(v, "")).lower(), natural_key(v)),
    )
    order = [v for v in PFAS_ORDER if v in values.columns] + non_pfas
    values = values[order]
    meta = pd.DataFrame(
        {
            "variable": order,
            "category": [cluster_category.get(v, "") for v in order],
            "group": [cluster_group(v, cluster_category.get(v, "")) for v in order],
        }
    )
    members = work.groupby("variable")["Name"].agg(lambda s: "; ".join(sorted(set(s))))
    meta["members"] = meta["variable"].map(members).fillna("")
    meta["n_compounds"] = meta["variable"].map(
        work.groupby("variable")["Name"].nunique()
    ).fillna(0).astype(int)
    return values, meta


def build_compound_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Appendix resolution: individual compounds.

    Retains every PFAS-1 / PFAS-2 member plus any other compound detected in at
    least COMPOUND_MIN_DETECTED organ specimens.
    """
    work = df.copy()
    values = work.pivot_table(
        index=["sample_id", "subject", "Organ"],
        columns="Name",
        values="Concentration",
        fill_value=0.0,
        aggfunc="sum",
    )
    detected = values.gt(0).sum()
    info = work.drop_duplicates("Name").set_index("Name")[["Category", "Clusters"]]
    focus_members = [n for n in values.columns if info.loc[n, "Clusters"] in FOCUS]
    others = [
        n for n in values.columns
        if n not in focus_members and detected[n] >= COMPOUND_MIN_DETECTED
    ]

    def sort_key(name: str) -> tuple:
        cl = info.loc[name, "Clusters"]
        rank = FOCUS.index(cl) if cl in FOCUS else (2 if info.loc[name, "Category"] == "PFAS" else 3)
        return (rank, -int(detected[name]), name)

    keep = sorted(focus_members + others, key=sort_key)
    values = values[keep]
    meta = pd.DataFrame(
        {
            "variable": keep,
            "category": [info.loc[n, "Category"] for n in keep],
            "cluster": [info.loc[n, "Clusters"] for n in keep],
        }
    )
    meta["group"] = np.where(
        meta["cluster"].isin(FOCUS),
        "Focus PFAS compound",
        np.where(meta["category"].eq("PFAS"), "Other PFAS compound", "Non-PFAS compound"),
    )
    meta["members"] = meta["variable"]
    return values, meta


# --------------------------------------------------------------------------
# exact tests (self-contained, no scipy dependency)
# --------------------------------------------------------------------------
def logcomb(n: int, k: int) -> float:
    """log(n choose k); -inf outside the support."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _logsumexp(terms: list[float]) -> float:
    """Numerically stable log(sum(exp(terms)))."""
    finite = [t for t in terms if t > -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(t - m) for t in finite))


def hypergeom_sf(k: int, n_a: int, n_b: int, n: int) -> float:
    """P[X >= k] for X ~ Hypergeom(n, n_a, n_b); one-sided Fisher exact p."""
    upper = min(n_a, n_b)
    if k <= max(0, n_a + n_b - n):
        return 1.0
    if k > upper:
        return 0.0
    terms = [
        logcomb(n_a, i) + logcomb(n - n_a, n_b - i) - logcomb(n, n_b)
        for i in range(k, upper + 1)
        if 0 <= n_b - i <= n - n_a
    ]
    return min(1.0, math.exp(_logsumexp(terms))) if terms else 1.0


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P[Binomial(n, p) >= k]."""
    if n <= 0:
        return math.nan
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    terms = [logcomb(n, i) + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k, n + 1)]
    return min(1.0, math.exp(_logsumexp(terms)))


def exact_mcnemar_greater(b: int, c: int) -> float:
    """One-sided exact McNemar: P[Binom(b+c, 0.5) >= b].

    b = #samples with source detected but target absent (source-only)
    c = #samples with target detected but source absent (target-only)
    A small p means the source marginal significantly exceeds the target's,
    i.e. the asymmetry index A is significantly positive.
    """
    if b + c == 0:
        return 1.0
    return binom_sf_ge(b, b + c, 0.5)


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (NaN p-values are treated as 1)."""
    p = np.asarray(pd.Series(pvalues).fillna(1.0), dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = m - rank + 1
        prev = min(prev, p[idx] * m / true_rank)
        q[idx] = min(1.0, prev)
    return q


# --------------------------------------------------------------------------
# Steps 1-3: directed edge statistics
# --------------------------------------------------------------------------
def binarize(values: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """mode='detection': >0 (limit of detection). mode='high_burden': >0 and >=Q75."""
    binary = pd.DataFrame(index=values.index)
    rows = []
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").fillna(0.0)
        if mode == "detection":
            threshold = 0.0
            binary[col] = x > 0
        elif mode == "high_burden":
            positive = x[x > 0]
            threshold = float(positive.quantile(0.75)) if len(positive) else math.inf
            binary[col] = (x > 0) & (x >= threshold)
        else:
            raise ValueError(f"Unknown binarization mode: {mode}")
        rows.append(
            {
                "variable": col,
                "mode": mode,
                "threshold": threshold,
                "n_positive": int(binary[col].sum()),
                "positive_rate": float(binary[col].mean()),
            }
        )
    return binary.astype(bool), pd.DataFrame(rows)


def bootstrap_asymmetry(
    binary: pd.DataFrame, subjects: np.ndarray, pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Subject-level cluster bootstrap CI for A and for coverage P(S|T).

    The resampling groups follow the input Sample field, as in the original
    analysis. Group counts therefore depend on the workbook's Sample coding.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    uniq = np.unique(subjects)
    idx_by_subject = [np.flatnonzero(subjects == s) for s in uniq]
    arr = {c: binary[c].to_numpy() for c in binary.columns}
    cov = {p: np.empty(N_BOOTSTRAP) for p in pairs}
    asym = {p: np.empty(N_BOOTSTRAP) for p in pairs}
    for b in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_subject[i] for i in pick])
        for src, tgt in pairs:
            xs, xt = arr[src][rows], arr[tgt][rows]
            n_s, n_t = xs.sum(), xt.sum()
            both = np.logical_and(xs, xt).sum()
            p_st = both / n_t if n_t else np.nan
            p_ts = both / n_s if n_s else np.nan
            cov[(src, tgt)][b] = p_st
            asym[(src, tgt)][b] = p_st - p_ts
    out = []
    for src, tgt in pairs:
        a, c = asym[(src, tgt)], cov[(src, tgt)]
        out.append(
            {
                "source": src,
                "target": tgt,
                "A_boot_mean": float(np.nanmean(a)),
                "A_boot_ci_low": float(np.nanpercentile(a, 2.5)),
                "A_boot_ci_high": float(np.nanpercentile(a, 97.5)),
                "A_boot_frac_gt_0": float(np.nanmean(a > 0)),
                "coverage_boot_ci_low": float(np.nanpercentile(c, 2.5)),
                "coverage_boot_ci_high": float(np.nanpercentile(c, 97.5)),
            }
        )
    return pd.DataFrame(out)


def directed_edge_table(
    binary: pd.DataFrame, meta: pd.DataFrame, subjects: np.ndarray, mode: str,
    bootstrap: bool = True,
) -> pd.DataFrame:
    """All ordered pairs with coverage, enrichment, asymmetry and FDR."""
    variables = list(binary.columns)
    group = meta.set_index("variable")["group"].to_dict()
    category = meta.set_index("variable")["category"].to_dict()
    n = len(binary)
    arr = {c: binary[c].to_numpy() for c in variables}
    counts = {c: int(arr[c].sum()) for c in variables}
    rows = []
    for src in variables:
        for tgt in variables:
            if src == tgt:
                continue
            n_s, n_t = counts[src], counts[tgt]
            both = int(np.logical_and(arr[src], arr[tgt]).sum())
            src_only, tgt_only = n_s - both, n_t - both
            neither = n - both - src_only - tgt_only
            # Step 1: conditional coverage probabilities
            p_src_given_tgt = both / n_t if n_t else np.nan
            p_tgt_given_src = both / n_s if n_s else np.nan
            # Step 2: one-sided hypergeometric (Fisher) enrichment
            expected = n_s * n_t / n if n else np.nan
            fold = both / expected if expected and expected > 0 else np.nan
            odds = (
                (both * neither) / (src_only * tgt_only)
                if src_only > 0 and tgt_only > 0
                else (math.inf if both * neither > 0 else np.nan)
            )
            rows.append(
                {
                    "mode": mode,
                    "source": src,
                    "target": tgt,
                    "source_group": group.get(src, ""),
                    "target_group": group.get(tgt, ""),
                    "source_category": category.get(src, ""),
                    "target_category": category.get(tgt, ""),
                    "N_samples": n,
                    "n_source": n_s,
                    "n_target": n_t,
                    "n_both": both,
                    "n_source_only": src_only,
                    "n_target_only": tgt_only,
                    "n_neither": neither,
                    "coverage_P_source_given_target": p_src_given_tgt,
                    "reverse_P_target_given_source": p_tgt_given_src,
                    "asymmetry_index_A": p_src_given_tgt - p_tgt_given_src,
                    "n_both_expected": expected,
                    "fold_enrichment": fold,
                    "log2_fold_enrichment": math.log2(fold) if fold and fold > 0 else np.nan,
                    "odds_ratio": odds,
                    "p_fisher_greater": hypergeom_sf(both, n_s, n_t, n),
                    "p_coverage_gt_baseline": binom_sf_ge(both, n_t, COVERAGE_BASELINE),
                    "p_mcnemar_asymmetry": exact_mcnemar_greater(src_only, tgt_only),
                }
            )
    edges = pd.DataFrame(rows)
    edges["q_fisher"] = bh_fdr(edges["p_fisher_greater"])
    edges["q_coverage"] = bh_fdr(edges["p_coverage_gt_baseline"])
    edges["q_mcnemar"] = bh_fdr(edges["p_mcnemar_asymmetry"])
    if bootstrap:
        pairs = list(zip(edges["source"], edges["target"]))
        edges = edges.merge(
            bootstrap_asymmetry(binary, subjects, pairs), on=["source", "target"], how="left"
        )
    else:
        for col in ("A_boot_mean", "A_boot_ci_low", "A_boot_ci_high", "A_boot_frac_gt_0",
                    "coverage_boot_ci_low", "coverage_boot_ci_high"):
            edges[col] = np.nan
    # Step 3/4: edge retention
    edges["A_significant"] = (edges["q_mcnemar"] < FDR_ALPHA) & (edges["A_boot_ci_low"] > 0)
    edges["coverage_significant"] = edges["q_coverage"] < FDR_ALPHA
    edges["enrichment_significant"] = edges["q_fisher"] < FDR_ALPHA
    edges["is_directed_edge"] = (
        (edges["coverage_P_source_given_target"] >= COVERAGE_MIN)
        & (edges["asymmetry_index_A"] > 0)
        & edges["A_significant"]
        & edges["coverage_significant"]
    )
    edges["is_enriched_directed_edge"] = edges["is_directed_edge"] & edges["enrichment_significant"]
    return edges.sort_values(
        ["is_directed_edge", "coverage_P_source_given_target", "asymmetry_index_A"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 5: node centralities
# --------------------------------------------------------------------------
def hits_scores(nodes: list[str], edges: list[tuple[str, str, float]],
                iterations: int = 200, tol: float = 1e-12) -> tuple[dict, dict]:
    """Weighted HITS. hub = points to many good authorities (a 'coverer');
    authority = pointed to by many good hubs (a 'covered' node)."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n))
    for src, tgt, w in edges:
        adj[idx[src], idx[tgt]] += w
    hub = np.ones(n) / math.sqrt(n)
    auth = np.ones(n) / math.sqrt(n)
    for _ in range(iterations):
        new_auth = adj.T @ hub
        new_hub = adj @ new_auth
        na, nh = np.linalg.norm(new_auth), np.linalg.norm(new_hub)
        new_auth = new_auth / na if na > 0 else new_auth
        new_hub = new_hub / nh if nh > 0 else new_hub
        if np.abs(new_hub - hub).sum() + np.abs(new_auth - auth).sum() < tol:
            hub, auth = new_hub, new_auth
            break
        hub, auth = new_hub, new_auth
    return (
        {v: float(hub[idx[v]]) for v in nodes},
        {v: float(auth[idx[v]]) for v in nodes},
    )


def betweenness(nodes: list[str], edges: list[tuple[str, str, float]]) -> dict:
    """Brandes' betweenness centrality for unweighted directed graphs,
    normalized by (n-1)(n-2)."""
    succ = {v: [] for v in nodes}
    for src, tgt, _ in edges:
        succ[src].append(tgt)
    bc = {v: 0.0 for v in nodes}
    for s in nodes:
        stack, pred = [], {v: [] for v in nodes}
        sigma = {v: 0.0 for v in nodes}
        dist = {v: -1 for v in nodes}
        sigma[s], dist[s] = 1.0, 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in succ[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {v: 0.0 for v in nodes}
        for w in reversed(stack):
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]) if sigma[w] else 0.0
            if w != s:
                bc[w] += delta[w]
    n = len(nodes)
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    return {v: bc[v] * scale for v in nodes}


def node_metrics(edges: pd.DataFrame, meta: pd.DataFrame, binary: pd.DataFrame,
                 mode: str, edge_flag: str = "is_directed_edge") -> pd.DataFrame:
    """Degree, strength, HITS and betweenness for every node of the retained graph."""
    nodes = list(binary.columns)
    sel = edges[edges[edge_flag]]
    triples = list(
        zip(sel["source"], sel["target"], sel["coverage_P_source_given_target"])
    )
    hub, auth = hits_scores(nodes, triples)
    bc = betweenness(nodes, triples)
    out_deg = sel.groupby("source").size().reindex(nodes).fillna(0).astype(int)
    in_deg = sel.groupby("target").size().reindex(nodes).fillna(0).astype(int)
    out_w = sel.groupby("source")["coverage_P_source_given_target"].sum().reindex(nodes).fillna(0.0)
    in_w = sel.groupby("target")["coverage_P_source_given_target"].sum().reindex(nodes).fillna(0.0)
    mean_out_A = sel.groupby("source")["asymmetry_index_A"].mean().reindex(nodes)
    info = meta.set_index("variable")
    n_total = len(binary)
    df = pd.DataFrame(
        {
            "mode": mode,
            "node": nodes,
            "group": [info.loc[v, "group"] for v in nodes],
            "category": [info.loc[v, "category"] for v in nodes],
            "n_detected": [int(binary[v].sum()) for v in nodes],
            "detection_rate": [float(binary[v].mean()) for v in nodes],
            "out_degree": out_deg.to_numpy(),
            "in_degree": in_deg.to_numpy(),
            "out_strength": out_w.to_numpy(),
            "in_strength": in_w.to_numpy(),
            "mean_outgoing_A": mean_out_A.to_numpy(),
            "hub_score": [hub[v] for v in nodes],
            "authority_score": [auth[v] for v in nodes],
            "betweenness": [bc[v] for v in nodes],
        }
    )
    df["net_degree_out_minus_in"] = df["out_degree"] - df["in_degree"]
    df["N_samples"] = n_total
    return df.sort_values(
        ["out_degree", "hub_score", "n_detected"], ascending=[False, False, False]
    ).reset_index(drop=True)


def coverage_threshold_sensitivity(edges: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Edge counts and focus out-degree across the coverage-threshold grid."""
    rows = []
    base = (
        (edges["asymmetry_index_A"] > 0) & edges["A_significant"] & edges["coverage_significant"]
    )
    for thr in COVERAGE_GRID:
        sel = edges[base & (edges["coverage_P_source_given_target"] >= thr)]
        counts = sel.groupby("source").size()
        rows.append(
            {
                "mode": mode,
                "coverage_threshold": thr,
                "n_directed_edges": int(len(sel)),
                "n_enriched_edges": int(sel["enrichment_significant"].sum()),
                "n_source_nodes": int(sel["source"].nunique()),
                "PFAS-1_out_degree": int(counts.get("PFAS-1", 0)),
                "PFAS-2_out_degree": int(counts.get("PFAS-2", 0)),
                "PFAS-1_in_degree": int((sel["target"] == "PFAS-1").sum()),
                "PFAS-2_in_degree": int((sel["target"] == "PFAS-2").sum()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# methods document (Chinese)
# --------------------------------------------------------------------------
def write_methods_docx(path: Path, n_samples: int, n_subjects: int,
                       nodes_df: pd.DataFrame, edges: pd.DataFrame,
                       hb_nodes: pd.DataFrame) -> None:
    """Write the methods description (Chinese) to a Word document."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    def para(doc, text, size=10.5, bold=False):
        """Append one paragraph using Arial / SimSun."""
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "Arial"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        return p

    sel = edges[edges["is_directed_edge"]]
    focus_out = {f: int((sel["source"] == f).sum()) for f in FOCUS}
    focus_in = {f: int((sel["target"] == f).sum()) for f in FOCUS}
    pfas1_n = int(nodes_df.set_index("node").loc["PFAS-1", "n_detected"])

    doc = Document()
    para(doc, "有向共检出网络分析方法", 14, True)

    para(doc, "1. 数据准备与二元化", 12, True)
    para(
        doc,
        f"数据来自 {WORKBOOK.name} 的首个工作表。分析单元为唯一的（Organ，Sample）组合，"
        f"共 {n_samples} 份样本，按原代码对应 {n_subjects} 个 Sample 分组。Clusters 编号"
        "直接沿用 Excel，PFAS-1、PFAS-2 为短链，PFAS-3、PFAS-4、PFAS-5 为长链/前体。节点采用两种"
        f"分辨率：正文使用 {len(nodes_df)} 个簇级节点（PFAS 保留 PFAS-1 至 PFAS-5，"
        f"非 PFAS 污染物保留各 Clusters 子簇，Category 仅用于注释）；附图使用具体"
        f"物质节点（保留 PFAS-1 与 PFAS-2 的全部成员化合物，并纳入检出样本数不少于 "
        f"{COMPOUND_MIN_DETECTED} 份（约 5%）的其它化合物）。以检出限为界将浓度转为 0/1："
        f"浓度大于 0 记为检出。敏感性分析采用定量浓度离散化，即以各变量阳性样本浓度的第 "
        f"75 百分位数（Q75）为界定义“高负荷检出”。",
    )

    para(doc, "2. 方向性边定义与统计推断", 12, True)
    para(
        doc,
        "对每一对污染物 (S, T) 计算条件共检出概率：coverage = P(S 检出 | T 检出)，"
        "反向概率 = P(T 检出 | S 检出)。边的方向定义为 S → T，表示“S 覆盖 T”：即 T 被"
        "检出时 S 几乎必然同时被检出。",
    )
    para(
        doc,
        "（1）富集检验：以超几何分布（等价于单侧 Fisher 精确检验）检验 S 与 T 的共检出"
        "数是否显著高于在边际检出率下的随机期望，P[X ≥ n_both]，X ~ Hypergeom(N, n_S, n_T)。",
    )
    para(
        doc,
        f"（2）覆盖率检验：以单侧精确二项检验判断 coverage 是否显著高于基线 "
        f"{COVERAGE_BASELINE:g}，即在 H0: P(S|T) ≤ {COVERAGE_BASELINE:g} 下计算 "
        f"P[Binomial(n_T, {COVERAGE_BASELINE:g}) ≥ n_both]。",
    )
    para(
        doc,
        "（3）方向性不对称指数：A = P(S|T) − P(T|S)。因 A = n_both × (1/n_T − 1/n_S)，"
        "其符号完全由两节点的检出普遍度顺序决定，故方向本身不携带独立的统计信息，"
        "统计推断由 A 的幅度及上述覆盖率与富集检验承担。A > 0 的统计显著性以配对的"
        "单侧精确 McNemar 检验评价（比较 S 独有与 T 独有的样本数）。整群 bootstrap"
        f"沿用原代码以 Sample 为分组单位（{N_BOOTSTRAP} 次重抽样），给出 A 与 coverage "
        "的 95% 置信区间。Sample 的唯一值数量不直接视为已核实的独立个体数。",
    )
    para(
        doc,
        f"以上三类 p 值分别在全部有序节点对内使用 Benjamini–Hochberg 法进行 FDR 多重"
        f"校正。保留为有向边的条件为：coverage ≥ {COVERAGE_MIN:g}、A > 0、"
        f"McNemar FDR q < {FDR_ALPHA:g} 且 A 的 bootstrap 95% 置信区间下限大于 0、"
        f"覆盖率检验 FDR q < {FDR_ALPHA:g}。同时满足富集检验 FDR q < {FDR_ALPHA:g} 的边"
        f"另行标注为“富集显著边”。覆盖率阈值在 "
        f"{', '.join(f'{t:g}' for t in COVERAGE_GRID)} 上做敏感性分析。",
    )

    para(doc, "3. 网络可视化与节点中心性", 12, True)
    para(
        doc,
        "网络图以每个焦点枢纽为中心绘制放射状星形图（panel a/b）：枢纽居中，其覆盖的"
        "目标节点分布在外围环上。边的线宽与 coverage 成正比，边的颜色按不对称指数 A 的"
        "大小着色（单色红色渐变），节点颜色对应污染物大类（每个节点圆内直接标注名称与"
        "coverage 数值，图例重复该配色映射）。节点半径由检出样本数驱动，并在不与相邻"
        "节点相撞的前提下取到最大值：环上节点数超过 12 个时采用内外双半径交错排布，"
        "每个圆的半径上限由其与最近邻圆心距离的一半（减去固定间隙）逐对确定。"
        "内环半径与边的弯曲度联合求解：交错排布会使内环节点落在相邻外环节点连线的"
        "弧线上从而遮挡该边，因此按候选内环半径（0.70–0.82R）依次尝试，取第一个能"
        "保证所有边的可见段与任何非目标节点圆保持间隙的组合，使边不被节点遮挡的同时"
        "保留放射状弧线的方向感。"
        "节点中心性包括出度、入度、加权出强度与入强度、HITS 枢纽值（hub score，覆盖"
        "他者的能力）与权威值（authority score，被他者覆盖的程度），以及基于 Brandes "
        "算法的中介中心性。",
    )
    para(
        doc,
        "全部图形字体统一为 Arial，PNG 以 1000 dpi 导出（栅格化倍率 5，网络图约 "
        "7.5 × 4.4 英寸），并以 1000 dpi 导出参数同时输出矢量 PDF。",
    )

    para(doc, "4. 主要结果", 12, True)
    para(
        doc,
        f"在簇级节点网络中共保留 {len(sel)} 条有向边，其中 "
        f"{int(sel['is_enriched_directed_edge'].sum())} 条同时满足富集显著。"
        f"PFAS-1 出度为 {focus_out['PFAS-1']}、入度为 {focus_in['PFAS-1']}；"
        f"PFAS-2 出度为 {focus_out['PFAS-2']}、入度为 {focus_in['PFAS-2']}，"
        f"与短链 PFAS 作为共检出网络上游覆盖节点的预期一致。",
    )
    para(
        doc,
        "敏感性分析中，以 Q75 定义的“高负荷检出”在所有覆盖率阈值下均无保留边。"
        f"Q75 二元化后各变量阳性样本数为 {hb_nodes['n_detected'].min()}–"
        f"{hb_nodes['n_detected'].max()} 份，"
        "此时即使 coverage 达到 1.00，单侧精确二项检验的未校正 p 也仅为 0.5^n_T，"
        f"在 {len(edges)} 个有序节点对上做 FDR 校正后未保留有向边。这反映高负荷子集的检验功效"
        "限制，而非共检出结构消失；主结果依赖于以检出限为界的二元化。",
    )
    para(
        doc,
        f"需要说明的是，PFAS-1 在 {pfas1_n}/{n_samples} 份样本中被检出"
        f"（{pfas1_n / n_samples:.1%}），其边际检出率已接近"
        "上限，因此超几何富集检验对该节点存在天花板效应；"
        "该节点的方向性证据来自覆盖率检验与不对称指数，而非富集检验。本网络刻画的是"
        "共检出的条件覆盖结构，不能解释为因果传递或暴露来源的方向性推断。",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class Level:
    """One node resolution: cluster-level (main text) or compound-level (appendix)."""
    key: str
    label: str
    values: pd.DataFrame
    meta: pd.DataFrame


def readme_frame(values: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    """README sheet describing every analysis choice in the exported workbook."""
    n_samples = len(values)
    n_groups = values.index.get_level_values("subject").nunique()
    pfas1_n = int(nodes_df.set_index("node").loc["PFAS-1", "n_detected"])
    return pd.DataFrame(
        {
            "item": [
                "analysis_unit",
                "node_resolution",
                "binarization",
                "edge_direction",
                "edge_weight",
                "edge_color",
                "step2_enrichment_test",
                "coverage_test",
                "step3_asymmetry",
                "clustering_correction",
                "multiple_testing",
                "edge_retention",
                "step5_centrality",
                "sensitivity",
                "ceiling_effect_caveat",
                "interpretation_limit",
            ],
            "description": [
                f"Input: {WORKBOOK.name}; first worksheet. One specimen = one unique (Organ, Sample) pair; {n_samples} specimens and {n_groups} Sample groups. Sample groups are not independently verified person IDs.",
                "Clusters follow Excel unchanged. Short-chain focus: PFAS-1/PFAS-2; long-chain/precursor: PFAS-3/PFAS-4/PFAS-5. Main text uses cluster resolution; appendix uses individual compounds.",
                "Detection: concentration > 0 (limit of detection). Sensitivity: high burden = >0 and >= Q75 of positive values.",
                "Edge S -> T means 'S covers T': when T is detected, S is almost always detected as well.",
                "Edge weight = coverage = P(S detected | T detected).",
                "Edge color = asymmetry index A = P(S|T) - P(T|S), single-hue red sequential ramp.",
                "One-sided hypergeometric / Fisher exact test on n_both vs the marginal-rate expectation.",
                "One-sided exact binomial test of coverage against baseline 0.5.",
                "A = P(S|T) - P(T|S) = n_both * (1/n_T - 1/n_S); sign is fixed by prevalence ordering, so inference rests on the magnitude of A plus the coverage and enrichment tests. Significance of A>0 via one-sided exact McNemar.",
                f"Original grouping rule retained: cluster bootstrap over {n_groups} unique Sample groups ({N_BOOTSTRAP} resamples) gives 95% CI for A and coverage.",
                "Benjamini-Hochberg FDR within all ordered node pairs, separately for each of the three test families.",
                "coverage >= 0.90 AND A > 0 AND McNemar q < 0.05 AND bootstrap A CI lower bound > 0 AND coverage-test q < 0.05. Edges also passing enrichment q < 0.05 are flagged separately.",
                "Out-degree, in-degree, out/in strength, HITS hub (coverer) and authority (covered) scores, Brandes betweenness.",
                "Coverage threshold grid 0.70/0.80/0.90/0.95 and the high-burden (Q75) discretization.",
                f"PFAS-1 is detected in {pfas1_n}/{n_samples} specimens ({pfas1_n / n_samples:.1%}); high marginal detection prevalence limits the enrichment test. Directional coverage and enrichment are distinct tests.",
                "This is a conditional co-detection coverage structure, not causal transfer and not a directional inference about exposure sources.",
            ],
        }
    )


def run_level(level: Level, subjects: np.ndarray) -> dict:
    """Primary (detection) and sensitivity (high-burden) analysis for one level."""
    binary, thresholds = binarize(level.values, "detection")
    edges = directed_edge_table(binary, level.meta, subjects, "detection")
    nodes = node_metrics(edges, level.meta, binary, "detection")
    grid = coverage_threshold_sensitivity(edges, "detection")

    hb_binary, hb_thresholds = binarize(level.values, "high_burden")
    hb_edges = directed_edge_table(hb_binary, level.meta, subjects, "high_burden")
    hb_nodes = node_metrics(hb_edges, level.meta, hb_binary, "high_burden")
    hb_grid = coverage_threshold_sensitivity(hb_edges, "high_burden")
    return {
        "binary": binary,
        "thresholds": pd.concat([thresholds, hb_thresholds], ignore_index=True),
        "edges": edges,
        "nodes": nodes,
        "grid": pd.concat([grid, hb_grid], ignore_index=True),
        "hb_edges": hb_edges,
        "hb_nodes": hb_nodes,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_summary(path: Path, n_samples: int, n_subjects: int,
                  cluster: dict, compound: dict) -> None:
    """Write the Markdown result summary (Chinese)."""
    sel = cluster["edges"][cluster["edges"]["is_directed_edge"]]
    nodes = cluster["nodes"]
    lines = [
        "# PFAS-1 / PFAS-2 有向共检出网络分析结果",
        "",
        f"- 输入数据：{WORKBOOK.name}（首个工作表）。",
        "- 分组按 Excel 的 Clusters 原值：PFAS-1、PFAS-2 为短链，PFAS-3、PFAS-4、PFAS-5 为长链/前体；不互换编号。",
        f"- 分析单元：{n_samples} 份器官样本，沿用原代码对应 {n_subjects} 个 Sample 分组。",
        "- Sample 分组数量取决于输入编码，不直接视为已核实的独立个体数。",
        f"- 簇级节点网络：{len(nodes)} 个节点，{len(sel)} 条有向边"
        f"（其中富集检验亦显著者 {int(sel['is_enriched_directed_edge'].sum())} 条）。",
        f"- 边定义：coverage = P(源|靶) ≥ {COVERAGE_MIN:g}，A > 0，"
        f"McNemar FDR q < {FDR_ALPHA:g} 且 A 的整群 bootstrap 95% CI 下限 > 0，"
        f"覆盖率检验 FDR q < {FDR_ALPHA:g}。",
        "",
        "## 焦点节点中心性",
        "",
        "| 节点 | 检出数 | 检出率 | 出度 | 入度 | 枢纽值 | 权威值 | 中介中心性 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in nodes.iterrows():
        lines.append(
            f"| {r['node']} | {int(r['n_detected'])} | {r['detection_rate']:.3f} | "
            f"{int(r['out_degree'])} | {int(r['in_degree'])} | {r['hub_score']:.4f} | "
            f"{r['authority_score']:.4f} | {r['betweenness']:.4f} |"
        )
    lines += ["", "## PFAS-1 / PFAS-2 的出边", "",
              "| 源 | 靶 | coverage | 反向 | A | A 的 95% CI | 覆盖率 q | 富集 q |",
              "|---|---|---|---|---|---|---|---|"]
    focus_edges = sel[sel["source"].isin(FOCUS)].sort_values(
        ["source", "coverage_P_source_given_target"], ascending=[True, False]
    )
    for _, r in focus_edges.iterrows():
        lines.append(
            f"| {r['source']} | {r['target']} | "
            f"{r['coverage_P_source_given_target']:.3f} | "
            f"{r['reverse_P_target_given_source']:.3f} | {r['asymmetry_index_A']:.3f} | "
            f"[{r['A_boot_ci_low']:.3f}, {r['A_boot_ci_high']:.3f}] | "
            f"{r['q_coverage']:.3g} | {r['q_fisher']:.3g} |"
        )
    comp_sel = compound["edges"][compound["edges"]["is_directed_edge"]]
    lines += [
        "",
        "## 敏感性分析",
        "",
        _markdown_table(cluster["grid"]),
        "",
        f"- 高负荷（Q75）离散化的阳性样本数为 {cluster['hb_nodes']['n_detected'].min()}–"
        f"{cluster['hb_nodes']['n_detected'].max()} 份；各覆盖率阈值下的保留边数见上表。"
        f"检验在 {len(cluster['edges'])} 个有序节点对内分别进行 FDR 校正，阈值 q < {FDR_ALPHA:g}。",
        "",
        f"- 物质级网络（附图）：{len(compound['nodes'])} 个节点，{len(comp_sel)} 条有向边。",
        "",
        "## 重要说明",
        "",
        "- A = P(Y|X) − P(X|Y) = n_both × (1/n_Y − 1/n_X)，其符号完全由两节点检出普遍度"
        "顺序决定，方向本身不含独立统计信息；统计推断由 A 的幅度与覆盖率/富集检验承担。",
        f"- PFAS-1 检出率 {nodes.set_index('node').loc['PFAS-1', 'detection_rate']:.1%}"
        f"（{int(nodes.set_index('node').loc['PFAS-1', 'n_detected'])}/{n_samples}），"
        "高边际检出率限制超几何富集检验的检验功效；方向性覆盖与富集检验应分别解读。",
        f"- 中介中心性非零的簇级节点数：{int(nodes['betweenness'].gt(0).sum())}。",
        "- 本网络为条件共检出覆盖结构，不可解释为因果传递或暴露来源方向。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_workbook(path: Path, cluster: dict, compound: dict,
                    cluster_level: Level, compound_level: Level) -> None:
    """Export all node, edge, threshold and sensitivity tables to one workbook."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        readme_frame(cluster_level.values, cluster["nodes"]).to_excel(writer, sheet_name="README", index=False)
        cluster_level.meta.to_excel(writer, sheet_name="node_annotations_category", index=False)
        cluster["binary"].astype(int).reset_index().to_excel(
            writer, sheet_name="binary_matrix_category", index=False
        )
        cluster["thresholds"].to_excel(writer, sheet_name="binarization_thresholds", index=False)
        cluster["edges"].to_excel(writer, sheet_name="edges_all_pairs_category", index=False)
        cluster["edges"][cluster["edges"]["is_directed_edge"]].to_excel(
            writer, sheet_name="edges_retained_category", index=False
        )
        cluster["nodes"].to_excel(writer, sheet_name="node_centrality_category", index=False)
        cluster["grid"].to_excel(writer, sheet_name="threshold_sensitivity", index=False)
        cluster["hb_edges"][cluster["hb_edges"]["is_directed_edge"]].to_excel(
            writer, sheet_name="edges_retained_highburden", index=False
        )
        cluster["hb_nodes"].to_excel(writer, sheet_name="node_centrality_highburden", index=False)
        compound_level.meta.to_excel(writer, sheet_name="node_annotations_compound", index=False)
        compound["edges"].to_excel(writer, sheet_name="edges_all_pairs_compound", index=False)
        compound["edges"][compound["edges"]["is_directed_edge"]].to_excel(
            writer, sheet_name="edges_retained_compound", index=False
        )
        compound["nodes"].to_excel(writer, sheet_name="node_centrality_compound", index=False)

        book = writer.book
        header = book.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1,
                                  "font_name": "Arial"})
        num = book.add_format({"num_format": "0.0000", "font_name": "Arial"})
        text = book.add_format({"font_name": "Arial"})
        for name, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            ws.set_row(0, None, header)
            ws.set_column(0, 3, 24, text)
            ws.set_column(4, 60, 15, num)
        writer.sheets["README"].set_column(1, 1, 130, text)


def main() -> None:
    """Run the full pipeline and write every output file."""
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and build both node resolutions.
    df = read_main_table(find_input_workbook())

    c_values, c_meta = build_cluster_matrix(df)
    p_values, p_meta = build_compound_matrix(df)
    subjects = c_values.index.get_level_values("subject").to_numpy()
    n_samples = len(c_values)
    n_subjects = int(pd.Series(subjects).nunique())

    # 2. Edge statistics, retention and centralities for each resolution.
    cluster_level = Level("cluster", "all pollutant clusters", c_values, c_meta)
    compound_level = Level("compound", "individual compounds", p_values, p_meta)
    cluster = run_level(cluster_level, subjects)
    compound = run_level(compound_level, p_values.index.get_level_values("subject").to_numpy())

    # 3. Excel workbook.
    excel = OUT_DIR / "pfas_directional_cooccurrence_network.xlsx"
    export_workbook(excel, cluster, compound, cluster_level, compound_level)

    # 4. Figures (network stars and centrality bars, both resolutions).
    n_edges = int(cluster["edges"]["is_directed_edge"].sum())
    draw_network_figure(
        cluster["edges"], cluster["nodes"],
        FIG_DIR / "directed_network_category_1000dpi.png",
        FIG_DIR / "directed_network_category_1000dpi.pdf",
        edges_all=cluster["edges"],
    )
    draw_centrality_figure(
        cluster["nodes"],
        FIG_DIR / "node_centrality_category_1000dpi.png",
        FIG_DIR / "node_centrality_category_1000dpi.pdf",
        "Node centrality in the directed co-detection network (category level)",
    )
    draw_network_figure(
        compound["edges"], compound["nodes"],
        FIG_DIR / "directed_network_compound_1000dpi.png",
        FIG_DIR / "directed_network_compound_1000dpi.pdf",
        edges_all=compound["edges"],
    )
    draw_centrality_figure(
        compound["nodes"],
        FIG_DIR / "node_centrality_compound_1000dpi.png",
        FIG_DIR / "node_centrality_compound_1000dpi.pdf",
        "Node centrality in the directed co-detection network (compound level)",
    )

    # 5. Word methods and Markdown summary.
    methods = OUT_DIR / "methods_directional_cooccurrence_network.docx"
    write_methods_docx(methods, n_samples, n_subjects, cluster["nodes"], cluster["edges"], cluster["hb_nodes"])
    summary = OUT_DIR / "network_summary_cn.md"
    write_summary(summary, n_samples, n_subjects, cluster, compound)

    print(f"Excel:    {excel}")
    print(f"Methods:  {methods}")
    print(f"Summary:  {summary}")
    print(f"Figures:  {FIG_DIR}")
    print()
    print(f"Category level: {len(cluster['nodes'])} nodes, {n_edges} directed edges "
          f"({int(cluster['edges']['is_enriched_directed_edge'].sum())} enrichment-significant)")
    print(cluster["nodes"][["node", "n_detected", "out_degree", "in_degree",
                           "hub_score", "authority_score", "betweenness"]].to_string(index=False))
    print()
    print(cluster["grid"].to_string(index=False))


if __name__ == "__main__":
    main()
