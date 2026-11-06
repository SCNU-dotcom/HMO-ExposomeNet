# Co-occurrence and Enrichment Analysis of Pollutants

Statistical co-occurrence and enrichment analyses of short-chain PFAS
(clusters **PFAS-1** and **PFAS-2**) against other emerging pollutants in a
sample-resolved human chemical exposome (355 organ specimens).

The repository has three independent analyses:

| Script | Question | Output folder |
|---|---|---|
| `analyze_pfas_focus_directional_coverage.py` | When pollutant *j* is detected, how often is PFAS-1 / PFAS-2 detected too? | `pfas_focus_directional_outputs/` |
| `analyze_pfas_directional_cooccurrence_network.py` | Which pollutants statistically "cover" others in a directed co-detection network? | `pfas_directional_network_outputs/` |
| `analyze_pfas_fisher_causal.py` | Are target pollutant concentrations higher in higher PFAS-1 / PFAS-2 concentration windows? | `pfas_fisher_causal_outputs/` |

## Repository Structure

```text
.
├── Sample-resolved_human_chemical_exposome.xlsx       # input data
├── gee_causal_core.py                                 # shared paths, PFAS groups, data loader
├── analyze_pfas_focus_directional_coverage.py         # analysis 1
├── analyze_pfas_directional_cooccurrence_network.py   # analysis 2
├── analyze_pfas_fisher_causal.py                      # analysis 3
├── network_figures.py                                 # PIL renderer for the network figures
├── vector_pdf.py                                      # replays PIL drawings as vector PDF
├── requirements.txt
├── pfas_focus_directional_outputs/
├── pfas_directional_network_outputs/
└── pfas_fisher_causal_outputs/
```

## Input Data

`Sample-resolved_human_chemical_exposome.xlsx`, first worksheet, long format
(one row per detected compound in one specimen, 6,394 rows).

| Column | Description |
|---|---|
| `Sample` | Donor ID (`S001`, ...) |
| `Organ` / `System` | Organ and organ system of the specimen |
| `Name` | Compound name |
| `Category` | Pollutant category (`PFAS`, `Pest.`, `Pharm.`, `OPEs`, `PAEs`, `SHDs`, `VDs`, `PCPs`, `DBPs`, `BPs`, `FAs`, `Others`) |
| `Clusters` | Compound cluster within its category (`PFAS-1` ... `PFAS-5`, `Pest.-1`, ...) |
| `Concentration` | Concentration (ng/g); values below the detection limit are not listed |
| `Gender`, `Height`, `Age`, `CC-CoD` | Donor metadata (not used by the scripts) |

Conventions shared by all scripts:

- One specimen is one unique (`Organ`, `Sample`) pair; concentrations of the
  same variable within a specimen are summed.
- A variable is "detected" in a specimen when its summed concentration is > 0.
- `Clusters` IDs are used as-is: **PFAS-1, PFAS-2 = short-chain**;
  PFAS-3, PFAS-4, PFAS-5 = long-chain / precursors.

The input path is defined once in `gee_causal_core.py` (`WORKBOOK`).

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Figures are rendered in **Arial**. `network_figures.py` looks for `arial.ttf`
(or Liberation Sans / Helvetica) in `C:/Windows/Fonts` and the per-user Windows
font folder, and stops with an error if none is found. On Linux or macOS,
install an Arial-metric font and add its folder to `font()` in
`network_figures.py`.

## Usage

Run each script from any working directory; outputs are written next to the
scripts.

```bash
python analyze_pfas_focus_directional_coverage.py
python analyze_pfas_directional_cooccurrence_network.py   # slowest: 2,000 bootstrap resamples
python analyze_pfas_fisher_causal.py
```

Re-running overwrites the files in the corresponding output folder.

## Methods

### 1. Directional co-detection coverage

PFAS keeps cluster resolution (PFAS-1 ... PFAS-5); non-PFAS pollutants are
summed by `Category`, giving a square matrix of
`C(i|j) = n(i and j detected) / n(j detected)`. For the PFAS-1 / PFAS-2 rows,
a one-sided exact binomial test checks `C(focus|other) > 0.5`, with
Benjamini-Hochberg FDR within the focus comparisons. A top-25% high-burden
Fisher enrichment table is exported for reference only.

### 2. Directed co-occurrence network

For every ordered node pair (S, T) at cluster resolution (main) and compound
resolution (appendix):

1. Coverage `P(S|T)` and reverse coverage `P(T|S)`.
2. One-sided hypergeometric (Fisher) enrichment test.
3. Asymmetry index `A = P(S|T) - P(T|S)`, one-sided exact McNemar test, and a
   cluster bootstrap over `Sample` (2,000 resamples, seed `20260728`).
4. Edge `S -> T` retained when coverage >= 0.90, A > 0, McNemar q < 0.05,
   bootstrap CI of A above 0, and coverage-test q < 0.05 (BH-FDR per test family).
5. Out/in-degree, strength, HITS hub/authority and Brandes betweenness.

Sensitivity analyses: coverage thresholds 0.70 / 0.80 / 0.90 / 0.95 and a
high-burden (>= Q75 of positive values) binarization.

Edge direction follows from detection prevalence, so the network describes
conditional co-detection structure. It does not imply causal transfer or
exposure-source direction.

### 3. Concentration-window enrichment

Each short-chain source (PFAS-1, PFAS-2) is log1p-transformed and split into
quintiles. The four upper quintiles (W1-W4) are each compared with the lowest
quintile for every non-PFAS cluster target:

- Primary test: one-sided Mann-Whitney U on target concentration (asymptotic,
  tie-corrected), with rank-biserial correlation as effect size.
- Direction gate: detection odds ratio with Haldane-Anscombe +0.5 correction.
- Secondary (audit only): one-sided Boschloo and Fisher exact tests.
- An edge is kept when raw p < 0.05 and OR > 1; the lowest-p window represents
  each source-target pair.

## Outputs

All figures are exported as 1000-dpi PNG plus PDF. Each Excel workbook keeps
the intermediate tables needed to reproduce the figures. The Word methods
files and `*_summary_cn.md` reports are written in Chinese.

### `pfas_focus_directional_outputs/`

| File | Content |
|---|---|
| `pfas1_pfas2_directional_coverage_analysis.xlsx` | `README`, grouped long data, specimen x variable matrix, log1p values, detection counts, variable annotations, full coverage matrix, pairwise table, focus binomial tests with FDR, PFAS-1/PFAS-2 vs others, top-25% reference enrichment |
| `figures/pfas1_pfas2_directional_coverage_matrix_1000dpi.{png,pdf}` | Coverage heatmap; black dot = focus FDR q < 0.05 |
| `methods_directional_codetection_coverage.docx` | Methods text |
| `analysis_summary_cn.md` | Result summary |

### `pfas_directional_network_outputs/`

| File | Content |
|---|---|
| `pfas_directional_cooccurrence_network.xlsx` | `README`, node annotations, binary matrix, thresholds, all / retained edges and centralities (cluster and compound level), threshold sensitivity, high-burden edges and centralities |
| `figures/directed_network_category_1000dpi.{png,pdf}` | Star networks centred on PFAS-1 and PFAS-2, cluster level |
| `figures/directed_network_compound_1000dpi.{png,pdf}` | Same, compound level |
| `figures/node_centrality_{category,compound}_1000dpi.{png,pdf}` | Degree, hub, authority and betweenness bars |
| `methods_directional_cooccurrence_network.docx` | Methods text |
| `network_summary_cn.md` | Result summary |

### `pfas_fisher_causal_outputs/`

| File | Content |
|---|---|
| `pfas_fisher_causal_network.xlsx` | `01_all_tests` (every source x window x target test), `02_network_edges` (best window per pair), `03_node_roles` |
| `figures/fisher_causal_PFAS-{1,2}_arc_circos_significant_1000dpi.{png,pdf}` | Arc circos of targets with at least one significant window |
| `figures/fisher_causal_PFAS-{1,2}_arc_circos_all_pollutants_1000dpi.{png,pdf}` | Same layout over every tested target |

## Citation

If you use this code or data, please cite the associated publication
(citation to be added).
