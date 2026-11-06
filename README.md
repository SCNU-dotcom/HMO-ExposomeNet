# HMO-ExposomeNet

**Analysis Resource for Human Multi-Organ Chemical Exposome Network Analysis**

HMO-ExposomeNet is a three-stage research workflow for organizing molecular
diversity, examining pollutant co-occurrence and enrichment, and estimating
interpretable exposure-related health risk in a sample-resolved human chemical
exposome. The intended analysis order is:

```text
Compound names + SMILES
        |
        v
1. Unsupervised molecular classification
   RDKit descriptors -> molecular similarity -> Louvain clusters
        |
        v
Cluster-labelled sample-by-organ exposure data
        |
        v
2. Co-occurrence and enrichment analysis
   conditional co-detection + directed networks + concentration windows
        |
        v
Candidate mixture structure and cluster-level exposure patterns
        |
        v
3. Machine learning-based exposure risk assessment
   conditional diffusion augmentation + transformer classifier + attribution
```

The stages answer different questions. Molecular clustering asks which compounds
are physicochemically similar without using outcome labels. Enrichment analysis
asks which cluster-level exposures tend to occur together or at elevated
concentrations. The final model asks whether the observed exposure profile is
associated with the binary `CCVM` outcome and provides several descriptive
interpretability views. Co-occurrence, enrichment, attention, and attribution do
not by themselves establish causality.

## Repository structure

```text
HMO-ExposomeNet/
|-- README.md
|-- Unsupervised molecular classification/
|   |-- Net.py
|   |-- compounds_list.xlsx
|   |-- requirements-minimal.txt
|   `-- results/
|-- Co-occurrence and enrichment analysis of pollutants/
|   |-- gee_causal_core.py
|   |-- analyze_pfas_focus_directional_coverage.py
|   |-- analyze_pfas_directional_cooccurrence_network.py
|   |-- analyze_pfas_fisher_causal.py
|   |-- network_figures.py
|   |-- vector_pdf.py
|   `-- Sample-resolved_human_chemical_exposome.xlsx
`-- Machine Learning-based Exposure Risk Assessment/
    |-- group_exposome.py
    |-- Risk.py
    |-- Sample-resolved_human_chemical_exposome.xlsx
    |-- Grouped sample-resolved human chemical exposome.xlsx
    `-- cocktail_risk_results/
```

Each stage has a detailed README in its own directory. This document describes
the integrated workflow and the data handoffs between stages.

## Requirements

- Python 3.10 or 3.11 is recommended for the integrated workflow.
- Windows 10/11 is the environment used for the supplied analyses.
- A CUDA-capable GPU is strongly recommended for `Risk.py`; CPU execution can
  be substantially slower.
- Interactive molecular networks require internet access to load D3.js and
  html2canvas from CDNs.
- The enrichment figures are designed for Arial. On systems without Arial,
  install an Arial-compatible font and update the font lookup if necessary.

The three stages record different historical environments. In particular, the
full `requirements.txt` in the machine-learning directory is a machine-specific
snapshot containing CUDA wheels and local editable paths. It should not be used
as a portable lockfile. For a clean environment, install PyTorch from the
[official selector](https://pytorch.org/get-started/locally/) and then install
the direct packages used by the workflow:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux or macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
# Install the appropriate torch build first: https://pytorch.org/get-started/locally/
python -m pip install numpy pandas scipy scikit-learn networkx rdkit openpyxl \
  matplotlib seaborn pillow XlsxWriter python-docx
```

For exact stage-specific versions, see:

- `Unsupervised molecular classification/requirements-minimal.txt`
- `Co-occurrence and enrichment analysis of pollutants/requirements.txt`
- `Machine Learning-based Exposure Risk Assessment/requirements.txt`

Because the stage-specific pinned versions are not identical, isolated virtual
environments are the most conservative choice for exact reproduction.

## Data flow and input contracts

### Stage 1 input

`Unsupervised molecular classification/compounds_list.xlsx` must contain one
compound per row.

| Column | Required | Meaning |
|---|---:|---|
| `Name` | Yes | Unique compound name |
| `SMILES` | Yes | Molecular structure; invalid values are skipped |
| `Category` | No | Chemical class; clustering is performed separately by class |

### Stage 1 to Stage 2 handoff

Stage 1 writes `results/compounds_with_clusters.csv`, adding a `Cluster` column.
Before Stage 2, join these labels to the long-format exposure table by a stable
compound identifier, normally `Name`, and store the result in the exposure
table's `Clusters` column. Verify unmatched names, duplicate names, and category
consistency before analysis. The supplied workbooks already contain cluster IDs.

The enrichment scripts expect the first worksheet of
`Sample-resolved_human_chemical_exposome.xlsx` to contain at least:

| Column | Meaning |
|---|---|
| `Sample` | Donor identifier |
| `Organ` | Sampled organ; one specimen is a `(Sample, Organ)` pair |
| `System` | Organ-system annotation |
| `Name` | Compound name |
| `Category` | Pollutant category |
| `Clusters` | Stage 1 cluster identifier, such as `PFAS-1` |
| `Concentration` | Concentration in ng/g; absent rows are treated as non-detections |

The current enrichment modules focus on PFAS clusters. `PFAS-1` and `PFAS-2`
are configured as the short-chain focus; `PFAS-3`, `PFAS-4`, and `PFAS-5` are
configured as long-chain or precursor clusters. If Stage 1 produces different
IDs, update `PFAS_ORDER`, `SHORT_CHAIN`, and `LONG_CHAIN` in
`gee_causal_core.py` before running Stage 2. Cluster numbers are dataset-specific
labels, not universal chemical classes.

### Stage 2 to Stage 3 handoff

The risk model uses the same cluster-labelled long-format data plus molecular,
demographic, and outcome fields. `group_exposome.py` sums concentration within
each `(Sample, Organ, Clusters)` group and renames `Clusters` to `Cluster`.
`Risk.py` requires the grouped workbook to contain:

```text
Name, Sample, Organ, Concentration, SMILES, CCVM, Age, Height, Gender, Cluster
```

`CCVM` is the binary prediction target. The supplied raw exposure workbook uses
the outcome name `CC-CoD` and does not include SMILES, whereas the supplied
grouped modelling workbook already uses `CCVM` and includes SMILES. Therefore, a
fresh end-to-end run requires an explicit preparation step before aggregation:
join SMILES from the Stage 1 compound table by `Name`, rename `CC-CoD` to `CCVM`,
and validate the mapping. `group_exposome.py` does not perform those transformations.
A compound name must map consistently to one SMILES string and one cluster label.
Metadata must also be constant within each aggregation group; the grouping
utility stops on inconsistent metadata.

## Reproducing the workflow

Run commands from the indicated stage directory. Paths contain spaces, so quote
them in shell commands.

### 1. Unsupervised molecular classification

```powershell
Set-Location "Unsupervised molecular classification"
python Net.py
```

For each chemical category, `Net.py` computes approximately 200 RDKit
physicochemical descriptors (excluding `Ipc`), applies min-max scaling, builds a
continuous-Tanimoto distance matrix, and constructs a similarity graph. Louvain
community detection is run over a resolution grid with repeated runs; a stable
cluster-count plateau selects the resolution. A 5-nearest-neighbour fallback is
used for sparse or single-community graphs. t-SNE supports visualization only;
it does not determine cluster membership.

Important outputs are:

- `results/compounds_with_clusters.csv`: compound-to-cluster assignments.
- `results/<Category>_network.html`: interactive molecular networks.
- `results/<Category> - Molecular Network.{png,pdf}`: static exports, when
  generated through the network interface.

Custom locations can be supplied with `--input` and `--output`.

### 2. Co-occurrence and enrichment analysis

After transferring and validating the cluster labels, run:

```powershell
Set-Location "../Co-occurrence and enrichment analysis of pollutants"
python analyze_pfas_focus_directional_coverage.py
python analyze_pfas_directional_cooccurrence_network.py
python analyze_pfas_fisher_causal.py
```

The three analyses are complementary:

1. **Directional co-detection coverage** computes
   `C(i|j) = P(i detected | j detected)`. For PFAS-1 and PFAS-2, one-sided exact
   binomial tests and Benjamini-Hochberg false-discovery-rate correction assess
   coverage above 0.5. A top-quartile burden analysis is exported as a reference.
2. **Directed co-occurrence networks** retain an edge `S -> T` when `S` covers
   `T`, meaning `P(S detected | T detected)` is high and exceeds the reverse
   coverage. Edge inference combines hypergeometric enrichment, exact McNemar
   testing, FDR control, and a 2,000-resample donor-level cluster bootstrap.
   The primary coverage threshold is 0.90, with threshold and high-burden
   sensitivity analyses. Direction reflects prevalence-dominant conditional
   co-detection, not exposure-source direction.
3. **Concentration-window enrichment** log-transforms each short-chain PFAS
   source, divides it into quintiles, and compares each upper window with the
   lowest window. The primary test is one-sided Mann-Whitney U; detection odds
   ratios provide a direction gate, while Boschloo and Fisher exact tests are
   secondary audit outputs. The current edge rule uses raw `p < 0.05` and
   `OR > 1`; interpret it as exploratory unless multiplicity control is added.

Each module writes an Excel workbook with intermediate tables and 1000-dpi PNG
and PDF figures into its own output directory. The directional network is the
slowest stage because of bootstrap resampling. Despite the historical filename
`analyze_pfas_fisher_causal.py`, its concentration-window associations do not
identify causal effects.

### 3. Interpretable exposure-risk assessment

First prepare the long-format table as described above, aggregate the
cluster-labelled exposure table, and then train the model. The command below can
be run directly with the supplied files because the repository already includes
the prepared grouped modelling workbook. Re-running `group_exposome.py` from the
supplied raw workbook without first adding `SMILES` and renaming the outcome will
produce a workbook that does not satisfy `Risk.py`'s input contract.

```powershell
Set-Location "../Machine Learning-based Exposure Risk Assessment"
# After preparing a long-format input with SMILES and CCVM:
python group_exposome.py --input "prepared_exposome.xlsx" \
  --output "Grouped sample-resolved human chemical exposome.xlsx"
python Risk.py
```

Optional environment variables override the default paths:

```powershell
$env:HMO_EXPOSOME_DATA = "C:\path\to\grouped_exposome.xlsx"
$env:HMO_EXPOSOME_OUTPUT = "C:\path\to\results"
python Risk.py
```

The model combines log-transformed concentration profiles, RDKit molecular
descriptors, organ identity, and age/height/gender covariates. Its main elements
are a transformer-based exposure encoder, structural and concentration
co-occurrence priors, attention-based mixture interactions, focal loss, and a
conditional diffusion model used for training-data augmentation. The configured
run uses an 80/20 stratified development/test split, 10-fold model selection
within the development partition, seed 51, up to 300 epochs, and validation-based
decision-threshold selection.

`cocktail_risk_results/` contains model checkpoints, tabular plot data, and:

- training and validation curves;
- held-out ROC, precision-recall, confusion-matrix, and score-distribution views;
- attention-derived mixture-effect summaries;
- cluster-level independent-effect summaries;
- descriptor attribution and highlighted molecular substructures;
- diffusion-augmentation diagnostics;
- organ-specific network visualizations and demographic sensitivity analysis.

These views make the fitted model inspectable, but they require careful
interpretation. Attention weights are not causal interaction effects. Descriptor
attributions explain model behavior, not toxicological mechanisms. Post-training
score offsets and pairwise attention-bias calibration in `Risk.py` encode
configured PFAS reporting profiles; outputs after those adjustments are
descriptive and are not independent validation results. Report pre-adjustment
held-out metrics separately, and use an external cohort for confirmatory claims.

## Reproducibility checklist

- Freeze the exact input workbook and record its checksum.
- Preserve the Stage 1 cluster assignment table used in downstream analyses.
- Record all unmatched compounds during the cluster-label join.
- Keep the compound, cluster, category, and SMILES mappings versioned together.
- Record Python, RDKit, PyTorch, CUDA, operating-system, and font versions.
- Keep random seeds, split indices, thresholds, and model checkpoints.
- Perform multiplicity-aware and threshold sensitivity analyses for exploratory
  enrichment results.
- Validate predictive performance and interpretation patterns on an independent
  dataset before making biological, clinical, or regulatory claims.

The supplied workbooks and generated outputs may be subject to privacy, consent,
data-use, or third-party licensing restrictions. Review these constraints before
publishing them. Do not commit personal identifiers, restricted donor metadata,
local environment files, or large regenerated artifacts unless distribution is
explicitly permitted.

## Scope and limitations

This repository is research software, not a clinical diagnostic or regulatory
risk-assessment instrument. Its output depends on the sampled population, organ
coverage, detection limits, missingness, cluster definitions, and outcome labels.
Absent rows are treated as non-detections by the co-occurrence analyses, which is
appropriate only when the input table and assay reporting rules support that
interpretation. Molecular similarity does not guarantee shared exposure sources
or toxicological effects, and observational enrichment cannot establish a causal
pathway.

## Citation

If you use HMO-ExposomeNet, cite the associated publication and archived software
release. Publication metadata and a repository DOI should be added here before
the first public release. The molecular-classification design also references:

- Smer-Barreto, V. et al. Discovery of senolytics using machine learning.
  *Nature Communications* **14**, 3445 (2023).
- Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models.
  *Advances in Neural Information Processing Systems* **33** (2020).
- Benjamini, Y. & Hochberg, Y. Controlling the False Discovery Rate: A Practical
  and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical
  Society: Series B* **57**, 289-300 (1995).

## License

No open-source license is currently included. Until the authors confirm ownership
of the code and redistribution rights for the data, reuse and redistribution are
not granted by default. An appropriate open-source license for the code should be
added, and the data license should be documented separately. A code license does
not automatically grant permission to redistribute the workbooks.

## Contributing and support

Bug reports and reproducibility issues are welcome through GitHub Issues. Reports
should include the stage, command, operating system, Python version, dependency
versions, and a minimal traceback or reproducible example. Do not attach
restricted exposure data to a public issue.
