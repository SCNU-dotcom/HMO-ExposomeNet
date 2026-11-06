# Unsupervised Molecular Classification

Unsupervised clustering and interactive network visualization of cardiovascular-related
emerging contaminants. Within each compound category, molecules are grouped by
physicochemical similarity and rendered as a D3.js force-directed network.

The pipeline follows the descriptor-based approach of Smer-Barreto et al.,
*Nature Communications* (2023):

```text
SMILES
  -> RDKit physicochemical descriptors (~200, Ipc excluded)
  -> min-max scaling -> continuous Tanimoto distance matrix
  -> Tanimoto similarity graph -> Louvain community detection (gamma plateau scan)
  -> t-SNE embedding + MST / k-NN display network
  -> interactive HTML + CSV with cluster labels
```

Every category goes through the same unsupervised pipeline with the same parameters.
No cluster labels are assigned or edited by hand.

## Repository Layout

```text
Net.py                    Main script (full pipeline + HTML generation)
compounds_list.xlsx       Input compound list (Name, Category, SMILES); not tracked, place it here
requirements.txt          Full pip freeze of the original environment
requirements-minimal.txt  Pinned direct dependencies of Net.py
results/                  Output directory (generated, not tracked)
```

## Installation

The original environment used Python 3.9 on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements-minimal.txt
```

`requirements.txt` is a complete `pip freeze` of the development environment. It
contains many packages that `Net.py` does not use, and some entries point to local
conda build paths (`@ file:///...`), so it cannot be installed as-is on another
machine. Use it as a version reference; `requirements-minimal.txt` lists only the
packages the script imports, pinned to the same versions.

## Input Format

An Excel (`.xlsx`) or CSV file with these columns:

| Column     | Required | Description                                          |
|------------|----------|------------------------------------------------------|
| `Name`     | yes      | Compound name, shown in the network and tooltips     |
| `SMILES`   | yes      | SMILES string; unparseable entries are skipped       |
| `Category` | no       | Compound class; each category is clustered separately |

If `Category` is missing, all compounds are processed as one group (`All_Compounds`).

## Usage

```bash
# Defaults: ./compounds_list.xlsx -> ./results
python Net.py

# Custom paths
python Net.py --input path/to/compounds.xlsx --output path/to/results
```

## Outputs

- `<Category>_network.html`: interactive network for each category. Open it in a
  browser (internet access is needed to load D3.js and html2canvas from CDNs).
  It provides:
  - nodes colored by cluster, with dashed rings marking cluster centers
  - curved edges weighted by descriptor similarity
  - hover tooltips with the structure image, key properties and functional groups
  - a sidebar per cluster with size, center compound, functional-group pie chart,
    representative structure and mean MolLogP / TPSA / MolWt
  - export buttons for a 1000 DPI PNG and a vector PDF (via the browser print dialog)
- `compounds_with_clusters.csv`: the input table with an added `Cluster` column.
  Clusters are numbered from 1 within each category, largest cluster first.
  Compounds in skipped categories are left empty.

## Method Details

| Step | Implementation | Key parameters |
|------|----------------|----------------|
| Descriptors | `rdkit.Chem.Descriptors._descList` without `Ipc` | |
| Distance | Continuous Tanimoto on min-max scaled descriptors, `d = 1 - dot(A,B) / (|A|^2 + |B|^2 - dot(A,B))` | |
| Similarity graph | Edge if distance < threshold; threshold relaxed in 0.05 steps until the graph has at least n edges | `TANIMOTO_EDGE_THRESHOLD = 0.7` (max 0.95) |
| Clustering | Louvain over a gamma grid, 30 runs per gamma; gamma chosen at the most stable plateau of the mean cluster count | `LOUVAIN_GAMMA_RANGE = (0.5, 3.0)`, `LOUVAIN_GAMMA_STEP = 0.1`, `LOUVAIN_N_RUNS = 30` |
| Fallback | Louvain (gamma = 1) on a 5-NN graph when the similarity graph is too sparse or yields a single cluster | |
| Cluster center | Member with the lowest mean intra-cluster distance | |
| Embedding | t-SNE on the precomputed distance matrix | perplexity `min(50, n-1)`, learning rate 200, 1200 iterations |
| Display network | Minimum spanning tree plus up to 2 nearest neighbours per node with distance < 0.7 | `KNN_EXTRA = 2`, `EDGE_DIST_CUTOFF = 0.7` |


Functional groups are detected with SMARTS patterns (quinone, phenol, aromatic amine,
nitro, sulfonamide, tertiary amine, ester, amide, halide, organophosphate, epoxide,
carboxylic acid). Each pattern carries a short note on its cardiovascular relevance
in `CVD_FUNCTIONAL_GROUPS`. These annotations are descriptive and do not affect the
clustering.

## Reference

Smer-Barreto, V. et al. Discovery of senolytics using machine learning.
*Nature Communications* 14, 3445 (2023).
