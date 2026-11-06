# HMO-ExposomeNet

Research code for sample-resolved chemical exposure analysis with a tabular transformer and conditional diffusion augmentation. The repository contains the model implementation, an optional workbook aggregation utility, input workbooks, and generated analysis artifacts.

## Repository contents

- `Risk.py`: preprocessing, model definitions, training, evaluation, and figure generation.
- `group_exposome.py`: aggregates the source workbook by `Sample`, `Organ`, and `Clusters`, summing `Concentration`. It writes a separate output file and does not modify the source workbook.
- `Sample-resolved_human_chemical_exposome.xlsx`: source workbook expected by the aggregation utility.
- `Grouped sample-resolved human chemical exposome.xlsx`: grouped workbook expected by `Risk.py`.
- `requirements.txt`: environment snapshot from the original development environment. See [Installation](#installation) for the recommended minimal dependencies.
- `cocktail_risk_results/`: example outputs and checkpoints. These are derived artifacts, not required to run the pipeline.

## Data aggregation

Install the spreadsheet dependencies and run:

```bash
python -m pip install pandas openpyxl
python group_exposome.py
```

Rows sharing the same `Sample`, `Organ`, and `Clusters` values are treated as one sample-cluster observation. Their `Concentration` values are summed. Other metadata columns are retained from the group only when they are consistent within that group; inconsistent metadata raises an error. The output cluster key is named `Cluster` to match the model input schema.

Custom input and output paths can be supplied with `--input` and `--output`.

## Installation

Use Python 3.10 or 3.11 in a virtual environment. PyTorch installation depends on the available operating system and accelerator; install a compatible PyTorch build using the [official selector](https://pytorch.org/get-started/locally/). Then install the runtime packages used by `Risk.py`:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn pillow openpyxl rdkit
```

`requirements.txt` records a machine-specific environment and includes CUDA-specific wheels and editable packages from local paths. Those entries are not portable and are not needed by the imports in `Risk.py`; use it as a provenance record, not as a cross-platform installation lockfile.

## Running the analysis

Place the grouped workbook beside `Risk.py`, then run:

```bash
python Risk.py
```

The script uses the adjacent grouped workbook by default and writes generated figures, tables, logs, and model checkpoints under `cocktail_risk_results/`. Paths can be overridden with `HMO_EXPOSOME_DATA` and `HMO_EXPOSOME_OUTPUT`.

The training script can require substantial compute, particularly when CUDA is unavailable. The configured folds, epochs, and diffusion training steps are source-level settings.

## Method and interpretation

The implementation includes a supervised exposure-risk model, descriptor-derived molecular features, an attention-based interaction pathway, demographic covariates, and conditional diffusion augmentation.

## Reproducibility and data governance

The random seed, data split, software versions, and hardware affect results. Preserve the exact input version and record the configuration used for each run. Before publishing the workbooks or generated artifacts, verify that their licenses permit redistribution and that they contain no restricted or identifiable information. Cite the original data source and any external methods used in publications.

## Literature

- Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems* 33 (2020). https://arxiv.org/abs/2006.11239
- Peebles, W. & Xie, S. Scalable Diffusion Models with Transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision* (2023). https://arxiv.org/abs/2212.09748
- Lin, T.-Y. et al. Focal Loss for Dense Object Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42, 318–327 (2020). https://doi.org/10.1109/TPAMI.2018.2858826
- Benjamini, Y. & Hochberg, Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B* 57, 289–300 (1995). https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Hernán, M. A. & Robins, J. M. *Causal Inference: What If*. Chapman & Hall/CRC (2020). https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

## License

No license has been specified yet. Add a license only after confirming the authors' rights to the code and data. Until then, reuse and redistribution permissions are not granted by this repository.
