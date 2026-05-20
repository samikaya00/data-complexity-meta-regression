# Data Complexity Meta-Regression

Code repository for the manuscript

> **Estimating Classification Difficulty from Data Complexity: An Algorithm-Agnostic Meta-Regression Framework**

This repository contains the full pipeline used to compute the 11-metric complexity feature vector on 70 binary classification datasets, construct the stratified 53/17 training/validation split, train and validate the GradientBoosting meta-regressor, and reproduce the secondary analyses (SHAP / permutation / bootstrap / Kolmogorov–Smirnov diagnostics / algorithm-selection gap).

---

## Repository layout

```
.
├── src/
│   ├── 00_make_splits.py                    # Stratified 53/17 split (IR × N1, seed=42)
│   ├── 01_select_datasets.py                # Pool construction from KEEL
│   ├── 02_benchmark_overlap_measure.py      # 11-metric complexity computation
│   ├── 03_benchmark_overlap_classifiers.py  # Cross-validated 9-classifier benchmark
│   ├── 04_predictive_modeling_complete.py   # GB meta-regressor; LODO + external; primary figures
│   └── 05_supplementary_analyses.py         # SHAP, permutation, bootstrap, KS, RF supplementary, gap
├── data/
│   ├── overlap_metrics.csv                  # 58 datasets × 11 complexity metrics (original pool)
│   ├── validation_metrics.csv               # 15 datasets × 11 complexity metrics (reserved pool)
│   ├── combined_overlap_performance.csv     # Per-(dataset, classifier) results on the original pool
│   ├── validation_classification.csv        # Per-(dataset, classifier) results on the reserved pool
│   ├── train_split.csv                      # 53 stratified training datasets (output of 00_make_splits.py)
│   └── val_split.csv                        # 17 stratified held-out validation datasets
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

Python 3.10 or newer. Install dependencies into a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data

The 70 binary classification datasets analyzed in the manuscript are publicly available from the [KEEL repository](https://sci2s.ugr.es/keel/). The pool was constructed from 73 candidate datasets; three (`glass-0-1-4-6_vs_2`, `ecoli-0-1-3-7_vs_2-6`, `shuttle-c2-vs-c4`) were excluded for computational reasons, leaving the 70 datasets used throughout.

The repository ships the intermediate CSVs that feed every downstream analysis:

| File | Contents |
|---|---|
| `data/overlap_metrics.csv` | 11 complexity metrics for the 58 datasets in the original pool |
| `data/validation_metrics.csv` | 11 complexity metrics for the 15 datasets reserved for external use |
| `data/combined_overlap_performance.csv` | Per-(dataset, classifier) accuracy / F1 / AUC on the original pool |
| `data/validation_classification.csv` | Per-(dataset, classifier) accuracy / F1 / AUC on the reserved pool |
| `data/train_split.csv` | 53 stratified training datasets, with 11 features and 3 aggregated targets |
| `data/val_split.csv` | 17 stratified validation datasets, same schema |

`train_split.csv` and `val_split.csv` are deterministic outputs of `00_make_splits.py`; the file pair is provided directly so that the manuscript's reported numerical values can be reproduced exactly without rerunning the upstream pipeline.

---

## Reproducing the manuscript

### Recommended path — start from the provided splits

The provided `train_split.csv` and `val_split.csv` are sufficient input for the meta-regression and supplementary analyses, and using them guarantees exact reproduction of the manuscript's tables and figures.

**Primary results (Sections 4.1–4.3, Figures 1–4, Tables 2–4)**

```bash
python src/04_predictive_modeling_complete.py
```

**Secondary analyses (Sections 4.2, 4.4, 4.5; Tables 5–8)**

```bash
python src/05_supplementary_analyses.py \
    --train data/train_split.csv \
    --val   data/val_split.csv \
    --perf  data/combined_overlap_performance.csv \
    --out   results_supplementary
```

The supplementary script produces six outputs:

| Manuscript element | Output file |
|---|---|
| Nine-family regressor comparison (Section 3.4) | `regression_model_performance.csv` |
| KS distribution-shift diagnostics (Section 4.2) | `ks_test_results.csv` |
| Gini / permutation / SHAP importance (Section 4.4, Table 6) | `importance_gb_{accuracy,f1,auc}.csv`, `importance_rf_{accuracy,f1,auc}.csv`, `importance_triangulation_summary.csv` |
| Bootstrap rank stability (Section 4.4) | `bootstrap_rank_stability.csv` |
| RandomForest supplementary meta-regressor (Section 4.1) | `rf_supplementary_lodo.csv`, `rf_supplementary_external.csv` |
| Algorithm-selection gap analysis (Section 4.5, Table 8) | `gap_analysis.csv` |

Individual analyses can be skipped with `--skip` (e.g. `--skip D F`).

### Reconstructing the split from scratch

To re-derive `train_split.csv` and `val_split.csv` from the four intermediate CSVs:

```bash
python src/00_make_splits.py \
    --metrics-train data/overlap_metrics.csv \
    --metrics-val   data/validation_metrics.csv \
    --perf-train    data/combined_overlap_performance.csv \
    --perf-val      data/validation_classification.csv \
    --out-dir       data
```

Stratification is computed on three IR quantile bins crossed with two N1 quantile bins, yielding six strata; `StratifiedShuffleSplit(test_size=17, random_state=42)` then samples 17 validation datasets while preserving stratum proportions. The resulting partition is identical to the shipped `train_split.csv` / `val_split.csv`.

### Full pipeline from raw KEEL archives

`01_select_datasets.py`, `02_benchmark_overlap_measure.py`, and `03_benchmark_overlap_classifiers.py` reproduce the original-pool intermediate CSVs from raw KEEL `.dat` files. These scripts require local copies of the KEEL imbalanced-classification archives and **include hardcoded path constants at the top of each file that need to be adjusted for the local environment**. They are included for full transparency; the recommended reproduction path (above) does not need them.

---

## Deterministic execution

All random number generators are seeded with `random_state=42`. The shipped splits and the manuscript's reported numerical values were produced with `scikit-learn==1.3.x` and `shap==0.43.x`. Minor numerical differences may arise across major scikit-learn releases owing to default-parameter changes; the dataset membership of the split is unaffected.

---

## Citation

If you use this code, please cite the manuscript (citation to be updated upon publication):

```bibtex
@article{kaya2026complexity,
  author  = {Kaya, Sami},
  title   = {Estimating Classification Difficulty from Data Complexity:
             An Algorithm-Agnostic Meta-Regression Framework},
  journal = {Mathematics},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code is released under the MIT License (see `LICENSE`). The KEEL datasets retain their original licenses and are not redistributed here.
