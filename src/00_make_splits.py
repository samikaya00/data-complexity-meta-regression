"""
Reconstruct the 53/17 stratified split for Paper 2.

Combines `overlap_metrics.csv` (58 original-pool datasets) with
`validation_metrics.csv` (15 datasets reserved for external use), drops
three datasets excluded for computational reasons, and partitions the
remaining 70 datasets into 53 training and 17 stratified validation
datasets using IR x N1 strata.

Stratification follows Task 2.2 of the experimental-validation plan:
  - imbalance_ratio binned into 3 quantile groups (low / mid / high IR)
  - N1 binned into 2 quantile groups (low / high N1)
  - combined into 6 strata, used as the stratify variable
  - StratifiedShuffleSplit with test_size = 17, random_state = 42
  - if a stratum is too small to split, fall back to IR-only (q=4)
    stratification with the same random_state

Outputs are written to:
  data/train_split.csv
  data/val_split.csv

This is the same procedure that produced the splits originally; it is
included here to make the split deterministic and inspectable.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SEED = 42
TEST_SIZE = 17

# Datasets excluded from the original 73-dataset pool for computational
# reasons (size or convergence cost), per the manuscript text.
EXCLUDED_DATASETS = {
    "glass-0-1-4-6_vs_2",
    "ecoli-0-1-3-7_vs_2-6",
    "shuttle-c2-vs-c4",
}

# Eleven complexity metrics; must be present in both input CSVs.
FEATURE_COLS = [
    "F1", "overlap_region_count", "mean_feature_relevance",
    "N3", "mean_margin", "outlier_percentage", "N1",
    "decision_boundary_density", "local_density_ratio",
    "cluster_compactness_ratio", "imbalance_ratio",
]

# Three target columns aggregated from the per-classifier benchmark.
TARGET_COLS = ["accuracy", "f1", "auc"]


def get_dataset_family(name: str) -> str:
    """Extract a coarse 'family' label so that variants of the same base
    dataset can be tracked across the train/val partition (e.g. all
    `yeast*` datasets share the family `yeast`).
    """
    parts = name.split("-")
    if len(parts) > 2:
        return parts[0]
    base = re.split(r"[\d_-]", name)[0]
    return base if base else name


def _attach_targets(df: pd.DataFrame, perf_path: str,
                    classifier_subset: list[str] | None) -> pd.DataFrame:
    """Compute mean accuracy/f1/auc per dataset from a per-(dataset, classifier)
    performance CSV, restricted to a chosen classifier subset.
    """
    perf = pd.read_csv(perf_path)
    if classifier_subset is not None:
        perf = perf[perf["classifier"].isin(classifier_subset)]
    agg = perf.groupby("dataset")[TARGET_COLS].mean().reset_index()
    return df.merge(agg, on="dataset", how="inner")


def build_combined_pool(metrics_train: str, metrics_val: str,
                        perf_train: str, perf_val: str,
                        classifier_subset: list[str] | None) -> pd.DataFrame:
    """Build the 70-dataset pool with features and aggregated targets.

    The original-pool metrics file (`metrics_train`) and the
    pre-reserved validation metrics file (`metrics_val`) are stacked
    after dropping the three excluded datasets. Mean targets are then
    attached from the matching per-classifier performance files.
    """
    om = pd.read_csv(metrics_train)
    vm = pd.read_csv(metrics_val)
    om["source"] = "original_train"
    vm["source"] = "original_val"
    combined = pd.concat([om, vm], ignore_index=True)
    combined = combined[~combined["dataset"].isin(EXCLUDED_DATASETS)].copy()

    # Targets: aggregate per-dataset from the two performance files.
    perf_train_df = _attach_targets(
        combined[combined["source"] == "original_train"],
        perf_train, classifier_subset,
    )
    perf_val_df = _attach_targets(
        combined[combined["source"] == "original_val"],
        perf_val, classifier_subset,
    )
    combined = pd.concat([perf_train_df, perf_val_df], ignore_index=True)
    combined["family"] = combined["dataset"].apply(get_dataset_family)
    return combined


def stratified_split(combined: pd.DataFrame, test_size: int = TEST_SIZE,
                     seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 53/17 split on IR x N1 strata.

    Falls back to IR-only (4 quantile bins) stratification if any 6-cell
    stratum is too small for StratifiedShuffleSplit, matching the
    original implementation.
    """
    df = combined.copy()
    df["ir_stratum"] = pd.qcut(df["imbalance_ratio"], q=3,
                               labels=["low_ir", "mid_ir", "high_ir"],
                               duplicates="drop")
    df["n1_stratum"] = pd.qcut(df["N1"], q=2,
                               labels=["low_n1", "high_n1"],
                               duplicates="drop")
    df["stratum"] = df["ir_stratum"].astype(str) + "_" + df["n1_stratum"].astype(str)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        for train_idx, val_idx in sss.split(df, df["stratum"].values):
            new_train, new_val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    except ValueError:
        df["ir_stratum2"] = pd.qcut(df["imbalance_ratio"], q=4,
                                    labels=False, duplicates="drop")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
        for train_idx, val_idx in sss.split(df, df["ir_stratum2"].values):
            new_train, new_val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    return new_train, new_val


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-train", default="overlap_metrics.csv")
    ap.add_argument("--metrics-val",   default="validation_metrics.csv")
    ap.add_argument("--perf-train",    default="combined_overlap_performance.csv")
    ap.add_argument("--perf-val",      default="validation_classification.csv")
    ap.add_argument("--out-dir",       default="data")
    ap.add_argument("--classifiers", nargs="*", default=None,
                    help="If given, restrict target aggregation to this "
                         "classifier subset (the 9 retained classifiers).")
    args = ap.parse_args()

    combined = build_combined_pool(
        args.metrics_train, args.metrics_val,
        args.perf_train,   args.perf_val,
        args.classifiers,
    )
    print(f"Combined pool: {len(combined)} datasets "
          f"(after excluding {len(EXCLUDED_DATASETS)} non-tractable datasets)")
    new_train, new_val = stratified_split(combined)
    print(f"Split: {len(new_train)} train + {len(new_val)} val")

    os.makedirs(args.out_dir, exist_ok=True)
    cols = ["dataset"] + FEATURE_COLS + TARGET_COLS
    new_train[cols].to_csv(os.path.join(args.out_dir, "train_split.csv"),
                           index=False)
    new_val[cols].to_csv(os.path.join(args.out_dir, "val_split.csv"),
                         index=False)
    print(f"Wrote {args.out_dir}/train_split.csv and {args.out_dir}/val_split.csv")


if __name__ == "__main__":
    main()
