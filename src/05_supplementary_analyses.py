"""
Supplementary Analyses for "Estimating Classification Difficulty from
Data Complexity" (Paper 2)
=====================================================================

This script reproduces the secondary analyses reported in the manuscript
that are not produced by 04_predictive_modeling_complete.py:

  A. Nine-family regressor comparison (Section 3.4)
       -> regression_model_performance.csv

  B. Distribution-shift diagnostics on the 11-metric feature vector,
     comparing the stratified 53/17 split against the original 58/15 split
     (Section 4.2)
       -> ks_test_results.csv

  C. Triangulated feature-importance: Gini + permutation + SHAP,
     evaluated on both the GradientBoosting (primary) and RandomForest
     (supplementary) meta-regressors (Section 4.4, Table 6)
       -> importance_gini_{accuracy,f1,auc}.csv
       -> importance_permutation_{accuracy,f1,auc}.csv
       -> importance_shap_{accuracy,f1,auc}.csv
       -> importance_triangulation_summary.csv

  D. Bootstrap rank stability (50 resamples, SHAP-based) (Section 4.4)
       -> bootstrap_rank_stability.csv

  E. RandomForest supplementary meta-regressor (LODO + external)
     (Section 4.1 / Section 4.2 robustness check)
       -> rf_supplementary_lodo.csv
       -> rf_supplementary_external.csv

  F. Algorithm selection gap analysis on the 53 training datasets
     (Section 4.5, Table 8)
       -> gap_analysis.csv

Inputs
------
  train_split.csv  : 53 training datasets, 11 features + 3 targets
                     (one row per dataset; targets are means across the
                      9-classifier pool)
  val_split.csv    : 17 stratified held-out validation datasets,
                     same schema as train_split.csv
  combined_overlap_performance.csv : per-(dataset, classifier) rows used
                                     by the gap analysis (Section F).
                                     Only the 53 training datasets and
                                     the 9 retained classifiers are used.

Outputs
-------
All CSVs are written to OUTPUT_DIR (default: ./results_supplementary/).
The script is deterministic given the fixed RNG seed below.

Usage
-----
  python 05_supplementary_analyses.py

Run after 04_predictive_modeling_complete.py has produced the splits and
the per-(dataset, classifier) performance file.

Author: Sami Kaya
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, kendalltau

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42

# 11 complexity metrics, in the order reported in the manuscript
FEATURE_COLS = [
    "F1",
    "overlap_region_count",
    "mean_feature_relevance",
    "N3",
    "mean_margin",
    "outlier_percentage",
    "N1",
    "decision_boundary_density",
    "local_density_ratio",
    "cluster_compactness_ratio",
    "imbalance_ratio",
]

# Targets are mean classifier performance across the 9-classifier pool
TARGET_COLS = ["accuracy", "f1", "auc"]

# The 9 classifiers retained after exclusion of MLP / Bagging / Stacking.
# These names must match the strings used in combined_overlap_performance.csv.
NINE_CLASSIFIERS = [
    "Logistic Regression",
    "kNN",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "LightGBM",
    "SVM",
    "Naive Bayes",
]


# =============================================================================
# UTILITIES
# =============================================================================
def _load_splits(train_path: str, val_path: str):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    for col in FEATURE_COLS + TARGET_COLS + ["dataset"]:
        if col not in train_df.columns:
            raise ValueError(f"train split missing column: {col}")
        if col not in val_df.columns:
            raise ValueError(f"val split missing column: {col}")
    return train_df, val_df


def _xy(df: pd.DataFrame, target: str):
    sub = df[df[target].notna()].copy()
    X = sub[FEATURE_COLS].to_numpy()
    y = sub[target].to_numpy()
    return X, y, sub


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# =============================================================================
# A. NINE-FAMILY REGRESSOR COMPARISON
# =============================================================================
def regressor_zoo():
    """Return the nine regressor families compared in Section 3.4."""
    return {
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Ridge":            Ridge(random_state=RANDOM_STATE),
        "Lasso":            Lasso(random_state=RANDOM_STATE),
        "ElasticNet":       ElasticNet(random_state=RANDOM_STATE),
        "OLS":              LinearRegression(),
        "kNN":              KNeighborsRegressor(n_neighbors=5),
        "SVR":              SVR(kernel="rbf"),
    }


def _lodo_predict(model_factory, X: np.ndarray, y: np.ndarray):
    """Generic leave-one-dataset-out CV: returns y_true, y_pred arrays."""
    y_true_all, y_pred_all = [], []
    n = len(y)
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        X_tr, X_te = X[train_idx], X[i:i + 1]
        y_tr, y_te = y[train_idx], y[i:i + 1]

        model = model_factory()
        # Linear / SVR / kNN benefit from scaling; tree models don't but
        # are scale-invariant so the standardization does no harm.
        scaler = StandardScaler().fit(X_tr)
        model.fit(scaler.transform(X_tr), y_tr)
        y_hat = model.predict(scaler.transform(X_te))

        y_true_all.append(float(y_te[0]))
        y_pred_all.append(float(y_hat[0]))
    return np.array(y_true_all), np.array(y_pred_all)


def run_regressor_comparison(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    """Section A — LODO and external R² for nine regressor families."""
    rows = []
    for target in TARGET_COLS:
        X_tr, y_tr, _ = _xy(train_df, target)
        X_va, y_va, _ = _xy(val_df, target)

        for name, est in regressor_zoo().items():
            # LODO on training pool
            factory = lambda e=est.__class__, p=est.get_params(): e(**p)
            y_true, y_pred = _lodo_predict(factory, X_tr, y_tr)
            lodo = _metrics(y_true, y_pred)

            # External fit (train on full 53, predict on 17)
            scaler = StandardScaler().fit(X_tr)
            est_full = factory()
            est_full.fit(scaler.transform(X_tr), y_tr)
            y_va_pred = est_full.predict(scaler.transform(X_va))
            ext = _metrics(y_va, y_va_pred)

            rows.append({
                "target": target,
                "regressor": name,
                "lodo_r2":   lodo["r2"],   "lodo_rmse":   lodo["rmse"],   "lodo_mae":   lodo["mae"],
                "external_r2": ext["r2"],  "external_rmse": ext["rmse"],  "external_mae": ext["mae"],
                "lodo_external_gap": lodo["r2"] - ext["r2"],
            })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "regression_model_performance.csv")
    df.to_csv(path, index=False)
    print(f"[A] regressor comparison written: {path}  ({len(df)} rows)")
    return df


# =============================================================================
# B. DISTRIBUTION-SHIFT DIAGNOSTICS (KS TESTS)
# =============================================================================
def run_ks_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    """Section B — KS tests on the 11-metric feature vector, train vs val.

    Reports the per-feature KS statistic and two-sided p-value for the
    stratified 53/17 split. The manuscript reports 0/11 significant at
    alpha = 0.05 for this split; the earlier 58/15 unstratified split
    returned 11/11 significant.
    """
    rows = []
    for feat in FEATURE_COLS:
        a = train_df[feat].dropna().to_numpy()
        b = val_df[feat].dropna().to_numpy()
        stat, p = ks_2samp(a, b)
        rows.append({
            "feature": feat,
            "ks_statistic": float(stat),
            "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05),
            "train_mean": float(np.mean(a)),
            "val_mean": float(np.mean(b)),
            "train_n": int(len(a)),
            "val_n": int(len(b)),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "ks_test_results.csv")
    df.to_csv(path, index=False)
    sig = int(df["significant_at_0.05"].sum())
    print(f"[B] KS diagnostics written: {path}  ({sig}/{len(df)} features significant)")
    return df


# =============================================================================
# C. FEATURE-IMPORTANCE TRIANGULATION
# =============================================================================
def _gini_importance(model, feature_names: Iterable[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=list(feature_names))


def _permutation_importance(model, X: np.ndarray, y: np.ndarray,
                            feature_names: Iterable[str], n_repeats: int = 20) -> pd.Series:
    res = permutation_importance(
        model, X, y,
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="r2",
    )
    return pd.Series(res.importances_mean, index=list(feature_names))


def _shap_importance(model, X: np.ndarray, feature_names: Iterable[str]) -> pd.Series:
    """Mean |SHAP value| per feature, using TreeExplainer."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for the SHAP importance analysis. "
            "Install with: pip install shap"
        ) from exc
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # legacy multi-output return
        sv = sv[0]
    return pd.Series(np.abs(sv).mean(axis=0), index=list(feature_names))


def _meta_models():
    """Primary and supplementary meta-regressors used in Section 4.4."""
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE),
    }


def run_importance_triangulation(train_df: pd.DataFrame, out_dir: str):
    """Section C — Gini + permutation + SHAP across GB and RF meta-regressors.

    Writes a per-target wide table of normalized importances for the
    primary GradientBoosting model and a summary table of top-3 agreement
    counts across the six (model x method) configurations.
    """
    summary_rows = []
    for target in TARGET_COLS:
        X, y, _ = _xy(train_df, target)
        per_target = {}

        for model_name, est in _meta_models().items():
            est.fit(X, y)

            gini = _gini_importance(est, FEATURE_COLS)
            perm = _permutation_importance(est, X, y, FEATURE_COLS)
            shap = _shap_importance(est, X, FEATURE_COLS)

            def _norm(s: pd.Series) -> pd.Series:
                total = s.sum()
                return s / total if total > 0 else s

            per_target[(model_name, "gini")] = _norm(gini)
            per_target[(model_name, "permutation")] = _norm(perm)
            per_target[(model_name, "shap")] = _norm(shap)

        # write the GB primary table (one row per feature, one column per method)
        gb_table = pd.DataFrame({
            "feature": FEATURE_COLS,
            "gini":         per_target[("GradientBoosting", "gini")].values,
            "permutation":  per_target[("GradientBoosting", "permutation")].values,
            "shap":         per_target[("GradientBoosting", "shap")].values,
        })
        gb_path = os.path.join(out_dir, f"importance_gb_{target}.csv")
        gb_table.to_csv(gb_path, index=False)
        print(f"[C] GB importance ({target}) written: {gb_path}")

        # write the RF supplementary table
        rf_table = pd.DataFrame({
            "feature": FEATURE_COLS,
            "gini":         per_target[("RandomForest", "gini")].values,
            "permutation":  per_target[("RandomForest", "permutation")].values,
            "shap":         per_target[("RandomForest", "shap")].values,
        })
        rf_path = os.path.join(out_dir, f"importance_rf_{target}.csv")
        rf_table.to_csv(rf_path, index=False)
        print(f"[C] RF importance ({target}) written: {rf_path}")

        # top-3 agreement count + Kendall tau across the six rankings
        rankings = {
            f"{m}_{meth}": s.sort_values(ascending=False).index.tolist()
            for (m, meth), s in per_target.items()
        }
        top3_sets = [set(r[:3]) for r in rankings.values()]
        # Count features that appear in the top-3 of all six rankings
        common_top3 = set.intersection(*top3_sets)

        # Pairwise Kendall tau on the full rankings (rank vectors)
        keys = list(rankings.keys())
        taus = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ri = pd.Series(range(len(FEATURE_COLS)), index=rankings[keys[i]])
                rj = pd.Series(range(len(FEATURE_COLS)), index=rankings[keys[j]])
                tau, _ = kendalltau(ri.reindex(FEATURE_COLS).values,
                                    rj.reindex(FEATURE_COLS).values)
                taus.append(tau)

        summary_rows.append({
            "target": target,
            "common_top3_features": ", ".join(sorted(common_top3)),
            "n_common_top3": len(common_top3),
            "mean_pairwise_kendall_tau": float(np.mean(taus)),
            "min_pairwise_kendall_tau": float(np.min(taus)),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "importance_triangulation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[C] triangulation summary written: {summary_path}")
    return summary


# =============================================================================
# D. BOOTSTRAP RANK STABILITY (SHAP)
# =============================================================================
def run_bootstrap_rank_stability(train_df: pd.DataFrame, out_dir: str,
                                 n_resamples: int = 50):
    """Section D — bootstrap rank stability under SHAP on the GB primary model.

    Resamples the 53 training datasets with replacement, refits the
    GradientBoosting meta-regressor, recomputes SHAP-based feature
    rankings, and reports the standard deviation of each feature's rank.
    The manuscript reports sigma < 1.2 for mean_margin and sigma > 2.0
    for several features beyond rank 3.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for the bootstrap rank stability analysis. "
            "Install with: pip install shap"
        ) from exc

    rng = np.random.default_rng(RANDOM_STATE)
    all_rows = []

    for target in TARGET_COLS:
        X, y, _ = _xy(train_df, target)
        n = len(y)
        rank_matrix = np.zeros((n_resamples, len(FEATURE_COLS)), dtype=int)

        for b in range(n_resamples):
            idx = rng.integers(0, n, size=n)  # bootstrap with replacement
            Xb, yb = X[idx], y[idx]
            model = GradientBoostingRegressor(n_estimators=100,
                                              random_state=RANDOM_STATE)
            model.fit(Xb, yb)
            sv = shap.TreeExplainer(model).shap_values(Xb)
            if isinstance(sv, list):
                sv = sv[0]
            mean_abs = np.abs(sv).mean(axis=0)
            # rank 1 = most important
            order = np.argsort(-mean_abs)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(FEATURE_COLS) + 1)
            rank_matrix[b] = ranks

        rank_std = rank_matrix.std(axis=0)
        rank_mean = rank_matrix.mean(axis=0)
        for feat, rm, rs in zip(FEATURE_COLS, rank_mean, rank_std):
            all_rows.append({
                "target": target,
                "feature": feat,
                "mean_rank": float(rm),
                "rank_std": float(rs),
                "n_resamples": n_resamples,
            })

    df = pd.DataFrame(all_rows).sort_values(["target", "mean_rank"])
    path = os.path.join(out_dir, "bootstrap_rank_stability.csv")
    df.to_csv(path, index=False)
    print(f"[D] bootstrap rank stability written: {path}  "
          f"({n_resamples} resamples per target)")
    return df


# =============================================================================
# E. RANDOMFOREST SUPPLEMENTARY META-REGRESSOR
# =============================================================================
def run_rf_supplementary(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str):
    """Section E — RandomForest LODO and external fit, reported as a
    robustness check against the GradientBoosting primary model.
    """
    lodo_rows, ext_rows = [], []
    for target in TARGET_COLS:
        X_tr, y_tr, _ = _xy(train_df, target)
        X_va, y_va, _ = _xy(val_df, target)

        # LODO
        factory = lambda: RandomForestRegressor(n_estimators=100,
                                                random_state=RANDOM_STATE)
        y_true, y_pred = _lodo_predict(factory, X_tr, y_tr)
        m = _metrics(y_true, y_pred)
        lodo_rows.append({"target": target, **m, "n": len(y_true)})

        # External
        rf = factory()
        rf.fit(X_tr, y_tr)
        y_va_pred = rf.predict(X_va)
        m2 = _metrics(y_va, y_va_pred)
        ext_rows.append({"target": target, **m2, "n": len(y_va)})

    lodo_df = pd.DataFrame(lodo_rows)
    ext_df = pd.DataFrame(ext_rows)
    lodo_path = os.path.join(out_dir, "rf_supplementary_lodo.csv")
    ext_path  = os.path.join(out_dir, "rf_supplementary_external.csv")
    lodo_df.to_csv(lodo_path, index=False)
    ext_df.to_csv(ext_path, index=False)
    print(f"[E] RF supplementary LODO written: {lodo_path}")
    print(f"[E] RF supplementary external written: {ext_path}")
    return lodo_df, ext_df


# =============================================================================
# F. ALGORITHM SELECTION GAP ANALYSIS
# =============================================================================
def run_gap_analysis(perf_path: str, train_df: pd.DataFrame, out_dir: str):
    """Section F — per-dataset best vs second-best gap across the 9 classifiers.

    Requires combined_overlap_performance.csv with per-(dataset, classifier)
    rows. Filters to the 53 training datasets and the 9 retained classifiers.

    For each (dataset, target), computes:
      best_score, second_best_score, gap = best - second_best
    The dataset-level gap is then summarized in the manuscript by its
    median, mean, and the share below thresholds {0.02, 0.05}.
    """
    if not os.path.exists(perf_path):
        print(f"[F] SKIPPED: {perf_path} not found")
        return None

    df = pd.read_csv(perf_path)
    if "classifier" not in df.columns or "dataset" not in df.columns:
        print(f"[F] SKIPPED: {perf_path} missing required columns")
        return None

    training_datasets = set(train_df["dataset"])
    df = df[df["dataset"].isin(training_datasets)]
    df = df[df["classifier"].isin(NINE_CLASSIFIERS)]

    rows = []
    for target in ["accuracy", "f1", "auc"]:
        if target not in df.columns:
            continue
        for ds, grp in df.groupby("dataset"):
            scores = grp[["classifier", target]].dropna()
            if len(scores) < 2:
                continue
            sorted_scores = scores.sort_values(target, ascending=False)
            best = float(sorted_scores.iloc[0][target])
            second = float(sorted_scores.iloc[1][target])
            best_clf = str(sorted_scores.iloc[0]["classifier"])
            second_clf = str(sorted_scores.iloc[1]["classifier"])
            rows.append({
                "dataset": ds,
                "target": target,
                "best_classifier": best_clf,
                "best_score": best,
                "second_classifier": second_clf,
                "second_score": second,
                "gap": best - second,
            })

    gap_df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "gap_analysis.csv")
    gap_df.to_csv(path, index=False)
    # report quick summary so the user can sanity-check against the manuscript
    for target in ["f1", "accuracy", "auc"]:
        sub = gap_df[gap_df["target"] == target]
        if len(sub) == 0:
            continue
        median_gap = sub["gap"].median()
        mean_gap = sub["gap"].mean()
        share_lt_005 = float((sub["gap"] < 0.05).mean())
        share_lt_002 = float((sub["gap"] < 0.02).mean())
        print(f"[F] {target}: median gap = {median_gap:.4f}, "
              f"mean = {mean_gap:.4f}, "
              f"share<0.05 = {share_lt_005:.3f}, share<0.02 = {share_lt_002:.3f}")
    print(f"[F] gap analysis written: {path}  ({len(gap_df)} rows)")
    return gap_df


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Supplementary analyses for Paper 2.")
    parser.add_argument("--train", default="train_split.csv",
                        help="Path to the 53-dataset training split CSV.")
    parser.add_argument("--val",   default="val_split.csv",
                        help="Path to the 17-dataset external validation CSV.")
    parser.add_argument("--perf",  default="combined_overlap_performance.csv",
                        help="Per-(dataset, classifier) performance CSV; "
                             "used only by the gap analysis.")
    parser.add_argument("--out",   default="results_supplementary",
                        help="Output directory.")
    parser.add_argument("--bootstrap-resamples", type=int, default=50,
                        help="Number of bootstrap resamples for Section D.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=list("ABCDEF"),
                        help="Skip one or more analyses by letter.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_df, val_df = _load_splits(args.train, args.val)
    print(f"Loaded {len(train_df)} training and {len(val_df)} validation datasets.")

    if "A" not in args.skip:
        run_regressor_comparison(train_df, val_df, args.out)
    if "B" not in args.skip:
        run_ks_diagnostics(train_df, val_df, args.out)
    if "C" not in args.skip:
        run_importance_triangulation(train_df, args.out)
    if "D" not in args.skip:
        run_bootstrap_rank_stability(train_df, args.out,
                                     n_resamples=args.bootstrap_resamples)
    if "E" not in args.skip:
        run_rf_supplementary(train_df, val_df, args.out)
    if "F" not in args.skip:
        run_gap_analysis(args.perf, train_df, args.out)


if __name__ == "__main__":
    main()
