"""
Predictive Modeling Paper: Complete Pipeline with External Validation
======================================================================

This script:
1. Loads original 58 datasets results
2. Scans imbalanced KEEL folders for NEW validation datasets (not in original 58)
3. Computes overlap metrics on new datasets
4. Runs classification experiments on new datasets
5. Trains predictive models on original 58
6. Validates on new datasets
7. Also performs LODO CV for comparison
8. Generates figures and report

Author: Sami
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              RandomForestRegressor, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             r2_score, mean_squared_error, mean_absolute_error)
from sklearn.neighbors import LocalOutlierFactor
import os
import glob
import json
import zipfile
import shutil
from collections import defaultdict
from datetime import datetime
import warnings
import re

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
BASE_PATH = "/Users/samikaya/Documents/VSCode/Benchmark"

CONFIG = {
    # Source folders with ALL KEEL datasets (including zips)
    'imbalanced_paths': [
        f"{BASE_PATH}/imbalanced_ir9_part1",
        f"{BASE_PATH}/imbalanced_ir9_part2",
        f"{BASE_PATH}/imbalanced_ir9_part3",
    ],
    
    # Folder with already-used 58 datasets
    'used_datasets_folder': f"{BASE_PATH}/extended_datasets",
    
    # Original study results
    'original_results_file': f"{BASE_PATH}/classification_results/combined_overlap_performance.csv",
    
    # Output directory
    'output_dir': f"{BASE_PATH}/predictive_modeling_results",
    
    # How many validation datasets to select
    'n_validation_datasets': 20,
    
    # Cross-validation folds for classification
    'cv_folds': 5,
    
    # Random state
    'random_state': 42,
}

# Feature columns (overlap/complexity metrics)
FEATURE_COLS = [
    'F1', 'overlap_region_count', 'mean_feature_relevance', 
    'N3', 'mean_margin', 'outlier_percentage', 'N1',
    'decision_boundary_density', 'local_density_ratio', 
    'cluster_compactness_ratio', 'imbalance_ratio'
]

# Target columns
TARGET_COLS = ['accuracy', 'f1', 'auc']


# =============================================================================
# PART 1: DATASET SCANNING AND SELECTION (adapted from select_datasets.py)
# =============================================================================

def is_macos_artifact(path):
    """Check if path is a macOS artifact"""
    return '__MACOSX' in path or '/._' in path or os.path.basename(path).startswith('._')


def extract_all_zips(folder_path):
    """Recursively find and extract all .zip files"""
    if not os.path.exists(folder_path):
        return
    
    zip_count = 0
    for root, dirs, files in os.walk(folder_path):
        if is_macos_artifact(root):
            continue
        
        for f in files:
            if f.endswith('.zip') and not f.startswith('._'):
                zip_path = os.path.join(root, f)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(root)
                    zip_count += 1
                except Exception as e:
                    pass
    
    if zip_count > 0:
        print(f"  Extracted {zip_count} zip files in {os.path.basename(folder_path)}")


def pick_best_dat_file(dat_files):
    """From a list of .dat files, pick the best one (prefer non-tra/tst, then -tra)"""
    if not dat_files:
        return None
    
    dat_files = [f for f in dat_files if not is_macos_artifact(f)]
    
    if not dat_files:
        return None
    
    # First try to find a file without tra/tst
    for f in dat_files:
        fname = os.path.basename(f).lower()
        if '-tra' not in fname and '-tst' not in fname and 'tra.' not in fname and 'tst.' not in fname:
            return f
    
    # Next, prefer -tra files (training set has more samples)
    for f in dat_files:
        fname = os.path.basename(f).lower()
        if '-tra' in fname or 'tra.' in fname:
            return f
    
    return dat_files[0]


def parse_keel_file(filepath):
    """Parse KEEL .dat file to get basic info - validates binary classification"""
    if is_macos_artifact(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        n_features = 0
        n_samples = 0
        class_counts = defaultdict(int)
        data_started = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('@attribute') and 'class' not in line.lower():
                n_features += 1
            elif line.startswith('@data'):
                data_started = True
            elif data_started and line and not line.startswith('@') and not line.startswith('%'):
                n_samples += 1
                parts = line.split(',')
                if parts:
                    label = parts[-1].strip()
                    class_counts[label] += 1
        
        # Must have exactly 2 classes for binary classification
        n_classes = len(class_counts)
        if n_classes != 2:
            return None
        
        counts = list(class_counts.values())
        ir = max(counts) / min(counts)
        
        # Both classes must have at least 5 samples
        if min(counts) < 5:
            return None
        
        return {
            'n_features': n_features,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'imbalance_ratio': ir,
            'minority_count': min(counts),
            'majority_count': max(counts),
        }
    except Exception as e:
        return None


def load_keel_dataset(filepath):
    """Load a KEEL .dat dataset file into a DataFrame"""
    if is_macos_artifact(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except:
        return None
    
    data = []
    feature_names = []
    data_started = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('@attribute') and not data_started:
            parts = line.split()
            if len(parts) > 1:
                feature_names.append(parts[1])
        
        if line.lower().startswith('@data'):
            data_started = True
            continue
        
        if data_started and line and not line.startswith('%'):
            if '%' in line:
                line = line.split('%')[0].strip()
            if line:
                values = line.split(',')
                data.append(values)
    
    if len(data) == 0:
        return None
    
    df = pd.DataFrame(data)
    
    if feature_names and len(feature_names) == len(df.columns):
        df.columns = feature_names
    
    # Get class column
    class_col = df.columns[-1]
    unique_classes = df[class_col].unique()
    
    if len(unique_classes) != 2:
        return None
    
    # Convert to numeric
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    if len(df) < 50:
        return None
    
    # Convert class to binary (1 = minority, 0 = majority)
    class_counts = df[class_col].value_counts()
    minority_class = class_counts.idxmin()
    df[class_col] = (df[class_col] == minority_class).astype(int)
    
    return df


def get_used_dataset_names(used_folder):
    """Get names of already-used datasets from the extended_datasets folder"""
    used_names = set()
    
    if not os.path.exists(used_folder):
        print(f"WARNING: Used datasets folder not found: {used_folder}")
        return used_names
    
    for f in os.listdir(used_folder):
        if f.endswith('.dat') and not f.startswith('._'):
            name = os.path.splitext(f)[0]
            # Normalize for matching
            used_names.add(name.lower().strip())
            # Also add without -tra/-tst suffix
            name_clean = re.sub(r'[-_](tra|tst)$', '', name.lower().strip())
            used_names.add(name_clean)
    
    print(f"Found {len(used_names)} used dataset names in {used_folder}")
    return used_names


def scan_imbalanced_folders(imbalanced_paths, used_names):
    """Scan imbalanced folders for NEW datasets not already used"""
    candidates = []
    
    for folder_path in imbalanced_paths:
        if not os.path.exists(folder_path):
            print(f"  Folder not found: {folder_path}")
            continue
        
        print(f"  Scanning {os.path.basename(folder_path)}...")
        
        # Extract zips first
        extract_all_zips(folder_path)
        
        # Group .dat files by subfolder
        folder_files = defaultdict(list)
        
        for root, dirs, files in os.walk(folder_path):
            if is_macos_artifact(root):
                continue
            
            for f in files:
                if f.endswith('.dat') and not f.startswith('._'):
                    filepath = os.path.join(root, f)
                    folder_name = os.path.basename(root)
                    folder_files[folder_name].append(filepath)
        
        # Process each subfolder
        for folder_name, dat_files in folder_files.items():
            best_file = pick_best_dat_file(dat_files)
            if not best_file:
                continue
            
            dataset_name = os.path.splitext(os.path.basename(best_file))[0]
            
            # Check if already used
            name_lower = dataset_name.lower().strip()
            name_clean = re.sub(r'[-_](tra|tst)$', '', name_lower)
            
            if name_lower in used_names or name_clean in used_names:
                continue
            
            # Parse and validate
            info = parse_keel_file(best_file)
            if info is None:
                continue
            
            candidates.append({
                'name': dataset_name,
                'filepath': best_file,
                'n_samples': info['n_samples'],
                'n_features': info['n_features'],
                'imbalance_ratio': info['imbalance_ratio'],
                'minority_count': info['minority_count'],
                'majority_count': info['majority_count'],
            })
    
    print(f"  Found {len(candidates)} NEW candidate datasets")
    return candidates


def select_validation_datasets(candidates, n_target):
    """Select diverse validation datasets"""
    if not candidates:
        return []
    
    print(f"\nSelecting {n_target} validation datasets...")
    
    # Sort by imbalance ratio for diversity
    df = pd.DataFrame(candidates)
    df = df.sort_values('imbalance_ratio')
    
    selected = []
    selected_names = set()
    
    # Try to get diverse IR range
    n_per_bucket = n_target // 4 + 1
    
    ir_buckets = [
        (0, 15, 'low'),
        (15, 25, 'medium'),
        (25, 40, 'high'),
        (40, np.inf, 'extreme')
    ]
    
    for ir_min, ir_max, bucket_name in ir_buckets:
        bucket = df[(df['imbalance_ratio'] >= ir_min) & (df['imbalance_ratio'] < ir_max)]
        bucket = bucket[~bucket['name'].isin(selected_names)]
        
        for _, row in bucket.head(n_per_bucket).iterrows():
            if len(selected) >= n_target:
                break
            selected.append(row.to_dict())
            selected_names.add(row['name'])
        
        print(f"  {bucket_name} IR ({ir_min}-{ir_max}): selected {min(len(bucket), n_per_bucket)}")
    
    # If still need more, add remaining
    if len(selected) < n_target:
        remaining = df[~df['name'].isin(selected_names)]
        for _, row in remaining.iterrows():
            if len(selected) >= n_target:
                break
            selected.append(row.to_dict())
            selected_names.add(row['name'])
    
    print(f"Total selected: {len(selected)} validation datasets")
    return selected


# =============================================================================
# PART 2: OVERLAP METRICS COMPUTATION
# =============================================================================

def compute_overlap_metrics(df):
    """Compute all overlap/complexity metrics for a dataset."""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    metrics = {}
    n_samples = len(df)
    
    # Basic info
    metrics['n_samples'] = n_samples
    metrics['n_features'] = X.shape[1]
    metrics['minority_class_count'] = int(np.sum(y == 1))
    metrics['majority_class_count'] = int(np.sum(y == 0))
    metrics['imbalance_ratio'] = metrics['majority_class_count'] / max(metrics['minority_class_count'], 1)
    
    classes = np.unique(y)
    X_0, X_1 = X[y == classes[0]], X[y == classes[1]]
    mean_0, mean_1 = np.mean(X_0, axis=0), np.mean(X_1, axis=0)
    var_0, var_1 = np.var(X_0, axis=0), np.var(X_1, axis=0)
    denom = var_0 + var_1
    denom[denom == 0] = 1e-10
    fisher_ratios = np.square(mean_0 - mean_1) / denom
    
    metrics['F1'] = float(np.max(fisher_ratios))
    metrics['mean_feature_relevance'] = float(np.mean(fisher_ratios))
    
    # Overlap region count
    overlap_count = 0
    for i in range(X.shape[1]):
        f_0, f_1 = X[y == classes[0], i], X[y == classes[1], i]
        if np.min(f_0) <= np.max(f_1) and np.max(f_0) >= np.min(f_1):
            overlap_count += 1
    metrics['overlap_region_count'] = overlap_count
    
    # N3: LOO error of 1-NN (approximate for large datasets)
    sample_size = min(500, n_samples)
    if n_samples > 500:
        idx = np.random.choice(n_samples, 500, replace=False)
        X_sample, y_sample = X[idx], y[idx]
    else:
        X_sample, y_sample = X, y
    
    knn = KNeighborsClassifier(n_neighbors=1)
    errors = 0
    for i in range(len(X_sample)):
        mask = np.ones(len(X_sample), dtype=bool)
        mask[i] = False
        knn.fit(X_sample[mask], y_sample[mask])
        if knn.predict(X_sample[i:i+1])[0] != y_sample[i]:
            errors += 1
    metrics['N3'] = errors / len(X_sample)
    
    # N1: Fraction with heterogeneous neighborhood
    k = min(5, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    hetero_count = 0
    for i in range(n_samples):
        neighbor_labels = y[indices[i, 1:]]
        if len(np.unique(neighbor_labels)) > 1 or neighbor_labels[0] != y[i]:
            hetero_count += 1
    metrics['N1'] = hetero_count / n_samples
    
    # Margin metrics
    try:
        svm = SVC(kernel='linear', max_iter=1000)
        svm.fit(X, y)
        margins = np.abs(svm.decision_function(X))
        metrics['mean_margin'] = float(np.mean(margins))
        threshold = 0.1 * np.std(margins)
        metrics['decision_boundary_density'] = float(np.sum(margins < threshold) / n_samples)
    except:
        metrics['mean_margin'] = 1.0
        metrics['decision_boundary_density'] = 0.1
    
    # Outlier metrics
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, n_samples-1))
        labels = lof.fit_predict(X)
        metrics['outlier_percentage'] = float(np.sum(labels == -1) / n_samples)
    except:
        metrics['outlier_percentage'] = 0.0
    
    # Local density ratio
    try:
        nn = NearestNeighbors(n_neighbors=min(6, n_samples-1))
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        ratios = []
        for i in range(n_samples):
            neighbor_labels = y[indices[i, 1:]]
            same = np.sum(neighbor_labels == y[i])
            diff = len(neighbor_labels) - same
            ratios.append((same + 1) / (diff + 1))
        metrics['local_density_ratio'] = float(np.mean(ratios))
    except:
        metrics['local_density_ratio'] = 1.0
    
    # Cluster compactness
    try:
        intra = []
        for c in classes:
            X_c = X[y == c]
            if len(X_c) > 1:
                centroid = np.mean(X_c, axis=0)
                dists = np.sqrt(np.sum((X_c - centroid) ** 2, axis=1))
                intra.extend(dists)
        centroids = [np.mean(X[y == c], axis=0) for c in classes]
        inter = np.sqrt(np.sum((centroids[0] - centroids[1]) ** 2))
        metrics['cluster_compactness_ratio'] = float(np.mean(intra) / inter) if inter > 0 else 1.0
    except:
        metrics['cluster_compactness_ratio'] = 1.0
    
    return metrics


# =============================================================================
# PART 3: CLASSIFICATION EXPERIMENTS
# =============================================================================

def get_classifiers():
    """Get dictionary of classifiers."""
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    try:
        from xgboost import XGBClassifier
        classifiers['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, 
                                                use_label_encoder=False, eval_metric='logloss')
    except ImportError:
        pass
    
    try:
        from lightgbm import LGBMClassifier
        classifiers['LightGBM'] = LGBMClassifier(n_estimators=100, class_weight='balanced', 
                                                  random_state=42, verbose=-1)
    except ImportError:
        pass
    
    return classifiers


def run_classification(df, cv_folds=5, random_state=42):
    """Run classification experiments on a dataset."""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    classifiers = get_classifiers()
    results = []
    
    for clf_name, clf in classifiers.items():
        try:
            from sklearn.base import clone
            clf_clone = clone(clf)
            
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            acc_scores, f1_scores, auc_scores = [], [], []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                clf_clone.fit(X_train, y_train)
                y_pred = clf_clone.predict(X_test)
                
                acc_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
                
                try:
                    if hasattr(clf_clone, 'predict_proba'):
                        y_prob = clf_clone.predict_proba(X_test)[:, 1]
                    else:
                        y_prob = clf_clone.decision_function(X_test)
                    auc_scores.append(roc_auc_score(y_test, y_prob))
                except:
                    auc_scores.append(np.nan)
            
            results.append({
                'classifier': clf_name,
                'accuracy': np.mean(acc_scores),
                'f1': np.mean(f1_scores),
                'auc': np.nanmean(auc_scores),
            })
            
        except Exception as e:
            continue
    
    return results


# =============================================================================
# PART 4: PREDICTIVE MODELING
# =============================================================================

def perform_lodo_cv(df_agg, feature_cols, target_cols):
    """Leave-One-Dataset-Out Cross-Validation."""
    lodo_results = {}
    
    for target in target_cols:
        valid_df = df_agg[df_agg[target].notna()].copy()
        n_valid = len(valid_df)
        
        if n_valid < 10:
            continue
        
        datasets = valid_df['dataset'].values
        y_true_all, y_pred_all = [], []
        
        for held_out in datasets:
            train_df = valid_df[valid_df['dataset'] != held_out]
            test_df = valid_df[valid_df['dataset'] == held_out]
            
            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            y_true_all.append(y_test[0])
            y_pred_all.append(y_pred[0])
        
        y_true = np.array(y_true_all)
        y_pred = np.array(y_pred_all)
        
        lodo_results[target] = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'n_datasets': n_valid,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    return lodo_results


def train_predictive_models(train_df, feature_cols, target_cols):
    """Train predictive models on training data."""
    models = {}
    feature_importance = {}
    
    for target in target_cols:
        valid_df = train_df[train_df[target].notna()]
        X = valid_df[feature_cols].values
        y = valid_df[target].values
        
        if len(y) < 10:
            continue
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
        
        models[target] = {
            'model': rf,
            'cv_r2': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores)
        }
        
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance[target] = imp_df
    
    return models, feature_importance


def validate_on_external(models, val_df, feature_cols, target_cols):
    """Validate predictive models on external validation data."""
    val_results = {}
    
    for target in target_cols:
        if target not in models or target not in val_df.columns:
            continue
        
        valid_df = val_df[val_df[target].notna()]
        
        if len(valid_df) < 3:
            continue
        
        X = valid_df[feature_cols].values
        y = valid_df[target].values
        
        model = models[target]['model']
        y_pred = model.predict(X)
        
        val_results[target] = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'n_datasets': len(y),
            'y_true': y,
            'y_pred': y_pred
        }
    
    return val_results


# =============================================================================
# PART 5: VISUALIZATION AND REPORTING
# =============================================================================

def create_visualizations(lodo_results, val_results, feature_importance, output_dir):
    """Create all figures."""
    
    # Figure 1: LODO CV - Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, target in enumerate(TARGET_COLS):
        if target not in lodo_results:
            continue
        
        ax = axes[idx]
        r = lodo_results[target]
        
        ax.scatter(r['y_true'], r['y_pred'], alpha=0.6, edgecolors='k', linewidth=0.5, s=60)
        
        min_val = min(min(r['y_true']), min(r['y_pred']))
        max_val = max(max(r['y_true']), max(r['y_pred']))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'Actual {target.upper()}', fontsize=12)
        ax.set_ylabel(f'Predicted {target.upper()}', fontsize=12)
        ax.set_title(f'{target.upper()} (LODO)\nRÂ² = {r["r2"]:.3f}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_lodo_predictions.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_lodo_predictions.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_lodo_predictions.png/pdf")
    
    # Figure 2: External Validation
    if val_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, target in enumerate(TARGET_COLS):
            if target not in val_results:
                axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{target.upper()} (External)')
                continue
            
            ax = axes[idx]
            r = val_results[target]
            
            ax.scatter(r['y_true'], r['y_pred'], alpha=0.6, edgecolors='k', linewidth=0.5, s=60, color='green')
            
            min_val = min(min(r['y_true']), min(r['y_pred']))
            max_val = max(max(r['y_true']), max(r['y_pred']))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            ax.set_xlabel(f'Actual {target.upper()}', fontsize=12)
            ax.set_ylabel(f'Predicted {target.upper()}', fontsize=12)
            ax.set_title(f'{target.upper()} (External)\nRÂ² = {r["r2"]:.3f}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig2_external_validation.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'fig2_external_validation.pdf'), bbox_inches='tight')
        plt.close()
        print("  Saved: fig2_external_validation.png/pdf")
    
    # Figure 3: Feature Importance
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, target in enumerate(TARGET_COLS):
        if target not in feature_importance:
            continue
        
        ax = axes[idx]
        imp_df = feature_importance[target]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
        ax.barh(range(len(imp_df)), imp_df['importance'].values, color=colors)
        ax.set_yticks(range(len(imp_df)))
        ax.set_yticklabels(imp_df['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance: {target.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_feature_importance.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_feature_importance.png/pdf")
    
    # Figure 4: Comparison
    if val_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        targets = [t for t in TARGET_COLS if t in lodo_results and t in val_results]
        if targets:
            x = np.arange(len(targets))
            width = 0.35
            
            lodo_r2 = [lodo_results[t]['r2'] for t in targets]
            val_r2 = [val_results[t]['r2'] for t in targets]
            
            bars1 = ax.bar(x - width/2, lodo_r2, width, label='LODO CV (58 datasets)', color='steelblue')
            bars2 = ax.bar(x + width/2, val_r2, width, label='External Validation', color='coral')
            
            ax.set_ylabel('RÂ² Score', fontsize=12)
            ax.set_title('Model Generalization: LODO vs External', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([t.upper() for t in targets])
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fig4_lodo_vs_external.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'fig4_lodo_vs_external.pdf'), bbox_inches='tight')
            plt.close()
            print("  Saved: fig4_lodo_vs_external.png/pdf")


def generate_report(lodo_results, val_results, feature_importance, 
                   n_train, n_val, val_datasets, output_dir):
    """Generate comprehensive report."""
    
    report = f"""================================================================================
PREDICTIVE MODELING PAPER: COMPLETE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

EXPERIMENTAL SETUP
--------------------------------------------------------------------------------
Training datasets (original study): {n_train}
Validation datasets (new):          {n_val}
Features used:                      {len(FEATURE_COLS)}
Target metrics:                     {', '.join(TARGET_COLS)}

"""
    
    if val_datasets:
        report += "VALIDATION DATASETS SELECTED:\n"
        for ds in val_datasets:
            report += f"  - {ds['name']}: {ds['n_samples']} samples, IR={ds['imbalance_ratio']:.2f}\n"
        report += "\n"
    
    report += """================================================================================
LEAVE-ONE-DATASET-OUT CROSS-VALIDATION (Training Data)
================================================================================

Train on 57 datasets, predict on held-out dataset. Repeat for all 58.

"""
    
    for target in TARGET_COLS:
        if target in lodo_results:
            r = lodo_results[target]
            report += f"""
{target.upper()}:
  RÂ² Score:      {r['r2']:.4f}  (explains {r['r2']*100:.1f}% of variance)
  RMSE:          {r['rmse']:.4f}
  MAE:           {r['mae']:.4f}
  N datasets:    {r['n_datasets']}
"""
    
    if val_results:
        report += """
================================================================================
EXTERNAL VALIDATION (New Datasets)
================================================================================

Model trained on ALL 58 original datasets, validated on NEW unseen datasets.

"""
        
        for target in TARGET_COLS:
            if target in val_results:
                r = val_results[target]
                report += f"""
{target.upper()}:
  RÂ² Score:      {r['r2']:.4f}  (explains {r['r2']*100:.1f}% of variance)
  RMSE:          {r['rmse']:.4f}
  MAE:           {r['mae']:.4f}
  N datasets:    {r['n_datasets']}
"""
    
    report += """
================================================================================
FEATURE IMPORTANCE (Random Forest)
================================================================================
"""
    
    for target in TARGET_COLS:
        if target in feature_importance:
            imp_df = feature_importance[target]
            report += f"\n{target.upper()} - Top 5 Predictors:\n"
            for rank, (_, row) in enumerate(imp_df.head(5).iterrows(), 1):
                report += f"  {rank}. {row['feature']}: {row['importance']:.4f}\n"
    
    report += """
================================================================================
END OF REPORT
================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
        f.write(report)
    
    print(report)
    return report


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PREDICTIVE MODELING PAPER: COMPLETE PIPELINE")
    print("=" * 70)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load original study data
    # =========================================================================
    print("\n[STEP 1] Loading original study data...")
    
    original_df = pd.read_csv(CONFIG['original_results_file'])
    print(f"  Loaded: {len(original_df)} rows, {original_df['dataset'].nunique()} datasets")
    
    train_agg = original_df.groupby('dataset')[FEATURE_COLS + TARGET_COLS].mean().reset_index()
    print(f"  Aggregated to {len(train_agg)} dataset-level observations")
    
    # =========================================================================
    # STEP 2: Find new validation datasets
    # =========================================================================
    print("\n[STEP 2] Scanning for new validation datasets...")
    
    used_names = get_used_dataset_names(CONFIG['used_datasets_folder'])
    candidates = scan_imbalanced_folders(CONFIG['imbalanced_paths'], used_names)
    validation_datasets = select_validation_datasets(candidates, CONFIG['n_validation_datasets'])
    
    # =========================================================================
    # STEP 3: Process validation datasets
    # =========================================================================
    if validation_datasets:
        print(f"\n[STEP 3] Processing {len(validation_datasets)} validation datasets...")
        
        val_metrics_list = []
        val_clf_results = []
        
        for i, ds in enumerate(validation_datasets):
            print(f"  [{i+1}/{len(validation_datasets)}] {ds['name']}")
            
            # Load dataset
            df = load_keel_dataset(ds['filepath'])
            if df is None:
                print(f"    Skipped: Could not load")
                continue
            
            ds['dataframe'] = df
            
            # Compute overlap metrics
            print(f"    Computing metrics...")
            metrics = compute_overlap_metrics(df)
            metrics['dataset'] = ds['name']
            val_metrics_list.append(metrics)
            
            # Run classification
            print(f"    Running classification...")
            clf_results = run_classification(df, CONFIG['cv_folds'], CONFIG['random_state'])
            for r in clf_results:
                r['dataset'] = ds['name']
            val_clf_results.extend(clf_results)
        
        # Save validation results
        val_metrics_df = pd.DataFrame(val_metrics_list)
        val_metrics_df.to_csv(os.path.join(CONFIG['output_dir'], 'validation_metrics.csv'), index=False)
        
        val_clf_df = pd.DataFrame(val_clf_results)
        val_clf_df.to_csv(os.path.join(CONFIG['output_dir'], 'validation_classification.csv'), index=False)
        
        # Aggregate validation results
        val_agg = val_clf_df.groupby('dataset')[TARGET_COLS].mean().reset_index()
        val_combined = val_metrics_df.merge(val_agg, on='dataset', how='left')
        
        print(f"  Processed {len(val_metrics_list)} validation datasets")
    else:
        val_combined = None
        print("\n[STEP 3] No validation datasets found. Proceeding with LODO only.")
    
    # =========================================================================
    # STEP 4: LODO Cross-Validation
    # =========================================================================
    print("\n[STEP 4] Leave-One-Dataset-Out Cross-Validation...")
    
    lodo_results = perform_lodo_cv(train_agg, FEATURE_COLS, TARGET_COLS)
    
    for target, r in lodo_results.items():
        print(f"  {target.upper()}: RÂ² = {r['r2']:.4f}, RMSE = {r['rmse']:.4f}")
        
        pred_df = pd.DataFrame({'y_true': r['y_true'], 'y_pred': r['y_pred']})
        pred_df.to_csv(os.path.join(CONFIG['output_dir'], f'lodo_predictions_{target}.csv'), index=False)
    
    # =========================================================================
    # STEP 5: Train and validate
    # =========================================================================
    print("\n[STEP 5] Training predictive models...")
    
    models, feature_importance = train_predictive_models(train_agg, FEATURE_COLS, TARGET_COLS)
    
    for target, m in models.items():
        print(f"  {target.upper()}: CV RÂ² = {m['cv_r2']:.4f} Â± {m['cv_r2_std']:.4f}")
    
    for target, imp_df in feature_importance.items():
        imp_df.to_csv(os.path.join(CONFIG['output_dir'], f'feature_importance_{target}.csv'), index=False)
    
    # External validation
    if val_combined is not None:
        print("\n[STEP 6] External validation...")
        
        val_results = validate_on_external(models, val_combined, FEATURE_COLS, TARGET_COLS)
        
        for target, r in val_results.items():
            print(f"  {target.upper()}: RÂ² = {r['r2']:.4f}, RMSE = {r['rmse']:.4f}")
            
            pred_df = pd.DataFrame({'y_true': r['y_true'], 'y_pred': r['y_pred']})
            pred_df.to_csv(os.path.join(CONFIG['output_dir'], f'external_predictions_{target}.csv'), index=False)
    else:
        val_results = None
    
    # =========================================================================
    # STEP 7: Visualizations and report
    # =========================================================================
    print("\n[STEP 7] Creating visualizations and report...")
    
    create_visualizations(lodo_results, val_results, feature_importance, CONFIG['output_dir'])
    
    generate_report(
        lodo_results, val_results, feature_importance,
        n_train=len(train_agg),
        n_val=len(validation_datasets) if validation_datasets else 0,
        val_datasets=validation_datasets,
        output_dir=CONFIG['output_dir']
    )
    
    # Save summary
    summary = {
        'n_training_datasets': int(len(train_agg)),
        'n_validation_datasets': int(len(validation_datasets)) if validation_datasets else 0,
        'lodo_results': {t: {'r2': float(r['r2']), 'rmse': float(r['rmse'])} for t, r in lodo_results.items()},
        'external_results': {t: {'r2': float(r['r2']), 'rmse': float(r['rmse'])} for t, r in val_results.items()} if val_results else None,
        'top_features': {t: feature_importance[t].head(3)['feature'].tolist() for t in feature_importance}
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {CONFIG['output_dir']}/")


if __name__ == "__main__":
    main()