import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ClassificationExperiment:
    def __init__(self, datasets_path, overlap_metrics_path):
        self.datasets_path = datasets_path
        self.overlap_metrics_path = overlap_metrics_path
        self.results = []
        self.datasets = []
        self.overlap_metrics = None
        
    def load_overlap_metrics(self):
        """Load the existing overlap metrics from CSV"""
        print("Loading overlap metrics...")
        self.overlap_metrics = pd.read_csv(self.overlap_metrics_path)
        print(f"Loaded overlap metrics for {len(self.overlap_metrics)} datasets")
        return self.overlap_metrics
    
    def parse_keel_file(self, filepath):
        """Parse KEEL .dat file format"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        # Find where @data section starts
        data_start_idx = None
        attributes = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 3:
                    attr_name = parts[1]
                    attributes.append(attr_name)
            elif line.startswith('@data'):
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError(f"No @data section found in {filepath}")
        
        # Parse the data
        X = []
        y = []
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('@'):
                values = line.split(',')
                if len(values) > 1:
                    features = []
                    for v in values[:-1]:
                        try:
                            features.append(float(v))
                        except ValueError:
                            features.append(np.nan)
                    
                    # Handle class label
                    label = values[-1].strip()
                    if label in ['positive', 'Positive', '1', '1.0']:
                        y.append(1)
                    else:
                        y.append(0)
                    
                    X.append(features)
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle missing values
        if np.any(np.isnan(X)):
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        return X, y
    
    def load_datasets(self):
        """Load all KEEL datasets"""
        print("\nLoading datasets...")
        dataset_files = [f for f in os.listdir(self.datasets_path) if f.endswith('.dat')]
        
        for file in tqdm(dataset_files, desc="Loading datasets"):
            dataset_name = file.replace('.dat', '')
            if dataset_name in self.overlap_metrics['dataset'].values:
                filepath = os.path.join(self.datasets_path, file)
                try:
                    X, y = self.parse_keel_file(filepath)
                    self.datasets.append({
                        'name': dataset_name,
                        'X': X,
                        'y': y,
                        'n_samples': len(y),
                        'n_features': X.shape[1],
                        'imbalance_ratio': sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else np.inf
                    })
                except Exception as e:
                    print(f"Error loading {dataset_name}: {str(e)}")
        
        print(f"Successfully loaded {len(self.datasets)} datasets")
    
    def get_classifiers(self):
        """Initialize all classifiers with appropriate parameters"""
        classifiers = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                          random_state=42, early_stopping=True),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                                  class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, 
                                        use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, 
                                          class_weight='balanced', verbose=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, 
                                                    class_weight='balanced'),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, 
                                              class_weight='balanced'),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), 
                                       n_estimators=50, random_state=42)
        }
        
        # Add Stacking Classifier
        base_learners = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ]
        classifiers['Stacking'] = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        
        return classifiers
    
    def evaluate_classifier(self, clf, X, y):
        """Evaluate classifier using 5-fold cross-validation"""
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': lambda clf, X, y: precision_score(y, clf.predict(X), zero_division=0),
            'recall': lambda clf, X, y: recall_score(y, clf.predict(X), zero_division=0),
            'f1': lambda clf, X, y: f1_score(y, clf.predict(X), zero_division=0),
            'roc_auc': 'roc_auc'
        }
        
        try:
            # Perform cross-validation
            cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, 
                                      return_train_score=False, n_jobs=-1)
            
            # Calculate log loss separately (requires probability predictions)
            log_losses = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_train, y_train)
                
                if hasattr(clf_copy, 'predict_proba'):
                    y_proba = clf_copy.predict_proba(X_test)
                    log_losses.append(log_loss(y_test, y_proba))
                else:
                    log_losses.append(np.nan)
            
            # Aggregate results
            results = {
                'accuracy': np.mean(cv_results['test_accuracy']),
                'precision': np.mean(cv_results['test_precision']),
                'recall': np.mean(cv_results['test_recall']),
                'f1': np.mean(cv_results['test_f1']),
                'auc': np.mean(cv_results['test_roc_auc']),
                'log_loss': np.nanmean(log_losses) if log_losses else np.nan,
                'accuracy_std': np.std(cv_results['test_accuracy']),
                'f1_std': np.std(cv_results['test_f1'])
            }
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            results = {
                'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan,
                'f1': np.nan, 'auc': np.nan, 'log_loss': np.nan,
                'accuracy_std': np.nan, 'f1_std': np.nan
            }
        
        return results
    
    def run_experiments(self):
        """Run classification experiments on all datasets"""
        classifiers = self.get_classifiers()
        total_experiments = len(self.datasets) * len(classifiers)
        
        print(f"\nRunning {total_experiments} experiments...")
        print(f"Datasets: {len(self.datasets)}, Classifiers: {len(classifiers)}")
        
        with tqdm(total=total_experiments, desc="Running experiments") as pbar:
            for dataset in self.datasets:
                X, y = dataset['X'], dataset['y']
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                for clf_name, clf in classifiers.items():
                    # Evaluate classifier
                    metrics = self.evaluate_classifier(clf, X_scaled, y)
                    
                    # Store results
                    result = {
                        'dataset': dataset['name'],
                        'classifier': clf_name,
                        **metrics
                    }
                    self.results.append(result)
                    pbar.update(1)
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        print("\nExperiments completed!")
        return self.results_df
    
    def merge_with_overlap_metrics(self):
        """Merge classification results with overlap metrics"""
        print("\nMerging results with overlap metrics...")
        self.combined_df = pd.merge(
            self.results_df,
            self.overlap_metrics,
            on='dataset',
            how='inner'
        )
        print(f"Combined dataset shape: {self.combined_df.shape}")
        return self.combined_df
    
    def analyze_correlations(self):
        """Analyze correlations between overlap metrics and performance"""
        print("\nAnalyzing correlations...")
        
        # Select overlap and performance columns
        overlap_cols = ['F1', 'overlap_region_count', 'mean_feature_relevance',
                       'N3', 'mean_margin', 'outlier_percentage',
                       'N1', 'decision_boundary_density', 'local_density_ratio',
                       'cluster_compactness_ratio']
        
        performance_cols = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Calculate correlations for each classifier
        correlations = {}
        for clf in self.combined_df['classifier'].unique():
            clf_data = self.combined_df[self.combined_df['classifier'] == clf]
            corr_matrix = clf_data[overlap_cols + performance_cols].corr()
            correlations[clf] = corr_matrix.loc[overlap_cols, performance_cols]
        
        return correlations
    
    def build_predictive_models(self):
        """Build regression models to predict performance from overlap metrics"""
        print("\nBuilding predictive models...")
        
        overlap_features = ['F1', 'overlap_region_count', 'mean_feature_relevance',
                          'N3', 'mean_margin', 'outlier_percentage',
                          'N1', 'decision_boundary_density', 'local_density_ratio',
                          'cluster_compactness_ratio', 'imbalance_ratio']
        
        target_metrics = ['accuracy', 'f1', 'auc']
        regression_results = {}
        
        for target in target_metrics:
            print(f"\nPredicting {target}...")
            results_by_classifier = {}
            
            for clf in self.combined_df['classifier'].unique():
                clf_data = self.combined_df[self.combined_df['classifier'] == clf]
                
                X = clf_data[overlap_features].values
                y = clf_data[target].values
                
                # Remove NaN values
                mask = ~np.isnan(y)
                X = X[mask]
                y = y[mask]
                
                if len(y) < 10:  # Skip if too few samples
                    continue
                
                # Train multiple regression models
                models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': Ridge(alpha=1.0),
                    'Lasso Regression': Lasso(alpha=0.1),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
                }
                
                model_results = {}
                for model_name, model in models.items():
                    # Use leave-one-out cross-validation for small datasets
                    from sklearn.model_selection import LeaveOneOut
                    loo = LeaveOneOut()
                    
                    y_true = []
                    y_pred = []
                    
                    for train_idx, test_idx in loo.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        model_copy = model.__class__(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        pred = model_copy.predict(X_test)
                        
                        y_true.extend(y_test)
                        y_pred.extend(pred)
                    
                    # Calculate metrics
                    r2 = r2_score(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    model_results[model_name] = {
                        'r2': r2,
                        'mse': mse,
                        'mae': mae,
                        'model': model.fit(X, y)  # Fit on full data for feature importance
                    }
                
                results_by_classifier[clf] = model_results
            
            regression_results[target] = results_by_classifier
        
        return regression_results
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Create output directory
        os.makedirs('classification_results', exist_ok=True)
        
        # 1. Performance distribution by classifier
        plt.figure(figsize=(15, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 3, i)
            self.results_df.boxplot(column=metric, by='classifier', rot=45)
            plt.title(f'{metric.capitalize()} by Classifier')
            plt.suptitle('')
            plt.tight_layout()
        
        plt.savefig('classification_results/performance_by_classifier.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmaps for each classifier
        correlations = self.analyze_correlations()
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, (clf_name, corr_matrix) in enumerate(correlations.items()):
            if idx < len(axes):
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                          fmt='.2f', ax=axes[idx], cbar_kws={'shrink': 0.8})
                axes[idx].set_title(f'{clf_name}')
                axes[idx].set_xlabel('Performance Metrics')
                axes[idx].set_ylabel('Overlap Metrics')
        
        # Hide unused subplots
        for idx in range(len(correlations), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('classification_results/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plots of key relationships
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        key_relationships = [
            ('N1', 'f1', 'N1 (Borderline Fraction) vs F1-Score'),
            ('N3', 'accuracy', 'N3 (1-NN Error Rate) vs Accuracy'),
            ('mean_margin', 'auc', 'Mean Margin vs AUC'),
            ('local_density_ratio', 'f1', 'Local Density Ratio vs F1-Score'),
            ('cluster_compactness_ratio', 'accuracy', 'Cluster Compactness vs Accuracy'),
            ('imbalance_ratio', 'recall', 'Imbalance Ratio vs Recall')
        ]
        
        for idx, (x_col, y_col, title) in enumerate(key_relationships):
            if idx < len(axes):
                for clf in self.combined_df['classifier'].unique():
                    clf_data = self.combined_df[self.combined_df['classifier'] == clf]
                    axes[idx].scatter(clf_data[x_col], clf_data[y_col], 
                                    label=clf, alpha=0.6, s=50)
                
                axes[idx].set_xlabel(x_col)
                axes[idx].set_ylabel(y_col)
                axes[idx].set_title(title)
                axes[idx].grid(True, alpha=0.3)
                
                if idx == 0:  # Add legend to first plot
                    axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('classification_results/scatter_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature importance from Random Forest predictive models
        regression_results = self.build_predictive_models()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        overlap_features = ['F1', 'overlap_region_count', 'mean_feature_relevance',
                          'N3', 'mean_margin', 'outlier_percentage',
                          'N1', 'decision_boundary_density', 'local_density_ratio',
                          'cluster_compactness_ratio', 'imbalance_ratio']
        
        for idx, target in enumerate(['accuracy', 'f1', 'auc']):
            # Average feature importance across classifiers
            all_importances = []
            
            for clf_name, models in regression_results[target].items():
                if 'Random Forest' in models:
                    rf_model = models['Random Forest']['model']
                    if hasattr(rf_model, 'feature_importances_'):
                        all_importances.append(rf_model.feature_importances_)
            
            if all_importances:
                avg_importance = np.mean(all_importances, axis=0)
                importance_df = pd.DataFrame({
                    'feature': overlap_features,
                    'importance': avg_importance
                }).sort_values('importance', ascending=True)
                
                importance_df.plot(x='feature', y='importance', kind='barh', 
                                 ax=axes[idx], legend=False)
                axes[idx].set_title(f'Feature Importance for Predicting {target.upper()}')
                axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('classification_results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to 'classification_results' directory")
    
    def save_results(self):
        """Save all results for next research phase"""
        print("\nSaving results...")
        
        # Save combined dataset
        self.combined_df.to_csv('classification_results/combined_overlap_performance.csv', index=False)
        
        # Save summary statistics
        summary = self.results_df.groupby('classifier')[
            ['accuracy', 'precision', 'recall', 'f1', 'auc']
        ].agg(['mean', 'std'])
        summary.to_csv('classification_results/classifier_summary_stats.csv')
        
        # Save regression model performance
        regression_performance = []
        regression_results = self.build_predictive_models()
        
        for target, clf_results in regression_results.items():
            for clf_name, models in clf_results.items():
                for model_name, results in models.items():
                    regression_performance.append({
                        'target_metric': target,
                        'classifier': clf_name,
                        'regression_model': model_name,
                        'r2': results['r2'],
                        'mse': results['mse'],
                        'mae': results['mae']
                    })
        
        pd.DataFrame(regression_performance).to_csv(
            'classification_results/regression_model_performance.csv', index=False
        )
        
        # Create a comprehensive report
        self.create_report()
        
        print("All results saved successfully!")
    
    def create_report(self):
        """Create a comprehensive text report of findings"""
        with open('classification_results/analysis_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLASS OVERLAP AND CLASSIFICATION PERFORMANCE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total datasets analyzed: {len(self.datasets)}\n")
            f.write(f"Total experiments run: {len(self.results)}\n")
            f.write(f"Classifiers tested: {len(self.results_df['classifier'].unique())}\n\n")
            
            # Best performing classifiers
            f.write("BEST PERFORMING CLASSIFIERS (by average metrics)\n")
            f.write("-" * 40 + "\n")
            summary = self.results_df.groupby('classifier')[
                ['accuracy', 'f1', 'auc']
            ].mean().sort_values('f1', ascending=False)
            f.write(summary.to_string())
            f.write("\n\n")
            
            # Key correlations
            f.write("KEY CORRELATIONS (averaged across classifiers)\n")
            f.write("-" * 40 + "\n")
            correlations = self.analyze_correlations()
            
            # Find strongest correlations
            strong_correlations = []
            for clf_name, corr_matrix in correlations.items():
                for overlap_metric in corr_matrix.index:
                    for perf_metric in corr_matrix.columns:
                        corr_value = corr_matrix.loc[overlap_metric, perf_metric]
                        if abs(corr_value) > 0.3:  # Threshold for "strong" correlation
                            strong_correlations.append({
                                'classifier': clf_name,
                                'overlap_metric': overlap_metric,
                                'performance_metric': perf_metric,
                                'correlation': corr_value
                            })
            
            strong_corr_df = pd.DataFrame(strong_correlations)
            if not strong_corr_df.empty:
                avg_correlations = strong_corr_df.groupby(
                    ['overlap_metric', 'performance_metric']
                )['correlation'].mean().sort_values(ascending=False)
                
                f.write("Strong correlations (|r| > 0.3):\n")
                for (overlap, perf), corr in avg_correlations.items():
                    f.write(f"  {overlap} <-> {perf}: {corr:.3f}\n")
            f.write("\n")
            
            # Predictive model performance
            f.write("PREDICTIVE MODEL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            regression_results = self.build_predictive_models()
            
            for target in ['accuracy', 'f1', 'auc']:
                f.write(f"\nPredicting {target.upper()}:\n")
                best_r2 = -np.inf
                best_model = None
                
                for clf_name, models in regression_results[target].items():
                    for model_name, results in models.items():
                        if results['r2'] > best_r2:
                            best_r2 = results['r2']
                            best_model = (clf_name, model_name)
                
                if best_model:
                    f.write(f"  Best model: {best_model[1]} for {best_model[0]} classifier\n")
                    f.write(f"  RÂ² score: {best_r2:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")


def main():
    """Main execution function"""
    # Initialize experiment
    datasets_path = "/Users/samikaya/Documents/VSCode/Benchmark/extended_datasets"
    overlap_metrics_path = "overlap_metrics.csv"
    
    experiment = ClassificationExperiment(datasets_path, overlap_metrics_path)
    
    # Step 1: Load overlap metrics
    print("STEP 1: Loading overlap metrics")
    experiment.load_overlap_metrics()
    
    # Step 2: Load datasets
    print("\nSTEP 2: Loading KEEL datasets")
    experiment.load_datasets()
    
    # Step 3: Run classification experiments
    print("\nSTEP 3: Running classification experiments")
    experiment.run_experiments()
    
    # Step 4: Merge with overlap metrics
    print("\nSTEP 4: Merging results with overlap metrics")
    experiment.merge_with_overlap_metrics()
    
    # Step 5: Analyze correlations
    print("\nSTEP 5: Analyzing correlations")
    correlations = experiment.analyze_correlations()
    
    # Step 6: Build predictive models
    print("\nSTEP 6: Building predictive models")
    regression_results = experiment.build_predictive_models()
    
    # Step 7: Create visualizations
    print("\nSTEP 7: Creating visualizations")
    experiment.visualize_results()
    
    # Step 8: Save all results
    print("\nSTEP 8: Saving results")
    experiment.save_results()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults saved in 'classification_results' directory:")
    print("- combined_overlap_performance.csv: Full dataset with overlap and performance metrics")
    print("- classifier_summary_stats.csv: Summary statistics for each classifier")
    print("- regression_model_performance.csv: Performance of predictive models")
    print("- analysis_report.txt: Comprehensive text report")
    print("- Various visualization PNG files")
    print("\nYou can now proceed to Step 4: Mitigation Strategies")


if __name__ == "__main__":
    main()