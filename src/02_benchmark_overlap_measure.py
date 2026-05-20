import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
import glob
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_keel_dataset(file_path):
    """
    Load a KEEL dataset from the given file path.
    KEEL datasets typically have a specific format with attribute information and data.
    """
    data = []
    feature_names = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_start = False
        
        for line in lines:
            line = line.strip()
            
            # Extract feature names if available
            if line.startswith('@attribute') and not data_start:
                parts = line.split()
                if len(parts) > 1:
                    feature_names.append(parts[1])
            
            if line.startswith('@data') or line == '@data':
                data_start = True
                continue
                
            if data_start and line:
                # Remove comments if any
                if '%' in line:
                    line = line.split('%')[0].strip()
                if line:
                    data.append(line.split(','))
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    
    # If feature names were found, use them
    if feature_names and len(feature_names) == len(df.columns):
        df.columns = feature_names
    
    # Convert class labels to binary (0, 1)
    # Assuming last column is the class label
    last_col = df.columns[-1]
    # Get unique class labels
    unique_labels = df[last_col].unique()
    if len(unique_labels) != 2:
        print(f"  WARNING: Expected 2 classes, found {len(unique_labels)}: {unique_labels}")
        return None

    # Map to binary: minority class = 1, majority class = 0
    label_counts = df[last_col].value_counts()
    minority_label = label_counts.idxmin()
    df[last_col] = df[last_col].map(lambda x: 1 if x == minority_label else 0)
    
    # Convert features to float
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle any missing values
    df = df.dropna()
    
    return df

# Alternative implementations for class overlap metrics
def fisher_discriminant_ratio(X, y):
    """
    Calculate Fisher's Discriminant Ratio for each feature and return the maximum.
    Higher values indicate better class separation.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This implementation is for binary classification only")
    
    # Get samples for each class
    X_0 = X[y == classes[0]]
    X_1 = X[y == classes[1]]
    
    # Calculate mean for each feature in each class
    mean_0 = np.mean(X_0, axis=0)
    mean_1 = np.mean(X_1, axis=0)
    
    # Calculate variance for each feature in each class
    var_0 = np.var(X_0, axis=0)
    var_1 = np.var(X_1, axis=0)
    
    # Calculate Fisher's ratio for each feature
    # F = (mean1 - mean2)^2 / (var1 + var2)
    numerator = np.square(mean_0 - mean_1)
    denominator = var_0 + var_1
    
    # Handle division by zero
    denominator[denominator == 0] = 1e-10
    
    fisher_ratios = numerator / denominator
    
    # Return the maximum ratio
    return np.max(fisher_ratios)

def count_overlap_regions(X, y):
    """
    Count regions where feature values overlap between classes.
    A simple implementation that checks histogram overlap for each feature.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This implementation is for binary classification only")
    
    overlap_count = 0
    n_features = X.shape[1]
    
    for i in range(n_features):
        # Get feature values for each class
        f_0 = X[y == classes[0], i]
        f_1 = X[y == classes[1], i]
        
        # Calculate min and max for each class
        min_0, max_0 = np.min(f_0), np.max(f_0)
        min_1, max_1 = np.min(f_1), np.max(f_1)
        
        # Check if there's overlap
        if (min_0 <= max_1 and max_0 >= min_1):
            overlap_count += 1
    
    return overlap_count

def feature_relevance(X, y):
    """
    Calculate feature relevance using a Random Forest classifier.
    Returns the feature importances.
    """
    try:
        # Train a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Return feature importances
        return rf.feature_importances_
    except Exception as e:
        print(f"Error in feature_relevance: {e}")
        return np.zeros(X.shape[1])

def n3_error_rate(X, y):
    """
    Calculate the leave-one-out error rate of the 1-NN classifier (N3 metric).
    Higher values indicate more class overlap.
    """
    n_samples = X.shape[0]
    
    # Initialize 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    
    # Calculate leave-one-out error
    errors = 0
    for i in range(n_samples):
        # Train on all samples except i
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        # Test sample i
        X_test = X[i:i+1]
        y_test = y[i:i+1]
        
        # Train and predict
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Check if prediction is correct
        if y_pred[0] != y_test[0]:
            errors += 1
    
    # Return error rate
    return errors / n_samples

def margin_distribution(X, y):
    """
    Calculate the margin of each instance to the decision boundary.
    Uses an SVM to approximate the decision boundary.
    """
    # Train an SVM with probability estimates
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X, y)
    
    # Get decision function values (distance to hyperplane)
    margins = np.abs(svm.decision_function(X))
    
    return margins

def n1_borderline_points(X, y):
    """
    Calculate the fraction of borderline points (N1 metric).
    Uses k-NN to identify points near the decision boundary.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This implementation is for binary classification only")
    
    n_samples = X.shape[0]
    k = min(5, n_samples // 2)  # k should be smaller than half the dataset
    
    # Initialize nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    nn.fit(X)
    
    # Find k nearest neighbors for each point
    distances, indices = nn.kneighbors(X)
    
    # Count points with different class neighbors
    borderline_count = 0
    for i in range(n_samples):
        # Get neighbors (excluding the point itself)
        neighbors = indices[i, 1:]
        
        # Count neighbors with different class
        diff_class_count = np.sum(y[neighbors] != y[i])
        
        # If more than half of neighbors are from different class, it's a borderline point
        if diff_class_count >= k / 2:
            borderline_count += 1
    
    # Return fraction of borderline points
    return borderline_count / n_samples

def decision_boundary_density(X, y):
    """
    Estimate the density of points near the decision boundary.
    Uses SVM to approximate the decision boundary.
    """
    # Train an SVM
    svm = SVC(kernel='linear')
    svm.fit(X, y)
    
    # Get distances to decision boundary
    distances = np.abs(svm.decision_function(X))
    
    # Define threshold for "near decision boundary"
    threshold = np.percentile(distances, 10)  # 10% closest points
    
    # Count points near decision boundary
    near_boundary_count = np.sum(distances <= threshold)
    
    # Return density (fraction of points near boundary)
    return near_boundary_count / X.shape[0]

def local_density(X, y):
    """
    Calculate local density measures for each class.
    Returns the ratio of average densities between classes.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This implementation is for binary classification only")
    
    # Compute average distance to k nearest neighbors for each class
    k = min(5, X.shape[0] // 2)
    
    densities = []
    for c in classes:
        X_c = X[y == c]
        if len(X_c) > k:
            nn = NearestNeighbors(n_neighbors=k+1)
            nn.fit(X_c)
            distances, _ = nn.kneighbors(X_c)
            # Average distance to neighbors (excluding self)
            avg_distance = np.mean(distances[:, 1:])
            # Density is inverse of distance
            densities.append(1 / avg_distance if avg_distance > 0 else float('inf'))
        else:
            densities.append(0)
    
    # Return ratio of densities if possible
    if len(densities) == 2 and densities[1] > 0:
        return densities[0] / densities[1]
    else:
        return 0

def cluster_compactness(X, y):
    """
    Calculate cluster compactness for each class.
    Returns the ratio of compactness between classes.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This implementation is for binary classification only")
    
    # Compute within-class scatter for each class
    compactness = []
    for c in classes:
        X_c = X[y == c]
        if len(X_c) > 1:
            # Compute class centroid
            centroid = np.mean(X_c, axis=0)
            
            # Compute average distance to centroid
            distances = np.sqrt(np.sum((X_c - centroid)**2, axis=1))
            avg_distance = np.mean(distances)
            
            # Compactness is inverse of average distance
            compactness.append(1 / avg_distance if avg_distance > 0 else float('inf'))
        else:
            compactness.append(0)
    
    # Return ratio of compactness if possible
    if len(compactness) == 2 and compactness[1] > 0:
        return compactness[0] / compactness[1]
    else:
        return 0

def compute_overlap_metrics(X, y):
    """
    Compute various overlap metrics for a dataset using custom implementations
    
    Parameters:
    X : feature matrix
    y : class labels
    
    Returns:
    Dictionary of overlap metrics
    """
    metrics = {}
    
    # Feature Overlap Metrics
    try:
        metrics['F1'] = fisher_discriminant_ratio(X, y)
    except Exception as e:
        print(f"Error calculating F1: {e}")
        metrics['F1'] = np.nan
    
    try:
        metrics['overlap_region_count'] = count_overlap_regions(X, y)
    except Exception as e:
        print(f"Error calculating overlap_region_count: {e}")
        metrics['overlap_region_count'] = np.nan
    
    try:
        feature_relevance_values = feature_relevance(X, y)
        metrics['mean_feature_relevance'] = np.mean(feature_relevance_values)
        metrics['min_feature_relevance'] = np.min(feature_relevance_values)
        metrics['max_feature_relevance'] = np.max(feature_relevance_values)
    except Exception as e:
        print(f"Error calculating feature_relevance: {e}")
        metrics['mean_feature_relevance'] = np.nan
        metrics['min_feature_relevance'] = np.nan
        metrics['max_feature_relevance'] = np.nan
    
    # Instance Overlap Metrics
    try:
        if X.shape[0] <= 100:  # Only calculate N3 for small datasets due to computational cost
            metrics['N3'] = n3_error_rate(X, y)
        else:
            # For larger datasets, use a sample
            sample_size = min(100, X.shape[0] // 2)
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            metrics['N3'] = n3_error_rate(X[indices], y[indices])
    except Exception as e:
        print(f"Error calculating N3: {e}")
        metrics['N3'] = np.nan
    
    try:
        margins = margin_distribution(X, y)
        metrics['mean_margin'] = np.mean(margins)
        metrics['std_margin'] = np.std(margins)
        metrics['min_margin'] = np.min(margins)
    except Exception as e:
        print(f"Error calculating margins: {e}")
        metrics['mean_margin'] = np.nan
        metrics['std_margin'] = np.nan
        metrics['min_margin'] = np.nan
    
    try:
        # We'll use Local Outlier Factor for outlier detection
        lof = LocalOutlierFactor(n_neighbors=min(20, X.shape[0]-1))
        outlier_scores = -lof.fit_predict(X)
        metrics['outlier_score_mean'] = np.mean(outlier_scores)
        metrics['outlier_percentage'] = np.sum(outlier_scores == -1) / len(outlier_scores)
    except Exception as e:
        print(f"Error calculating outlier scores: {e}")
        metrics['outlier_score_mean'] = np.nan
        metrics['outlier_percentage'] = np.nan
    
    # Structural Overlap Metrics
    try:
        metrics['N1'] = n1_borderline_points(X, y)
    except Exception as e:
        print(f"Error calculating N1: {e}")
        metrics['N1'] = np.nan
    
    try:
        metrics['decision_boundary_density'] = decision_boundary_density(X, y)
    except Exception as e:
        print(f"Error calculating decision_boundary_density: {e}")
        metrics['decision_boundary_density'] = np.nan
    
    try:
        density_ratio = local_density(X, y)
        metrics['local_density_ratio'] = density_ratio
    except Exception as e:
        print(f"Error calculating local_density: {e}")
        metrics['local_density_ratio'] = np.nan
    
    try:
        metrics['cluster_compactness_ratio'] = cluster_compactness(X, y)
    except Exception as e:
        print(f"Error calculating cluster_compactness: {e}")
        metrics['cluster_compactness_ratio'] = np.nan
    
    return metrics

def analyze_datasets_overlap(dataset_paths):
    """
    Analyze the overlap in multiple datasets and store results
    
    Parameters:
    dataset_paths : dictionary mapping dataset names to file paths
    
    Returns:
    DataFrame with overlap metrics for each dataset
    """
    all_metrics = []
    
    for name, path in dataset_paths.items():
        print(f"Processing dataset: {name}")
        
        try:
            # Load dataset
            df = load_keel_dataset(path)
            
            # Split into features and target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Basic dataset statistics
            n_samples = len(y)
            n_features = X.shape[1]
            imbalance_ratio = np.sum(y == 0) / max(1, np.sum(y == 1))
            
            print(f"  Samples: {n_samples}, Features: {n_features}, Imbalance ratio: {imbalance_ratio:.2f}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Compute overlap metrics
            metrics = compute_overlap_metrics(X_scaled, y)
            
            # Add dataset information
            metrics['dataset'] = name
            metrics['n_samples'] = n_samples
            metrics['n_features'] = n_features
            metrics['imbalance_ratio'] = imbalance_ratio
            metrics['majority_class_count'] = np.sum(y == 0)
            metrics['minority_class_count'] = np.sum(y == 1)
            
            all_metrics.append(metrics)
            print(f"  Completed overlap analysis for {name}")
            
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    return metrics_df

def visualize_overlap_metrics(metrics_df, output_dir='overlap_visualizations'):
    """
    Create visualizations of overlap metrics
    
    Parameters:
    metrics_df : DataFrame with overlap metrics for each dataset
    output_dir : directory to save visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Select only numeric columns, excluding basic dataset statistics
    exclude_cols = ['dataset', 'n_samples', 'n_features', 'majority_class_count', 'minority_class_count']
    numeric_cols = [col for col in metrics_df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # Calculate correlation matrix
    corr = metrics_df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Between Overlap Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_metrics_correlation.png'), dpi=300)
    plt.close()
    
    # 2. Plot each metric across datasets, sorted by metric value
    for metric in numeric_cols:
        if metric != 'imbalance_ratio':
            plt.figure(figsize=(12, 7))
            
            # Sort datasets by metric value and get top 20 (or all if fewer than 20)
            sorted_df = metrics_df.sort_values(by=metric).copy()
            if len(sorted_df) > 20:
                sorted_df = sorted_df.iloc[:20]
            
            # Create bar plot
            ax = sns.barplot(x='dataset', y=metric, data=sorted_df)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{metric} Across Datasets', fontsize=14)
            
            # Add value labels
            for i, v in enumerate(sorted_df[metric]):
                if not np.isnan(v):
                    ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_across_datasets.png'), dpi=300)
            plt.close()
    
    # 3. Scatter plot of metrics vs imbalance ratio
    for metric in numeric_cols:
        if metric != 'imbalance_ratio':
            plt.figure(figsize=(10, 7))
            
            # Create scatter plot with dataset names as tooltips
            scatter = sns.scatterplot(x='imbalance_ratio', y=metric, data=metrics_df, s=80)
            
            # Add dataset names as annotations
            for i, row in metrics_df.iterrows():
                plt.annotate(row['dataset'], 
                            (row['imbalance_ratio'], row[metric]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8)
            
            plt.title(f'{metric} vs Imbalance Ratio', fontsize=14)
            plt.xlabel('Imbalance Ratio', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_vs_imbalance.png'), dpi=300)
            plt.close()
    
    # 4. Create a summary heatmap of all metrics across datasets
    plt.figure(figsize=(16, max(8, len(metrics_df) * 0.4)))
    
    # Normalize the metrics for better visualization
    from sklearn.preprocessing import MinMaxScaler
    
    # Fill any NaN values with the mean of the column
    metrics_filled = metrics_df[numeric_cols].fillna(metrics_df[numeric_cols].mean())
    
    # Apply min-max scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(metrics_filled)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=metrics_df['dataset'])
    
    # Create the heatmap
    sns.heatmap(scaled_df, cmap='viridis', linewidths=0.5, annot=False)
    plt.title('Normalized Overlap Metrics Across Datasets', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_metrics_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def save_metrics(metrics_df, output_path='overlap_metrics.csv'):
    """
    Save metrics DataFrame to a CSV file
    
    Parameters:
    metrics_df : DataFrame with overlap metrics
    output_path : path to save the CSV file
    """
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

def main():
    # Define path to your datasets
    dataset_dir = "/Users/samikaya/Documents/VSCode/Benchmark/extended_datasets"
    
    # Test if directory exists and contains files
    if os.path.exists(dataset_dir):
        print(f"Directory exists: {dataset_dir}")
        
        # List all files in the directory
        all_files = os.listdir(dataset_dir)
        print(f"Total files in directory: {len(all_files)}")
        
        # Check for .dat files specifically
        dat_files = glob.glob(os.path.join(dataset_dir, "*.dat"))
        
        # If no .dat files, look for other common extensions
        if len(dat_files) == 0:
            print("No .dat files found, checking for other extensions...")
            for ext in ['.data', '.txt', '.csv']:
                ext_files = glob.glob(os.path.join(dataset_dir, f"*{ext}"))
                if len(ext_files) > 0:
                    print(f"Found {len(ext_files)} files with extension {ext}")
                    dat_files.extend(ext_files)
        
        print(f"Found {len(dat_files)} dataset files:")
        for file in dat_files[:10]:  # Show first 10 files
            print(f"  - {os.path.basename(file)}")
        
        if len(dat_files) > 10:
            print(f"  ... and {len(dat_files) - 10} more files")
    else:
        print(f"Directory does not exist: {dataset_dir}")
        print("Creating directory...")
        os.makedirs(dataset_dir, exist_ok=True)
    
    # If you have a directory with all datasets
    dataset_paths = {}
    
    # Automatically detect all .dat files in the directory
    try:
        for file_path in glob.glob(os.path.join(dataset_dir, "*.dat")):
            dataset_name = os.path.basename(file_path).split('.')[0]
            dataset_paths[dataset_name] = file_path
            
        # If no .dat files, try other extensions
        if len(dataset_paths) == 0:
            for ext in ['.data', '.txt', '.csv']:
                for file_path in glob.glob(os.path.join(dataset_dir, f"*{ext}")):
                    dataset_name = os.path.basename(file_path).split('.')[0]
                    dataset_paths[dataset_name] = file_path
        
        print(f"Found {len(dataset_paths)} datasets for analysis")
        
        if len(dataset_paths) == 0:
            raise Exception("No dataset files found in the specified directory")
            
    except Exception as e:
        print(f"Error detecting datasets: {e}")
        print("Falling back to example datasets. You should place your actual datasets in the directory.")
        
        # Create some dummy example datasets for testing
        # In a real scenario, you would replace these with your actual datasets
        from sklearn.datasets import make_classification
        
        # Create directory if it doesn't exist
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Generate synthetic datasets with varying degrees of class overlap
        for i in range(3):
            # Generate dataset with different degrees of class separation
            X, y = make_classification(
                n_samples=1000, 
                n_features=10,
                n_informative=5,
                n_redundant=3,
                n_clusters_per_class=1,
                weights=[0.9, 0.1],  # Imbalanced
                class_sep=0.5 + i*0.5,  # Increasing separation
                random_state=42+i
            )
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(X.shape[1])])
            df['class'] = y
            
            # Save to file
            file_path = os.path.join(dataset_dir, f'example_dataset_{i+1}.csv')
            df.to_csv(file_path, index=False)
            
            # Add to dataset_paths
            dataset_paths[f'example_dataset_{i+1}'] = file_path
        
        print(f"Created {len(dataset_paths)} example datasets for testing")
    
    # Analyze datasets and compute overlap metrics
    metrics_df = analyze_datasets_overlap(dataset_paths)
    
    # Save metrics to CSV
    save_metrics(metrics_df)
    
    # Create visualizations
    visualize_overlap_metrics(metrics_df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()