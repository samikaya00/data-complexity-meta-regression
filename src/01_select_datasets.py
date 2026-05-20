"""
Dataset Selection Script for Extended Overlap Study

Combines:
1. ALL datasets from noisy and borderline examples (one per folder)
2. ~30 new datasets from KEEL imbalanced (IR > 9) Parts 1, 2, 3

Selection strategy for new datasets:
- Prioritize diversity (different base datasets, IR ranges, sizes)
- If not enough unique bases, allow variants of same base dataset
- Target harder classification scenarios
- VALIDATE: Only include datasets with exactly 2 classes (binary classification)
"""

import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict
import re

class DatasetSelector:
    
    def __init__(self, 
                 noisy_borderline_path,
                 imbalanced_part1_path,
                 imbalanced_part2_path,
                 imbalanced_part3_path,
                 output_path='extended_datasets'):
        
        self.noisy_borderline_path = noisy_borderline_path
        self.imbalanced_paths = [
            imbalanced_part1_path,
            imbalanced_part2_path,
            imbalanced_part3_path
        ]
        self.output_path = output_path
        
        self.noisy_borderline_datasets = []
        self.candidate_datasets = []
        self.selected_new = []
        self.skipped_datasets = []  # Track skipped datasets
    
    def is_macos_artifact(self, path):
        """Check if path is a macOS artifact"""
        return '__MACOSX' in path or '/._' in path or path.startswith('._')
    
    def extract_all_zips(self, folder_path):
        """Recursively find and extract all .zip files"""
        if not os.path.exists(folder_path):
            return
            
        zip_count = 0
        for root, dirs, files in os.walk(folder_path):
            # Skip macOS artifact directories
            if self.is_macos_artifact(root):
                continue
                
            for f in files:
                if f.endswith('.zip') and not f.startswith('._'):
                    zip_path = os.path.join(root, f)
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(root)
                        zip_count += 1
                    except Exception as e:
                        print(f"Error extracting {zip_path}: {e}")
        
        if zip_count > 0:
            print(f"  Extracted {zip_count} zip files in {folder_path}")
    
    def find_dat_files(self, folder_path):
        """Recursively find all .dat files in folder and subfolders"""
        dat_files = []
        for root, dirs, files in os.walk(folder_path):
            # Skip macOS artifact directories
            if self.is_macos_artifact(root):
                continue
                
            for f in files:
                # Skip macOS artifact files
                if f.endswith('.dat') and not f.startswith('._'):
                    dat_files.append(os.path.join(root, f))
        return dat_files
        
    def parse_keel_file(self, filepath):
        """Parse KEEL .dat file to get basic info - validates binary classification"""
        # Skip macOS artifacts
        if self.is_macos_artifact(filepath):
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
            
            # VALIDATION: Must have exactly 2 classes for binary classification
            n_classes = len(class_counts)
            if n_classes != 2:
                return None  # Skip non-binary datasets
            
            counts = list(class_counts.values())
            ir = max(counts) / min(counts)
            
            # VALIDATION: Both classes must have at least 5 samples
            if min(counts) < 5:
                return None
                
            return {
                'n_features': n_features,
                'n_samples': n_samples,
                'n_classes': n_classes,
                'imbalance_ratio': ir,
                'minority_count': min(counts),
                'majority_count': max(counts),
                'class_labels': list(class_counts.keys())
            }
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def get_base_name(self, dataset_name):
        """Extract base dataset name (remove fold numbers, variants)"""
        base = re.sub(r'[-_]\d+[-_]\d+.*$', '', dataset_name)
        base = re.sub(r'[-_]sub\d+.*$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'[-_]\d+$', '', base)
        base = re.sub(r'[-_](tra|tst)$', '', base, flags=re.IGNORECASE)
        return base.lower()
    
    def pick_best_dat_file(self, dat_files):
        """From a list of .dat files, pick the best one (prefer non-tra/tst, then -tra)"""
        if not dat_files:
            return None
        
        # Filter out macOS artifacts
        dat_files = [f for f in dat_files if not self.is_macos_artifact(f)]
        
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
        
        # Last resort: any file
        return dat_files[0]
    
    def scan_noisy_borderline_datasets(self):
        """Scan noisy and borderline datasets - one dataset per subfolder"""
        print("Scanning noisy and borderline datasets...")
        
        if not os.path.exists(self.noisy_borderline_path):
            print(f"WARNING: Path not found: {self.noisy_borderline_path}")
            return
        
        # First extract all zips
        print("  Extracting zip files...")
        self.extract_all_zips(self.noisy_borderline_path)
        
        # Group .dat files by their immediate parent folder
        folder_files = defaultdict(list)
        
        for root, dirs, files in os.walk(self.noisy_borderline_path):
            # Skip macOS artifacts
            if self.is_macos_artifact(root):
                continue
                
            for f in files:
                if f.endswith('.dat') and not f.startswith('._'):
                    filepath = os.path.join(root, f)
                    folder_name = os.path.basename(root)
                    if folder_name != os.path.basename(self.noisy_borderline_path):
                        folder_files[folder_name].append(filepath)
        
        print(f"  Found {len(folder_files)} dataset folders")
        
        # Pick one .dat file per folder
        valid_count = 0
        skipped_count = 0
        
        for folder_name, dat_files in folder_files.items():
            # Skip macOS artifact folders
            if self.is_macos_artifact(folder_name):
                continue
                
            best_file = self.pick_best_dat_file(dat_files)
            if best_file:
                info = self.parse_keel_file(best_file)
                if info:  # Only add if validation passed
                    self.noisy_borderline_datasets.append({
                        'name': folder_name,
                        'filepath': best_file,
                        'source': 'noisy_borderline',
                        'base_name': self.get_base_name(folder_name),
                        **info
                    })
                    valid_count += 1
                else:
                    self.skipped_datasets.append(folder_name)
                    skipped_count += 1
        
        print(f"Found {valid_count} valid noisy/borderline datasets (skipped {skipped_count} non-binary)")
    
    def scan_imbalanced_datasets(self):
        """Scan all imbalanced dataset folders - one dataset per subfolder"""
        print("\nScanning imbalanced datasets (IR > 9)...")
        
        for i, path in enumerate(self.imbalanced_paths):
            if not os.path.exists(path):
                print(f"WARNING: Path not found: {path}")
                continue
            
            part_name = f"Part{i+1}"
            
            # First extract all zips
            print(f"  Extracting zip files in {part_name}...")
            self.extract_all_zips(path)
            
            # Group .dat files by their immediate parent folder
            folder_files = defaultdict(list)
            
            for root, dirs, files in os.walk(path):
                # Skip macOS artifacts
                if self.is_macos_artifact(root):
                    continue
                    
                for f in files:
                    if f.endswith('.dat') and not f.startswith('._'):
                        filepath = os.path.join(root, f)
                        folder_name = os.path.basename(root)
                        if folder_name != os.path.basename(path):
                            folder_files[folder_name].append(filepath)
            
            # Pick one .dat file per folder
            part_count = 0
            skipped_count = 0
            
            for folder_name, dat_files in folder_files.items():
                # Skip macOS artifact folders
                if self.is_macos_artifact(folder_name):
                    continue
                    
                best_file = self.pick_best_dat_file(dat_files)
                if best_file:
                    info = self.parse_keel_file(best_file)
                    if info:  # Only add if validation passed
                        self.candidate_datasets.append({
                            'name': folder_name,
                            'filepath': best_file,
                            'source': part_name,
                            'base_name': self.get_base_name(folder_name),
                            **info
                        })
                        part_count += 1
                    else:
                        self.skipped_datasets.append(folder_name)
                        skipped_count += 1
            
            print(f"  {part_name}: {part_count} valid datasets (skipped {skipped_count} non-binary)")
        
        print(f"Total valid candidates: {len(self.candidate_datasets)}")
    
    def select_new_datasets(self, n_select=30):
        """Select diverse new datasets - relaxed to allow variants if needed"""
        print(f"\nSelecting {n_select} new datasets...")
        
        if not self.candidate_datasets:
            print("No candidate datasets found!")
            return []
        
        # Get base names from noisy/borderline to avoid overlap
        noisy_borderline_bases = set(d['base_name'] for d in self.noisy_borderline_datasets)
        
        candidates = [d for d in self.candidate_datasets 
                      if d['base_name'] not in noisy_borderline_bases]
        
        print(f"Candidates after removing overlaps: {len(candidates)}")
        
        if not candidates:
            print("No valid candidates remaining!")
            return []
        
        df = pd.DataFrame(candidates)
        
        selected = []
        selected_names = set()
        selected_bases = set()
        
        # PHASE 1: Select diverse base names across IR buckets
        ir_buckets = [
            (9, 15, 'low_ir'),
            (15, 25, 'medium_ir'),
            (25, 50, 'high_ir'),
            (50, np.inf, 'extreme_ir')
        ]
        
        per_bucket = n_select // len(ir_buckets) + 1
        
        for ir_min, ir_max, bucket_name in ir_buckets:
            bucket_df = df[(df['imbalance_ratio'] >= ir_min) & 
                           (df['imbalance_ratio'] < ir_max)]
            
            bucket_df = bucket_df[~bucket_df['base_name'].isin(selected_bases)]
            
            if bucket_df.empty:
                continue
            
            bucket_df = bucket_df.sort_values('minority_count', ascending=False)
            
            bucket_selected = 0
            for _, row in bucket_df.iterrows():
                if row['base_name'] not in selected_bases:
                    selected.append(row.to_dict())
                    selected_names.add(row['name'])
                    selected_bases.add(row['base_name'])
                    bucket_selected += 1
                    
                    if bucket_selected >= per_bucket:
                        break
            
            print(f"  {bucket_name} (IR {ir_min}-{ir_max}): selected {bucket_selected}")
        
        print(f"  After phase 1 (unique bases): {len(selected)}")
        
        # PHASE 2: If we need more, allow variants (same base, different name)
        if len(selected) < n_select:
            remaining = df[~df['name'].isin(selected_names)]
            remaining = remaining.sort_values('imbalance_ratio', ascending=False)
            
            print(f"  Phase 2: Adding variants to reach {n_select}...")
            
            for _, row in remaining.iterrows():
                if len(selected) >= n_select:
                    break
                if row['name'] not in selected_names:
                    selected.append(row.to_dict())
                    selected_names.add(row['name'])
            
            print(f"  After phase 2 (with variants): {len(selected)}")
        
        self.selected_new = selected[:n_select]
        print(f"\nTotal new datasets selected: {len(self.selected_new)}")
        
        return self.selected_new
    
    def copy_to_output(self):
        """Copy all selected datasets to output folder"""
        print(f"\nCopying datasets to {self.output_path}...")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        for d in self.noisy_borderline_datasets:
            src = d['filepath']
            dst = os.path.join(self.output_path, d['name'] + '.dat')
            shutil.copy2(src, dst)
        
        for d in self.selected_new:
            src = d['filepath']
            dst = os.path.join(self.output_path, d['name'] + '.dat')
            shutil.copy2(src, dst)
        
        total = len(self.noisy_borderline_datasets) + len(self.selected_new)
        print(f"Copied {total} datasets to {self.output_path}")
    
    def generate_report(self):
        """Generate summary report"""
        report_path = os.path.join(self.output_path, 'dataset_selection_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATASET SELECTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"NOISY & BORDERLINE DATASETS ({len(self.noisy_borderline_datasets)})\n")
            f.write("-"*50 + "\n")
            for d in sorted(self.noisy_borderline_datasets, key=lambda x: x['name']):
                f.write(f"{d['name']}: IR={d['imbalance_ratio']:.1f}, "
                        f"n={d['n_samples']}, features={d['n_features']}\n")
            
            f.write(f"\nNEW IMBALANCED DATASETS ({len(self.selected_new)})\n")
            f.write("-"*50 + "\n")
            for d in sorted(self.selected_new, key=lambda x: x['imbalance_ratio']):
                f.write(f"{d['name']}: IR={d['imbalance_ratio']:.1f}, "
                        f"n={d['n_samples']}, features={d['n_features']}, "
                        f"minority={d['minority_count']}\n")
            
            f.write(f"\nSKIPPED DATASETS (non-binary or invalid)\n")
            f.write("-"*50 + "\n")
            for name in sorted(self.skipped_datasets):
                f.write(f"  {name}\n")
            
            f.write(f"\nSUMMARY STATISTICS\n")
            f.write("-"*50 + "\n")
            
            all_datasets = self.noisy_borderline_datasets + self.selected_new
            irs = [d['imbalance_ratio'] for d in all_datasets if d['imbalance_ratio'] < np.inf]
            sizes = [d['n_samples'] for d in all_datasets]
            
            f.write(f"Total datasets: {len(all_datasets)}\n")
            f.write(f"Skipped (non-binary): {len(self.skipped_datasets)}\n")
            f.write(f"IR range: {min(irs):.1f} - {max(irs):.1f}\n")
            f.write(f"IR median: {np.median(irs):.1f}\n")
            f.write(f"Sample size range: {min(sizes)} - {max(sizes)}\n")
            
            f.write(f"\nIR Distribution:\n")
            f.write(f"  IR 1-9: {sum(1 for ir in irs if ir < 9)}\n")
            f.write(f"  IR 9-15: {sum(1 for ir in irs if 9 <= ir < 15)}\n")
            f.write(f"  IR 15-25: {sum(1 for ir in irs if 15 <= ir < 25)}\n")
            f.write(f"  IR 25-50: {sum(1 for ir in irs if 25 <= ir < 50)}\n")
            f.write(f"  IR 50+: {sum(1 for ir in irs if ir >= 50)}\n")
        
        print(f"Report saved to {report_path}")
        
        csv_path = os.path.join(self.output_path, 'selected_datasets.csv')
        all_data = []
        for d in self.noisy_borderline_datasets:
            all_data.append({**d, 'group': 'noisy_borderline'})
        for d in self.selected_new:
            all_data.append({**d, 'group': 'imbalanced_ir9'})
        
        pd.DataFrame(all_data).to_csv(csv_path, index=False)
        print(f"Dataset list saved to {csv_path}")
    
    def run(self, n_new=30):
        """Run full selection pipeline"""
        self.scan_noisy_borderline_datasets()
        self.scan_imbalanced_datasets()
        self.select_new_datasets(n_select=n_new)
        self.copy_to_output()
        self.generate_report()
        
        print("\n" + "="*50)
        print("SELECTION COMPLETE")
        print("="*50)
        print(f"Noisy & borderline datasets: {len(self.noisy_borderline_datasets)}")
        print(f"New imbalanced datasets: {len(self.selected_new)}")
        print(f"Total: {len(self.noisy_borderline_datasets) + len(self.selected_new)}")
        print(f"Skipped (non-binary): {len(self.skipped_datasets)}")
        print(f"Output folder: {self.output_path}")


if __name__ == "__main__":
    
    BASE_PATH = "/Users/samikaya/Documents/VSCode/Benchmark"
    
    selector = DatasetSelector(
        noisy_borderline_path=f"{BASE_PATH}/noisy_borderline",
        imbalanced_part1_path=f"{BASE_PATH}/imbalanced_ir9_part1",
        imbalanced_part2_path=f"{BASE_PATH}/imbalanced_ir9_part2",
        imbalanced_part3_path=f"{BASE_PATH}/imbalanced_ir9_part3",
        output_path=f"{BASE_PATH}/extended_datasets"
    )
    
    selector.run(n_new=30)