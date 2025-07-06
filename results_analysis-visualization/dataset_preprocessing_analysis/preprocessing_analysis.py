#!/usr/bin/env python3
"""
Feature Analysis Script for Iron-Sulfur Cofactor Datasets

This script analyzes the impact of preprocessing on feature counts across all datasets.
It reports average feature counts after preprocessing and feature stability.

Usage: python preprocessing_analysis.py \
    --data_dir /path/to/feature_subsets \
    --output_dir /path/to/output
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
import warnings
from tqdm import tqdm
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def preprocess_data(X, n_iterations=10, correlation_threshold=0.95):
    """
    Apply preprocessing steps to different subsets of data to simulate CV folds
    """
    processed_dfs = []
    kept_features = []
    
    for i in range(n_iterations):
        # Take only 80% of data like in real CV
        X_subset = X.sample(frac=0.8, random_state=42+i)
        
        # Remove features with zero variance in this subset
        variance_selector = VarianceThreshold(threshold=0)
        X_var = pd.DataFrame(
            variance_selector.fit_transform(X_subset),
            columns=X_subset.columns[variance_selector.get_support()],
            index=X_subset.index
        )
        
        # Remove highly correlated features
        correlation_matrix = X_var.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > correlation_threshold)]
        
        X_processed = X_var.drop(columns=high_corr_features)
        
        # Handle missing values
        if X_processed.isnull().sum().sum() > 0:
            X_processed = X_processed.fillna(X_processed.median())
        
        processed_dfs.append(X_processed)
        kept_features.append(set(X_processed.columns))
    
    return processed_dfs, kept_features

def analyze_feature_stability(kept_features_sets, original_features):
    """
    Analyze the stability of features across preprocessing iterations
    Returns: Dict with stability statistics
    """
    # Count how many times each feature appears
    feature_counter = Counter()
    for feature_set in kept_features_sets:
        for feature in feature_set:
            feature_counter[feature] += 1
    
    n_iterations = len(kept_features_sets)
    
    # Group features by stability
    highly_stable = [f for f, count in feature_counter.items() if count/n_iterations >= 0.9]
    stable = [f for f, count in feature_counter.items() if count/n_iterations >= 0.7 and count/n_iterations < 0.9]
    moderate = [f for f, count in feature_counter.items() if count/n_iterations >= 0.5 and count/n_iterations < 0.7]
    unstable = [f for f, count in feature_counter.items() if count/n_iterations < 0.5]
    
    return {
        "highly_stable_count": len(highly_stable),
        "stable_count": len(stable),
        "moderate_count": len(moderate),
        "unstable_count": len(unstable),
        "total_features": len(feature_counter),
        "original_features": original_features,
        "highly_stable_pct": round(len(highly_stable) / original_features * 100, 1),
        "stable_pct": round(len(stable) / original_features * 100, 1),
        "moderate_pct": round(len(moderate) / original_features * 100, 1),
        "unstable_pct": round(len(unstable) / original_features * 100, 1),
        
        # Store some example feature names (limit to 10 each)
        "highly_stable_examples": highly_stable[:10],
        "stable_examples": stable[:10],
        "moderate_examples": moderate[:10],
        "unstable_examples": unstable[:10]
    }

def analyze_dataset(filepath, target_column="redox_potential", iterations=10):
    """
    Analyze a single dataset file
    Returns: Dict with analysis results
    """
    try:
        # Load data
        df = pd.read_csv(filepath)
        
        # Skip if empty
        if df.shape[0] < 5:  # Require at least 5 samples
            return {
                "status": "skipped",
                "reason": "Insufficient samples",
                "sample_count": df.shape[0]
            }
        
        # Check if target column exists
        if target_column not in df.columns:
            # Let's infer the target column or skip if we can't find one
            potential_targets = [col for col in df.columns if 'redox' in col.lower() or 'potential' in col.lower()]
            if potential_targets:
                target_column = potential_targets[0]
            else:
                return {
                    "status": "skipped",
                    "reason": "Target column not found",
                    "columns": list(df.columns)[:10]  # First 10 columns for reference
                }
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove any non-numeric columns
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            X = X.drop(columns=non_numeric_cols)
        
        original_features = X.shape[1]
        original_samples = X.shape[0]
        
        # Apply preprocessing multiple times
        processed_dfs, kept_features = preprocess_data(X, n_iterations=iterations)
        
        # Calculate statistics
        feature_counts = [df.shape[1] for df in processed_dfs]
        avg_features = np.mean(feature_counts)
        min_features = np.min(feature_counts)
        max_features = np.max(feature_counts)
        std_features = np.std(feature_counts)
        
        # Analyze feature stability
        stability_stats = analyze_feature_stability(kept_features, original_features)
        
        return {
            "status": "success",
            "original_features": original_features,
            "original_samples": original_samples,
            "avg_features": float(avg_features),
            "min_features": int(min_features),
            "max_features": int(max_features),
            "std_features": float(std_features),
            "percent_retained_avg": round((avg_features / original_features) * 100, 1),
            "stability": stability_stats
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

def find_all_datasets(data_dir):
    """Find all feature CSV files in the data directory structure"""
    data_path = Path(data_dir)
    all_files = []
    
    # Process radius-based files in subfolders
    for subfolder in data_path.iterdir():
        if subfolder.is_dir():
            # Check for radius-based files (r1, r2, etc.)
            radius_files = list(subfolder.glob("features_r*_*.csv"))
            for file in radius_files:
                # Extract radius from filename
                filename = file.name
                radius = int(''.join(filter(str.isdigit, filename.split("_")[1])))
                
                all_files.append({
                    "filepath": str(file),
                    "dataset_type": subfolder.name,
                    "radius": radius,
                    "is_protein_only": False
                })
            
            # Check for protein-only files
            protein_files = list(subfolder.glob("features_*_protein.csv"))
            for file in protein_files:
                all_files.append({
                    "filepath": str(file),
                    "dataset_type": subfolder.name,
                    "radius": None,
                    "is_protein_only": True
                })
    
    # Sort files by dataset_type and radius
    all_files.sort(key=lambda x: (x["dataset_type"], x["radius"] if x["radius"] is not None else 0))
    return all_files

def generate_report(results, output_dir):
    """Generate report files from analysis results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Convert results to DataFrame for easy manipulation
    results_df = pd.DataFrame(results)
    
    # Save full results as JSON
    with open(output_path / "feature_analysis_full.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for result in results:
        if result["status"] == "success":
            summary_data.append({
                "Dataset": f"{result['dataset_type']}" + 
                          (f" (r={result['radius']}Å)" if result['radius'] is not None else " (protein)"),
                "Original Features": result["original_features"],
                "After Preprocessing (avg)": f"{result['avg_features']:.1f} ± {result['std_features']:.1f}",
                "Retained (%)": f"{result['percent_retained_avg']}%",
                "Highly Stable Features (%)": f"{result['stability']['highly_stable_pct']}%",
                "Stable Features (%)": f"{result['stability']['stable_pct']}%"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "feature_analysis_summary.csv", index=False)
    
    # Create plots
    create_visualizations(results, output_path)
    
    return summary_df

def create_visualizations(results, output_path):
    """Create visualization plots from analysis results"""
    # Filter successful results
    successful_results = [r for r in results if r["status"] == "success"]
    
    # Group by dataset type
    dataset_types = set([r["dataset_type"] for r in successful_results])
    
    # Plot 1: Feature retention by radius (percentage) - original plot
    plt.figure(figsize=(14, 8))
    
    for dataset_type in dataset_types:
        # Get radius-based results for this dataset type
        dataset_results = [r for r in successful_results 
                          if r["dataset_type"] == dataset_type and r["radius"] is not None]
        
        if dataset_results:
            radii = [r["radius"] for r in dataset_results]
            retention = [r["percent_retained_avg"] for r in dataset_results]
            
            plt.plot(radii, retention, 'o-', label=dataset_type, linewidth=2)
    
    plt.xlabel('Radius (Å)', fontsize=12)
    plt.ylabel('Features Retained After Preprocessing (%)', fontsize=12)
    plt.title('Feature Retention by Radius (Percentage)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / "feature_retention_by_radius_percent.png", dpi=300)
    
    # Plot 2: NEW - Feature count by radius (absolute numbers)
    plt.figure(figsize=(14, 8))
    
    for dataset_type in dataset_types:
        # Get radius-based results for this dataset type
        dataset_results = [r for r in successful_results 
                          if r["dataset_type"] == dataset_type and r["radius"] is not None]
        
        if dataset_results:
            radii = [r["radius"] for r in dataset_results]
            feature_counts = [r["avg_features"] for r in dataset_results]
            
            plt.plot(radii, feature_counts, 'o-', label=dataset_type, linewidth=2)
    
    plt.xlabel('Radius (Å)', fontsize=12)
    plt.ylabel('Average Feature Count After Preprocessing', fontsize=12)
    plt.title('Feature Count by Radius (Absolute)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / "feature_count_by_radius.png", dpi=300)
    
    # Also add a separate plot for protein-only datasets (these don't have radii)
    protein_only_results = [r for r in successful_results if r["is_protein_only"]]
    if protein_only_results:
        plt.figure(figsize=(10, 6))
        dataset_types = [r["dataset_type"] for r in protein_only_results]
        feature_counts = [r["avg_features"] for r in protein_only_results]
        
        # Create bar chart for protein-only datasets
        plt.bar(dataset_types, feature_counts)
        plt.xlabel('Dataset Type', fontsize=12)
        plt.ylabel('Average Feature Count', fontsize=12)
        plt.title('Feature Count for Protein-Only Datasets', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count values on top of bars
        for i, v in enumerate(feature_counts):
            plt.text(i, v + 5, f"{v:.1f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(output_path / "protein_only_feature_count.png", dpi=300)
    
    # Plot feature stability (original code)
    stability_data = []
    for r in successful_results:
        dataset_label = f"{r['dataset_type']}" + (f" (r={r['radius']}Å)" if r['radius'] is not None else " (protein)")
        stability_data.append({
            'Dataset': dataset_label,
            'Highly_Stable': r['stability']['highly_stable_pct'],
            'Stable': r['stability']['stable_pct'],
            'Moderate': r['stability']['moderate_pct'],
            'Unstable': r['stability']['unstable_pct']
        })
    
    # Check if we have data to visualize
    if not stability_data:
        print("Warning: No stability data available for visualization")
        return
        
    stability_df = pd.DataFrame(stability_data)
    
    # Select top 15 datasets for readability if there are too many
    if len(stability_df) > 15:
        # For radius datasets, select a representative sample across radii
        radius_dfs = stability_df[stability_df['Dataset'].str.contains('r=')]
        protein_dfs = stability_df[stability_df['Dataset'].str.contains('protein')]
        
        # Group by dataset type and select one small, one medium, one large radius
        selected_rows = []
        for dataset_type in dataset_types:
            type_data = radius_dfs[radius_dfs['Dataset'].str.contains(dataset_type)]
            if len(type_data) > 0:
                indices = [type_data.index[0]]  # smallest radius
                if len(type_data) > 2:
                    indices.append(type_data.index[len(type_data) // 2])  # middle radius
                indices.append(type_data.index[-1])  # largest radius
                selected_rows.extend(indices)
        
        # Add all protein datasets
        selected_rows.extend(protein_dfs.index)
        
        # Create plot data
        plot_stability_df = stability_df.loc[selected_rows]
    else:
        plot_stability_df = stability_df
    
    # Melt the dataframe for seaborn - use the actual column names
    melted_df = pd.melt(plot_stability_df, id_vars=['Dataset'], 
                      value_vars=['Highly_Stable', 'Stable', 'Moderate', 'Unstable'],
                      var_name='Stability', value_name='Percentage')
    
    # Plot
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Dataset', y='Percentage', hue='Stability', data=melted_df)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Percentage of Features', fontsize=12)
    plt.title('Feature Stability Across Datasets', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Stability Category')
    plt.tight_layout()
    plt.savefig(output_path / "feature_stability.png", dpi=300)

def main():
    """Main function to run the feature analysis"""
    parser = argparse.ArgumentParser(description='Analyze feature retention and stability across datasets')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing feature subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--target_column', type=str, default='Em',
                       help='Name of target column (default: Em)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of preprocessing iterations to simulate CV folds (default: 10)')
    
    args = parser.parse_args()
    
    print(f"Finding datasets in {args.data_dir}...")
    datasets = find_all_datasets(args.data_dir)
    print(f"Found {len(datasets)} datasets")
    
    results = []
    
    print("Analyzing datasets...")
    for dataset in tqdm(datasets):
        result = analyze_dataset(
            dataset["filepath"], 
            target_column=args.target_column,
            iterations=args.iterations
        )
        # Add dataset info to result
        result.update({
            "dataset_type": dataset["dataset_type"],
            "radius": dataset["radius"],
            "is_protein_only": dataset["is_protein_only"],
            "filepath": dataset["filepath"]
        })
        results.append(result)
    
    print("Generating report...")
    summary_df = generate_report(results, args.output_dir)
    
    print("\nAnalysis Summary:")
    print(summary_df.to_string())
    print(f"\nDetailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()