#!/usr/bin/env python3
"""
Aggregate and analyze radius-independent ML results from multiple output folders

To run, had to copy interpretation_final.py and interpretation_final_protein.py to this folder
to avoid import errors, as the original files are not in the same directory.

Run with: python aggregate_radius_independent.py --parent_dir path/to/radius_independent_outputs
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add report_tests to path
from interpretation.interpretation_final_protein import ResultAnalyzer, OutputTee
from datetime import datetime
from scipy import stats

def aggregate_radius_independent(parent_dir: str):
    parent_dir = Path(parent_dir)
    output_dir = parent_dir / "aggregated_interpretation"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = output_dir / "aggregation_log.txt"
    original_stdout = sys.stdout
    log_stream = open(log_file, 'w')
    sys.stdout = OutputTee(original_stdout, log_stream)
    
    print(f"=== Aggregated Iron-Sulfur Redox Potential Prediction - Radius-Independent Results ===")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parent directory: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    all_results = {}
    for folder in parent_dir.glob("*"):
        if folder.is_dir() and (folder / "ml_results.json").exists():
            print(f"Processing: {folder.name}")
            analyzer = ResultAnalyzer(str(folder))
            all_results[folder.name] = analyzer.results
    
    # Aggregate model performance
    model_performance = []
    for dataset_name, results in all_results.items():
        radius = next(iter(results))  # Only one radius ("0")
        for model_name, metrics in results[radius].items():
            model_performance.append({
                'dataset': dataset_name,
                'model': model_name,
                'mae_mean': metrics.get('MAE_mean', float('inf')),
                'mae_std': metrics.get('MAE_std', 0),
                'rmse_mean': metrics.get('RMSE_mean', float('inf')),
                'rmse_std': metrics.get('RMSE_std', 0),
                'r2_mean': metrics.get('R2_mean', 0),
                'r2_std': metrics.get('R2_std', 0),
                'spearman_mean': metrics.get('Spearman_mean', 0),
                'spearman_std': metrics.get('Spearman_std', 0)
            })
    
    df = pd.DataFrame(model_performance)
    if df.empty:
        print("No data available for aggregation.")
        return
    
    # Compute overall statistics across all datasets for each metric
    metric_columns = ['mae_mean', 'rmse_mean', 'r2_mean', 'spearman_mean']
    overall_stats = {}
    
    for metric in metric_columns:
        overall_stats[metric] = df.groupby('model')[metric].agg(['mean', 'median', 'std']).round(3)
    
    all_models = set(df['model'])
    all_models_without_lr = all_models.copy()
    all_models_without_lr.discard('LinearRegression')  # Exclude LinearRegression if present
    top_models = sorted(list(all_models_without_lr))  # Use all non-LinearRegression models
    
    # Print comprehensive statistics
    print("\nOverall Performance Statistics Across All Radius-Independent Datasets:")
    print("="*80)
    print("MAE STATISTICS:")
    print("-" * 15)
    for model_name in top_models:
        if model_name in overall_stats['mae_mean'].index:
            stats = overall_stats['mae_mean'].loc[model_name]
            print(f"{model_name}: Mean MAE = {stats['mean']:.2f} mV, Median MAE = {stats['median']:.2f} mV, Std MAE = {stats['std']:.2f} mV")
    if 'LinearRegression' in overall_stats['mae_mean'].index:
        lr_stats = overall_stats['mae_mean'].loc['LinearRegression']
        print(f"LinearRegression: Mean MAE = {lr_stats['mean']:.2f} mV, Median MAE = {lr_stats['median']:.2f} mV, Std MAE = {lr_stats['std']:.2f} mV")
    
    print("\nRMSE STATISTICS:")
    print("-" * 16)
    for model_name in top_models:
        if model_name in overall_stats['rmse_mean'].index:
            stats = overall_stats['rmse_mean'].loc[model_name]
            print(f"{model_name}: Mean RMSE = {stats['mean']:.2f} mV, Median RMSE = {stats['median']:.2f} mV, Std RMSE = {stats['std']:.2f} mV")
    if 'LinearRegression' in overall_stats['rmse_mean'].index:
        lr_stats = overall_stats['rmse_mean'].loc['LinearRegression']
        print(f"LinearRegression: Mean RMSE = {lr_stats['mean']:.2f} mV, Median RMSE = {lr_stats['median']:.2f} mV, Std RMSE = {lr_stats['std']:.2f} mV")
    
    print("\nR² STATISTICS:")
    print("-" * 14)
    for model_name in top_models:
        if model_name in overall_stats['r2_mean'].index:
            stats = overall_stats['r2_mean'].loc[model_name]
            print(f"{model_name}: Mean R² = {stats['mean']:.3f}, Median R² = {stats['median']:.3f}, Std R² = {stats['std']:.3f}")
    if 'LinearRegression' in overall_stats['r2_mean'].index:
        lr_stats = overall_stats['r2_mean'].loc['LinearRegression']
        print(f"LinearRegression: Mean R² = {lr_stats['mean']:.3f}, Median R² = {lr_stats['median']:.3f}, Std R² = {lr_stats['std']:.3f}")
    
    print("\nSPEARMAN CORRELATION STATISTICS:")
    print("-" * 33)
    for model_name in top_models:
        if model_name in overall_stats['spearman_mean'].index:
            stats = overall_stats['spearman_mean'].loc[model_name]
            print(f"{model_name}: Mean Spearman = {stats['mean']:.3f}, Median Spearman = {stats['median']:.3f}, Std Spearman = {stats['std']:.3f}")
    if 'LinearRegression' in overall_stats['spearman_mean'].index:
        lr_stats = overall_stats['spearman_mean'].loc['LinearRegression']
        print(f"LinearRegression: Mean Spearman = {lr_stats['mean']:.3f}, Median Spearman = {lr_stats['median']:.3f}, Std Spearman = {lr_stats['std']:.3f}")
    
    # Best performers for each metric
    best_mae_model = overall_stats['mae_mean']['mean'].idxmin()
    best_mae_value = overall_stats['mae_mean']['mean'].min()
    best_rmse_model = overall_stats['rmse_mean']['mean'].idxmin()
    best_rmse_value = overall_stats['rmse_mean']['mean'].min()
    best_r2_model = overall_stats['r2_mean']['mean'].idxmax()
    best_r2_value = overall_stats['r2_mean']['mean'].max()
    best_spearman_model = overall_stats['spearman_mean']['mean'].idxmax()
    best_spearman_value = overall_stats['spearman_mean']['mean'].max()
    
    print(f"\nBest Performers:")
    print(f"MAE: {best_mae_model} with Mean MAE = {best_mae_value:.2f} mV")
    print(f"RMSE: {best_rmse_model} with Mean RMSE = {best_rmse_value:.2f} mV")
    print(f"R²: {best_r2_model} with Mean R² = {best_r2_value:.3f}")
    print(f"Spearman: {best_spearman_model} with Mean Spearman = {best_spearman_value:.3f}")
    
    # Top 5 models by MAE
    top_models_df = df.nsmallest(5, 'mae_mean')
    
    print("\nTop 5 Models by MAE Across All Radius-Independent Datasets:")
    print("="*60)
    for idx, row in top_models_df.iterrows():
        print(f"{row['model']} ({row['dataset']}): "
              f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
              f"RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f} mV, "
              f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}, "
              f"Spearman={row['spearman_mean']:.3f}±{row['spearman_std']:.3f}")
    
    # Save report
    report_file = output_dir / 'aggregated_report.txt'
    with open(report_file, 'w') as f:
        f.write("Aggregated Iron-Sulfur Cofactor Redox Potential Prediction - Radius-Independent Report\n")
        f.write("="*70 + "\n\n")
        
        # Write all statistics
        f.write("OVERALL PERFORMANCE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write("MAE STATISTICS:\n")
        for model_name in top_models:
            if model_name in overall_stats['mae_mean'].index:
                stats = overall_stats['mae_mean'].loc[model_name]
                f.write(f"{model_name}: Mean MAE = {stats['mean']:.2f} mV, Median MAE = {stats['median']:.2f} mV, Std MAE = {stats['std']:.2f} mV\n")
        if 'LinearRegression' in overall_stats['mae_mean'].index:
            lr_stats = overall_stats['mae_mean'].loc['LinearRegression']
            f.write(f"LinearRegression: Mean MAE = {lr_stats['mean']:.2f} mV, Median MAE = {lr_stats['median']:.2f} mV, Std MAE = {lr_stats['std']:.2f} mV\n")
        
        f.write("\nRMSE STATISTICS:\n")
        for model_name in top_models:
            if model_name in overall_stats['rmse_mean'].index:
                stats = overall_stats['rmse_mean'].loc[model_name]
                f.write(f"{model_name}: Mean RMSE = {stats['mean']:.2f} mV, Median RMSE = {stats['median']:.2f} mV, Std RMSE = {stats['std']:.2f} mV\n")
        if 'LinearRegression' in overall_stats['rmse_mean'].index:
            lr_stats = overall_stats['rmse_mean'].loc['LinearRegression']
            f.write(f"LinearRegression: Mean RMSE = {lr_stats['mean']:.2f} mV, Median RMSE = {lr_stats['median']:.2f} mV, Std RMSE = {lr_stats['std']:.2f} mV\n")
        
        f.write("\nR² STATISTICS:\n")
        for model_name in top_models:
            if model_name in overall_stats['r2_mean'].index:
                stats = overall_stats['r2_mean'].loc[model_name]
                f.write(f"{model_name}: Mean R² = {stats['mean']:.3f}, Median R² = {stats['median']:.3f}, Std R² = {stats['std']:.3f}\n")
        if 'LinearRegression' in overall_stats['r2_mean'].index:
            lr_stats = overall_stats['r2_mean'].loc['LinearRegression']
            f.write(f"LinearRegression: Mean R² = {lr_stats['mean']:.3f}, Median R² = {lr_stats['median']:.3f}, Std R² = {lr_stats['std']:.3f}\n")
        
        f.write("\nSPEARMAN CORRELATION STATISTICS:\n")
        for model_name in top_models:
            if model_name in overall_stats['spearman_mean'].index:
                stats = overall_stats['spearman_mean'].loc[model_name]
                f.write(f"{model_name}: Mean Spearman = {stats['mean']:.3f}, Median Spearman = {stats['median']:.3f}, Std Spearman = {stats['std']:.3f}\n")
        if 'LinearRegression' in overall_stats['spearman_mean'].index:
            lr_stats = overall_stats['spearman_mean'].loc['LinearRegression']
            f.write(f"LinearRegression: Mean Spearman = {lr_stats['mean']:.3f}, Median Spearman = {lr_stats['median']:.3f}, Std Spearman = {lr_stats['std']:.3f}\n")
        
        f.write(f"\nBest Performers:\n")
        f.write(f"MAE: {best_mae_model} with Mean MAE = {best_mae_value:.2f} mV\n")
        f.write(f"RMSE: {best_rmse_model} with Mean RMSE = {best_rmse_value:.2f} mV\n")
        f.write(f"R²: {best_r2_model} with Mean R² = {best_r2_value:.3f}\n")
        f.write(f"Spearman: {best_spearman_model} with Mean Spearman = {best_spearman_value:.3f}\n")
        
        f.write("\nTOP 5 MODELS BY MAE:\n")
        f.write("-" * 30 + "\n")
        for idx, row in top_models_df.iterrows():
            f.write(f"{row['model']} ({row['dataset']}): "
                    f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                    f"RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f} mV, "
                    f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}, "
                    f"Spearman={row['spearman_mean']:.3f}±{row['spearman_std']:.3f}\n")
    print(f"Aggregated report saved to: {report_file}")
    
    sys.stdout = original_stdout
    log_stream.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate radius-independent ML results')
    parser.add_argument('--parent_dir', type=str, required=True, help='Parent directory containing ML output folders')
    args = parser.parse_args()
    aggregate_radius_independent(args.parent_dir)