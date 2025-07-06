#!/usr/bin/env python3
"""
Cross-Dataset Comparison Script for Protein ML Results

This script creates comparative heatmaps showing model performance across
different protein datasets (All cofactors, FES proteins, SF4 proteins).

Usage: 
python compare_datasets_heatmaps.py \
  --results_dir /path/to/results_folder \
  --output_dir /path/to/comparison_results \
  --exclude_from_range LinearRegression \
  --use_fixed_ranges
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_performance_heatmap(results_dir, output_dir, metric='MAE_mean', exclude_models=None,
                              vmin_mae=None, vmax_mae=None, vmin_rmse=None, vmax_rmse=None, 
                              vmin_r2=None, vmax_r2=None):
    """Create heatmap comparing model performances across datasets from CSV files in a single directory
    
    Parameters:
    -----------
    results_dir : str
        Directory containing result CSV files for different datasets
    output_dir : str
        Directory to save output visualizations
    metric : str
        Metric to use for heatmap (default: 'MAE_mean')
    exclude_models : list
        List of model names to exclude from heatmap
    vmin_mae, vmax_mae : float
        Min and max values for MAE color scale
    vmin_rmse, vmax_rmse : float
        Min and max values for RMSE color scale
    vmin_r2, vmax_r2 : float
        Min and max values for R2 color scale
    """
    
    # Set default for exclude_models
    if exclude_models is None:
        exclude_models = []
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Dataset name mapping
    dataset_display_names = {
        'all_cofactors': 'All cofactors',
        'FES': 'Only FES proteins',
        'SF4': 'Only SF4 proteins'
    }
    
    # Dictionary to store results
    all_results = {}
    
    # Find all CSV files in the directory
    path = Path(results_dir)
    result_files = list(path.glob('*.csv'))
    
    if not result_files:
        print(f"No CSV files found in {results_dir}")
        return
    
    # Process each CSV file
    for file_path in result_files:
        filename = file_path.name
        
        # Determine dataset name from filename
        dataset_name = None
        for key in dataset_display_names:
            if key in filename:
                dataset_name = dataset_display_names[key]
                break
        
        if dataset_name is None:
            # Skip files that don't match known datasets
            print(f"Warning: Unable to determine dataset for {filename}, skipping")
            continue
        
        # Load CSV
        try:
            df = pd.read_csv(file_path)
            
            # Check if file has the expected structure
            if 'model' not in df.columns:
                print(f"Warning: {filename} doesn't have a 'model' column, skipping")
                continue
                
            # Store in dictionary
            all_results[dataset_name] = df.set_index('model')
            print(f"Loaded results for {dataset_name} from {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    # Prepare data for heatmap
    metrics = ['MAE_mean', 'RMSE_mean', 'R2_mean']
    for metric_name in metrics:
        heatmap_data = []
        
        # Get all model names across datasets
        all_models = set()
        for df in all_results.values():
            all_models.update(df.index)
        
        # Filter out excluded models (if any)
        models_for_display = [model for model in all_models if model not in exclude_models]
        
        # Create matrix for heatmap
        for model in sorted(models_for_display):
            row = {'Model': model}
            for dataset, df in all_results.items():
                if model in df.index and metric_name in df.columns:
                    value = df.loc[model, metric_name]
                    row[dataset] = value
                else:
                    row[dataset] = np.nan
            heatmap_data.append(row)
        
        df_heatmap = pd.DataFrame(heatmap_data).set_index('Model')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Set color range based on metric
        if metric_name.startswith('R2'):
            cmap = 'RdYlGn'  # Red-Yellow-Green (higher is better)
            vmin = vmin_r2 if vmin_r2 is not None else max(0, df_heatmap.min().min())
            vmax = vmax_r2 if vmax_r2 is not None else min(1, df_heatmap.max().max())
        elif metric_name.startswith('MAE'):
            cmap = 'RdYlGn_r'  # Reversed - Green-Yellow-Red (lower is better)
            vmin = vmin_mae if vmin_mae is not None else df_heatmap.min().min()
            vmax = vmax_mae if vmax_mae is not None else df_heatmap.max().max()
        elif metric_name.startswith('RMSE'):
            cmap = 'RdYlGn_r'  # Reversed - Green-Yellow-Red (lower is better)
            vmin = vmin_rmse if vmin_rmse is not None else df_heatmap.min().min()
            vmax = vmax_rmse if vmax_rmse is not None else df_heatmap.max().max()
        else:
            cmap = 'RdYlGn_r'  # Default for other metrics
            vmin = df_heatmap.min().min()
            vmax = df_heatmap.max().max()
        
        # Create the heatmap with annotations
        ax = sns.heatmap(
            df_heatmap, 
            annot=True, 
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            fmt='.3f' if metric_name.startswith('R2') else '.1f',
            linewidths=.5
        )
        
        # Format metric name for title
        metric_display = {
            'MAE_mean': 'Mean Absolute Error (mV)',
            'RMSE_mean': 'Root Mean Squared Error (mV)',
            'R2_mean': 'R² Score'
        }.get(metric_name, metric_name)
        
        # Add suffix to title if models are excluded
        exclude_suffix = ""
        if exclude_models:
            exclude_suffix = " (excluding " + ", ".join(exclude_models) + " from display)"
        
        plt.title(f'Model Performance Comparison - {metric_display}{exclude_suffix}', fontsize=14)
        plt.tight_layout()
        
        # Save figure with suffix if models are excluded
        metric_file = metric_name.replace('_', '-')
        filename_suffix = "_filtered" if exclude_models else ""
        output_file = output_path / f'dataset_comparison_{metric_file}{filename_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created heatmap for {metric_name}{exclude_suffix}: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create comparison heatmaps for protein dataset results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing result CSV files for different datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save comparison visualizations')
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                      help='Models to exclude from the heatmaps completely')
    parser.add_argument('--exclude_from_range', type=str, nargs='+', default=['LinearRegression'],
                      help='Models to exclude when calculating color ranges but still show in plot (default: LinearRegression)')
    parser.add_argument('--use_fixed_ranges', action='store_true',
                      help='Use fixed color ranges across all heatmaps')
    
    args = parser.parse_args()
    
    # Find global min/max values for consistent color scales (if requested)
    vmin_mae = vmax_mae = vmin_rmse = vmax_rmse = vmin_r2 = vmax_r2 = None
    
    if args.use_fixed_ranges:
        print("Calculating global min/max values for color scales...")
        
        # Load and process all results to find global min/max
        all_values = {'MAE_mean': [], 'RMSE_mean': [], 'R2_mean': []}
        
        # Find all CSV files in the directory
        path = Path(args.results_dir)
        result_files = list(path.glob('*.csv'))
        
        for file_path in result_files:
            try:
                df = pd.read_csv(file_path)
                
                # Filter out models to exclude from range calculation but still show in plot
                # This is the key modification - we exclude certain models from affecting the color range
                if 'model' in df.columns:
                    df_for_range = df[~df['model'].isin(args.exclude_from_range)]
                    
                    # Also exclude completely excluded models
                    df_for_range = df_for_range[~df_for_range['model'].isin(args.exclude)]
                else:
                    df_for_range = df
                
                for metric in all_values.keys():
                    if metric in df_for_range.columns:
                        all_values[metric].extend(df_for_range[metric].dropna().tolist())
            except Exception as e:
                print(f"Error processing file for range calculation: {e}")
                continue
        
        # Set global min/max values
        if all_values['MAE_mean']:
            vmin_mae = min(all_values['MAE_mean'])
            vmax_mae = max(all_values['MAE_mean'])
            print(f"MAE range (excluding {', '.join(args.exclude_from_range)}): {vmin_mae:.1f} to {vmax_mae:.1f}")
        
        if all_values['RMSE_mean']:
            vmin_rmse = min(all_values['RMSE_mean'])
            vmax_rmse = max(all_values['RMSE_mean'])
            print(f"RMSE range (excluding {', '.join(args.exclude_from_range)}): {vmin_rmse:.1f} to {vmax_rmse:.1f}")
        
        if all_values['R2_mean']:
            vmin_r2 = max(0, min(all_values['R2_mean']))  # Clip at 0
            vmax_r2 = min(1, max(all_values['R2_mean']))  # Clip at 1
            print(f"R2 range (excluding {', '.join(args.exclude_from_range)}): {vmin_r2:.3f} to {vmax_r2:.3f}")
    
    # Create heatmaps for all models (including LinearRegression, but with color range based on other models)
    create_performance_heatmap(args.results_dir, args.output_dir, metric='MAE_mean',
                             exclude_models=args.exclude,  # Only exclude models that should be completely removed
                             vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                             vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                             vmin_r2=vmin_r2, vmax_r2=vmax_r2)
    create_performance_heatmap(args.results_dir, args.output_dir, metric='RMSE_mean',
                             exclude_models=args.exclude,
                             vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                             vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                             vmin_r2=vmin_r2, vmax_r2=vmax_r2)
    create_performance_heatmap(args.results_dir, args.output_dir, metric='R2_mean',
                             exclude_models=args.exclude,
                             vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                             vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                             vmin_r2=vmin_r2, vmax_r2=vmax_r2)
    
    # If we want to create additional heatmaps that also exclude LinearRegression visually
    # Create a combined exclude list with both args.exclude and args.exclude_from_range
    if args.exclude_from_range:
        full_exclude = args.exclude + [model for model in args.exclude_from_range if model not in args.exclude]
        
        create_performance_heatmap(args.results_dir, args.output_dir, 
                                 metric='MAE_mean', exclude_models=full_exclude,
                                 vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                                 vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                                 vmin_r2=vmin_r2, vmax_r2=vmax_r2)
        create_performance_heatmap(args.results_dir, args.output_dir, 
                                 metric='RMSE_mean', exclude_models=full_exclude,
                                 vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                                 vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                                 vmin_r2=vmin_r2, vmax_r2=vmax_r2)
        create_performance_heatmap(args.results_dir, args.output_dir, 
                                 metric='R2_mean', exclude_models=full_exclude,
                                 vmin_mae=vmin_mae, vmax_mae=vmax_mae,
                                 vmin_rmse=vmin_rmse, vmax_rmse=vmax_rmse, 
                                 vmin_r2=vmin_r2, vmax_r2=vmax_r2)


if __name__ == "__main__":
    main()