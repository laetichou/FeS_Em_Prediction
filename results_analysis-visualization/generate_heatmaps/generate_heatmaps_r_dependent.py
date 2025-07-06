#!/usr/bin/env python3
"""
Multi-Dataset Radius Comparison Script for Protein ML Results

Creates two types of heatmaps:
1. **Dataset Heatmaps**: Models on y-axis, radii on x-axis,
   showing performance for each dataset type (e.g., SF4, FES) and feature type (e.g., all features, bar features).
2. **Model-Specific Heatmaps**: Datasets on y-axis, radii on x-axis,
   showing performance for each model across all dataset types and radii.
   These model-specific heatmaps are created with and without protein datasets

Usage: 
python generate_heatmaps_mae.py \
  --results_dir /path/to/csv/files \
  --output_dir /path/to/output \
  --metric MAE_mean \
  --use_fixed_ranges
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re


def load_csv_results(results_dir):
    """Load all CSV result files and organize by dataset type and cofactor type"""
    
    # Dictionary to store all results
    all_results = {}
    
    # Find all CSV files
    path = Path(results_dir)
    csv_files = list(path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return {}
    
    for file_path in csv_files:
        filename = file_path.name
        print(f"Processing {filename}")
        
        # Parse filename using regex pattern: ml_results_{cofactor}_{feature}.csv
        pattern = r'ml_results_([^_]+)_([^_]+)\.csv'
        match = re.match(pattern, filename)
        
        if not match:
            print(f"Warning: Filename {filename} doesn't match expected pattern 'ml_results_{{cofactor}}_{{feature}}.csv', skipping")
            continue
        
        cofactor_raw, feature_raw = match.groups()
        
        # Map cofactor names to consistent format
        cofactor_mapping = {
            'all': 'all_cofactors',
            'sf4': 'SF4', 
            'SF4': 'SF4',
            'fes': 'FES',
            'FES': 'FES'
        }
        
        # Map feature names to consistent format
        feature_mapping = {
            'all': 'all_features',
            'bar': 'bar_features',
            'protein': 'protein_features'
        }
        
        cofactor_type = cofactor_mapping.get(cofactor_raw.lower())
        feature_type = feature_mapping.get(feature_raw.lower())
        
        if cofactor_type is None:
            print(f"Warning: Unknown cofactor type '{cofactor_raw}' in {filename}, skipping")
            continue
            
        if feature_type is None:
            print(f"Warning: Unknown feature type '{feature_raw}' in {filename}, skipping")
            continue
        
        # Create nested dictionary structure if needed
        if cofactor_type not in all_results:
            all_results[cofactor_type] = {}
        
        # Load CSV data
        try:
            df = pd.read_csv(file_path)
            
            # Check if file has expected structure
            if 'model' not in df.columns:
                print(f"Warning: {filename} doesn't have a 'model' column, skipping")
                continue
            
            all_results[cofactor_type][feature_type] = df
            print(f"Loaded {cofactor_type} - {feature_type}: {len(df)} rows")
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return all_results


def calculate_global_min_max(all_results, metric='MAE_mean', exclude_from_range=None):
    """Calculate global min/max for consistent color range across all heatmaps,
    optionally excluding certain models from affecting the range calculation"""
    
    if exclude_from_range is None:
        exclude_from_range = ["LinearRegression"]  # Default to excluding LinearRegression from range calculation
    
    all_values = []
    
    for cofactor_type in all_results:
        for feature_type in all_results[cofactor_type]:
            df = all_results[cofactor_type][feature_type]
            
            # Skip models that shouldn't affect the color range
            df_for_range = df[~df['model'].isin(exclude_from_range)].copy()
            
            if df_for_range.empty:
                continue
            
            # Check if data has a 'radius' column (as in your example CSV)
            if 'radius' in df_for_range.columns:
                # For pivoted data with a 'radius' column, get the metric values directly
                values = df_for_range[metric].dropna().values
                all_values.extend(values)
            else:
                # The old way of identifying radius columns
                radius_cols = []
                for col in df_for_range.columns:
                    if col.startswith('radius_') or (col.replace('.', '').isdigit() and col != 'model'):
                        radius_cols.append(col)
                
                if not radius_cols:
                    for col in df_for_range.columns:
                        if col != 'model' and col.replace('.', '').replace('_', '').isdigit():
                            radius_cols.append(col)
                
                # Add all metric values to our list
                for col in radius_cols:
                    if col in df_for_range.columns:
                        values = df_for_range[col].dropna().values
                        all_values.extend(values)
    
    if all_values:
        return min(all_values), max(all_values)
    else:
        return None, None


def create_radius_model_heatmaps(all_results, output_dir, metric='MAE_mean', vmin=None, vmax=None, exclude_models=None):
    """Create heatmaps with radii on x-axis and models on y-axis for each dataset combination"""
    
    if exclude_models is None:
        exclude_models = []
    
    # Create output directory for dataset heatmaps
    output_path = Path(output_dir) / "dataset_heatmaps"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Display names for datasets
    display_names = {
        'SF4': {'all_features': 'SF4 - All Features',
                'bar_features': 'SF4 - Bar Features', 
                'protein_features': 'SF4 - Protein Features'},
        'FES': {'all_features': 'FES - All Features',
                'bar_features': 'FES - Bar Features',
                'protein_features': 'FES - Protein Features'},
        'all_cofactors': {'all_features': 'All Cofactors - All Features',
                         'bar_features': 'All Cofactors - Bar Features',
                         'protein_features': 'All Cofactors - Protein Features'}
    }
    
    for cofactor_type in all_results:
        for feature_type in all_results[cofactor_type]:
            df = all_results[cofactor_type][feature_type]
            
            # Print column names to help debug
            print(f"\nColumns in {cofactor_type} - {feature_type} dataset:")
            print(df.columns.tolist())
            
            # Filter out excluded models
            if exclude_models:
                df_filtered = df[~df['model'].isin(exclude_models)].copy()
            else:
                df_filtered = df.copy()
            
            if df_filtered.empty:
                print(f"No data after filtering for {cofactor_type} - {feature_type}, skipping...")
                continue
            
            # Check if data has a 'radius' column (as in your example CSV)
            if 'radius' in df_filtered.columns:
                # Pivot the data to have models as rows and radii as columns
                print(f"Found 'radius' column - pivoting data")
                df_pivot = df_filtered.pivot(index='model', columns='radius', values=metric)
                radius_cols = sorted(df_filtered['radius'].unique().astype(str))
            else:
                print(f"No 'radius' column found - attempting to identify radius columns")
                # The old way of identifying radius columns
                radius_cols = []
                for col in df_filtered.columns:
                    if col.startswith('radius_') or (col.replace('.', '').isdigit() and col != 'model'):
                        radius_cols.append(col)
                
                if not radius_cols:
                    for col in df_filtered.columns:
                        if col != 'model' and col.replace('.', '').replace('_', '').isdigit():
                            radius_cols.append(col)
                
                # Sort radius columns numerically
                radius_cols = sorted(radius_cols, key=lambda x: float(re.findall(r'\d+\.?\d*', x)[0]) if re.findall(r'\d+\.?\d*', x) else 0)
                
                if not radius_cols:
                    print(f"No radius columns found for {cofactor_type} - {feature_type}, skipping...")
                    continue
                
                # Prepare data for heatmap - pivot with models as index and radii as columns
                df_pivot = df_filtered.set_index('model')[radius_cols]
            
            # Clean up column names for display
            clean_cols = []
            for col in df_pivot.columns:
                # Extract number from column name
                numbers = re.findall(r'\d+\.?\d*', str(col))
                if numbers:
                    clean_cols.append(f"{numbers[0]}Å")
                else:
                    clean_cols.append(str(col))
            
            df_pivot.columns = clean_cols
            
            print(f"Heatmap data shape: {df_pivot.shape}")
            print(f"Rows (models): {df_pivot.index.tolist()}")
            print(f"Columns (radii): {df_pivot.columns.tolist()}")
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            
            # Set color parameters
            cmap = 'RdYlGn_r' if 'MAE' in metric or 'RMSE' in metric else 'RdYlGn'
            
            # Create heatmap
            ax = sns.heatmap(
                df_pivot,
                annot=True,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                fmt='.1f',
                linewidths=0.5,
                cbar_kws={'label': 'MAE (mV)' if 'MAE' in metric else metric}
            )
            
            # Format title
            dataset_name = display_names[cofactor_type][feature_type]
            title = f'{dataset_name} - {metric}'
            
            if exclude_models:
                title += f" (excluding {', '.join(exclude_models)})"
            
            if vmin is not None and vmax is not None:
                title += " - Fixed Range"
                
            plt.title(title, fontsize=14, pad=20)
            
            plt.xlabel('Radius from Cofactor Barycenter', fontsize=12)
            plt.ylabel('ML Model', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save figure
            filename = f"{cofactor_type.lower()}_{feature_type.lower()}_{metric.lower().replace('_', '-')}.png"
            output_file = output_path / filename
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created dataset heatmap: {output_file}")


def create_model_specific_heatmaps(all_results, output_dir, metric='MAE_mean', vmin=None, vmax=None, exclude_models=None):
    """Create one heatmap per model showing performance across all dataset types and radii"""
    
    if exclude_models is None:
        exclude_models = []
    
    # Create output directory for model heatmaps
    output_path = Path(output_dir) / "model_heatmaps"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all unique models across all datasets
    all_models = set()
    for cofactor_type in all_results:
        for feature_type in all_results[cofactor_type]:
            df = all_results[cofactor_type][feature_type]
            all_models.update(df['model'].unique())
    
    # Filter out excluded models
    models = [model for model in sorted(all_models) if model not in exclude_models]
    
    # Dataset display names for y-axis
    dataset_labels = []
    dataset_keys = []
    
    for cofactor in ['SF4', 'FES', 'all_cofactors']:
        for feature in ['all_features', 'bar_features', 'protein_features']:
            if cofactor in all_results and feature in all_results[cofactor]:
                cofactor_display = 'All Cofactors' if cofactor == 'all_cofactors' else cofactor
                feature_display = feature.replace('_features', '').title()
                dataset_labels.append(f"{cofactor_display}\n{feature_display}")
                dataset_keys.append((cofactor, feature))
    
    # Get radius values from first available dataset
    radius_values = []
    for cofactor_type in all_results:
        for feature_type in all_results[cofactor_type]:
            df = all_results[cofactor_type][feature_type]
            
            # Check if data has a 'radius' column
            if 'radius' in df.columns:
                radius_values = sorted(df['radius'].unique())
                break
            else:
                # The old way
                radius_cols = []
                for col in df.columns:
                    if col.startswith('radius_') or (col.replace('.', '').isdigit() and col != 'model'):
                        radius_cols.append(col)
                if not radius_cols:
                    for col in df.columns:
                        if col != 'model' and col.replace('.', '').replace('_', '').isdigit():
                            radius_cols.append(col)
                if radius_cols:
                    radius_values = sorted([float(re.findall(r'\d+\.?\d*', col)[0]) 
                                         for col in radius_cols 
                                         if re.findall(r'\d+\.?\d*', col)])
                    break
        if radius_values:
            break
    
    # Clean up column names for display
    clean_cols = [f"{r}Å" for r in radius_values]
    
    # Create heatmap for each model
    for model in models:
        matrix_data = []
        valid_labels = []
        
        for (cofactor, feature), label in zip(dataset_keys, dataset_labels):
            if cofactor in all_results and feature in all_results[cofactor]:
                df = all_results[cofactor][feature]
                
                # Check if data has a 'radius' column
                if 'radius' in df.columns:
                    model_data = df[df['model'] == model]
                    
                    if not model_data.empty:
                        row = []
                        for r in radius_values:
                            r_data = model_data[model_data['radius'] == r]
                            if not r_data.empty:
                                value = r_data[metric].iloc[0]
                                row.append(value)
                            else:
                                row.append(np.nan)
                        
                        matrix_data.append(row)
                        valid_labels.append(label)
                else:
                    model_data = df[df['model'] == model]
                    
                    if not model_data.empty:
                        row = []
                        for r in radius_values:
                            r_col = [col for col in model_data.columns 
                                   if re.search(f"^{r}$|^radius_{r}$", col)]
                            if r_col:
                                value = model_data[r_col[0]].iloc[0]
                                row.append(value)
                            else:
                                row.append(np.nan)
                        
                        matrix_data.append(row)
                        valid_labels.append(label)
        
        if not matrix_data:
            print(f"No data for model {model}, skipping...")
            continue
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(matrix_data, index=valid_labels, columns=clean_cols)
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        
        # Set color parameters
        cmap = 'RdYlGn_r' if 'MAE' in metric or 'RMSE' in metric else 'RdYlGn'
        
        # Create heatmap
        ax = sns.heatmap(
            df_heatmap,
            annot=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            fmt='.1f',
            linewidths=0.5,
            cbar_kws={'label': 'MAE (mV)' if 'MAE' in metric else metric}
        )
        
        title = f'{model} - {metric} Across Datasets and Radii'
        if vmin is not None and vmax is not None:
            title += " - Fixed Range"
        plt.title(title, fontsize=14, pad=20)
        
        plt.xlabel('Radius from Cofactor Barycenter', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        model_safe = re.sub(r'[^\w\-_]', '_', model)
        filename = f'model_{model_safe}_{metric.lower().replace("_", "-")}.png'
        
        output_file = output_path / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created model-specific heatmap: {output_file}")


def create_filtered_model_heatmaps(all_results, output_dir, metric='MAE_mean', vmin=None, vmax=None, 
                                 exclude_models=None, exclude_feature_types=None):
    """Create model-specific heatmaps excluding specified feature types"""
    
    if exclude_models is None:
        exclude_models = []
        
    if exclude_feature_types is None:
        exclude_feature_types = []
    
    # Filter out specified feature types
    filtered_results = {}
    for cofactor_type, features in all_results.items():
        filtered_results[cofactor_type] = {feature_type: data for feature_type, data in features.items() 
                                         if feature_type not in exclude_feature_types}
    
    # Create directory with suffix indicating what was excluded
    suffix = f"no_{'_'.join([f.replace('_features', '') for f in exclude_feature_types])}"
    output_path = Path(output_dir) / f"model_heatmaps_{suffix}"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get all unique models across all datasets
    all_models = set()
    for cofactor_type in filtered_results:
        for feature_type in filtered_results[cofactor_type]:
            df = filtered_results[cofactor_type][feature_type]
            all_models.update(df['model'].unique())
    
    # Filter out excluded models
    models = [model for model in sorted(all_models) if model not in exclude_models]
    
    # Dataset display names for y-axis
    dataset_labels = []
    dataset_keys = []
    
    for cofactor in ['SF4', 'FES', 'all_cofactors']:
        for feature in ['all_features', 'bar_features', 'protein_features']:
            if feature in exclude_feature_types:
                continue
            if cofactor in filtered_results and feature in filtered_results[cofactor]:
                cofactor_display = 'All Cofactors' if cofactor == 'all_cofactors' else cofactor
                feature_display = feature.replace('_features', '').title()
                dataset_labels.append(f"{cofactor_display}\n{feature_display}")
                dataset_keys.append((cofactor, feature))
    
    # Get radius values from first available dataset
    radius_values = []
    for cofactor_type in filtered_results:
        for feature_type in filtered_results[cofactor_type]:
            df = filtered_results[cofactor_type][feature_type]
            
            # Check if data has a 'radius' column
            if 'radius' in df.columns:
                radius_values = sorted(df['radius'].unique())
                break
            else:
                # The old way
                radius_cols = []
                for col in df.columns:
                    if col.startswith('radius_') or (col.replace('.', '').isdigit() and col != 'model'):
                        radius_cols.append(col)
                if not radius_cols:
                    for col in df.columns:
                        if col != 'model' and col.replace('.', '').replace('_', '').isdigit():
                            radius_cols.append(col)
                if radius_cols:
                    radius_values = sorted([float(re.findall(r'\d+\.?\d*', col)[0]) 
                                         for col in radius_cols 
                                         if re.findall(r'\d+\.?\d*', col)])
                    break
        if radius_values:
            break
    
    # Clean up column names for display
    clean_cols = [f"{r}Å" for r in radius_values]
    
    # Create heatmap for each model
    for model in models:
        matrix_data = []
        valid_labels = []
        
        for (cofactor, feature), label in zip(dataset_keys, dataset_labels):
            if cofactor in filtered_results and feature in filtered_results[cofactor]:
                df = filtered_results[cofactor][feature]
                
                # Check if data has a 'radius' column
                if 'radius' in df.columns:
                    model_data = df[df['model'] == model]
                    
                    if not model_data.empty:
                        row = []
                        for r in radius_values:
                            r_data = model_data[model_data['radius'] == r]
                            if not r_data.empty:
                                value = r_data[metric].iloc[0]
                                row.append(value)
                            else:
                                row.append(np.nan)
                        
                        matrix_data.append(row)
                        valid_labels.append(label)
                else:
                    model_data = df[df['model'] == model]
                    
                    if not model_data.empty:
                        row = []
                        for r in radius_values:
                            r_col = [col for col in model_data.columns 
                                   if re.search(f"^{r}$|^radius_{r}$", col)]
                            if r_col:
                                value = model_data[r_col[0]].iloc[0]
                                row.append(value)
                            else:
                                row.append(np.nan)
                        
                        matrix_data.append(row)
                        valid_labels.append(label)
        
        if not matrix_data:
            print(f"No data for model {model}, skipping...")
            continue
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(matrix_data, index=valid_labels, columns=clean_cols)
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        
        # Set color parameters
        cmap = 'RdYlGn_r' if 'MAE' in metric or 'RMSE' in metric else 'RdYlGn'
        
        # Create heatmap
        ax = sns.heatmap(
            df_heatmap,
            annot=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            fmt='.1f',
            linewidths=0.5,
            cbar_kws={'label': 'MAE (mV)' if 'MAE' in metric else metric}
        )
        
        # Construct title with information about excluded feature types
        excluded_features_display = ', '.join([f.replace('_features', '') for f in exclude_feature_types])
        title = f'{model} - {metric} Across Datasets and Radii (No {excluded_features_display})'
        
        if vmin is not None and vmax is not None:
            title += " - Fixed Range"
        plt.title(title, fontsize=14, pad=20)
        
        plt.xlabel('Radius from Cofactor Barycenter', fontsize=12)
        plt.ylabel('Dataset', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        model_safe = re.sub(r'[^\w\-_]', '_', model)
        filename = f'model_{model_safe}_{metric.lower().replace("_", "-")}_no_{excluded_features_display.replace(", ", "_")}.png'
        
        output_file = output_path / filename
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created filtered model heatmap: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create comparison heatmaps for protein ML results')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing result CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save visualizations')
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                      help='Models to exclude from the heatmaps entirely')
    parser.add_argument('--exclude_from_range', type=str, nargs='+', 
                      default=["LinearRegression"],
                      help='Models to include in heatmaps but exclude from color range calculation')
    parser.add_argument('--use_fixed_ranges', action='store_true',
                      help='Use fixed color ranges across all heatmaps')
    parser.add_argument('--metric', type=str, default='MAE_mean',
                      help='Metric to visualize (default: MAE_mean)')
    
    args = parser.parse_args()
    
    # Load all results
    print("Loading CSV results...")
    all_results = load_csv_results(args.results_dir)
    
    if not all_results:
        print("No valid results found. Exiting.")
        return
    
    # Calculate global min/max for color range if requested
    vmin = vmax = None
    if args.use_fixed_ranges:
        print(f"Calculating global range for {args.metric}...")
        print(f"Excluding from range calculation: {args.exclude_from_range}")
        vmin, vmax = calculate_global_min_max(all_results, args.metric, args.exclude_from_range)
        if vmin is not None and vmax is not None:
            print(f"Global range: {vmin:.1f} to {vmax:.1f}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("\nCreating dataset heatmaps (models vs. radii)...")
    create_radius_model_heatmaps(all_results, output_path, args.metric, vmin, vmax, args.exclude)
    
    print("\nCreating model-specific heatmaps (datasets vs. radii)...")
    create_model_specific_heatmaps(all_results, output_path, args.metric, vmin, vmax, args.exclude)
    
    print("\nCreating model-specific heatmaps without protein features...")
    create_filtered_model_heatmaps(all_results, output_path, args.metric, vmin, vmax, 
                                args.exclude, exclude_feature_types=['protein_features'])
    
    print(f"\nAll heatmaps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()