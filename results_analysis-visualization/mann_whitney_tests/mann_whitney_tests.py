import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path
import ast
from itertools import combinations

def parse_list_column(series):
    """Parse string representations of lists in a pandas Series."""
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def load_and_aggregate_data(directory, radii=None):
    """Load and aggregate mae_test_list from CSV files, handling per-radius and aggregated cases."""
    datasets = {
        'radius_independent': ['all_cofactors_protein'],  # Only datasets without radius
        'radius_dependent': ['FES_all', 'FES_bar', 'SF4_bar', 'SF4_all', 'all_cofactors_bar', 'complete_dataset'],  # Only datasets with radius
        'fes': ['FES_all', 'FES_bar', 'FES_protein'],
        'non_fes': ['all_cofactors_bar', 'complete_dataset', 'SF4_bar', 'SF4_all'],
        'fes_individual': {'FES_all': [], 'FES_bar': [], 'FES_protein': []}
    }
    
    models = ['ElasticNet', 'GaussianProcessRegressor', 'GradientBoostingRegressor', 
              'KNeighborsRegressor', 'LinearRegression', 'RandomForestRegressor', 'SVR']
    
    # Data structure: {group: {radius or 'aggregated': {model: [mae_values]}}}
    aggregated_data = {
        'radius_independent': {'aggregated': {}},
        'radius_dependent': {},
        'fes': {},
        'non_fes': {},
        'fes_all': {}, 'fes_bar': {}, 'fes_protein': {}
    }
    
    # Get available radii from radius-dependent datasets only
    if radii is None:
        radii = set()
        for dataset in datasets['radius_dependent']:
            file_path = Path(directory) / f"{dataset}_ml_results_summary.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                if 'radius' in df.columns:
                    radii.update(df['radius'].unique())
        radii = sorted(radii) + ['aggregated']
    else:
        radii = radii + ['aggregated']
    
    # Initialize data structure for each radius and aggregated
    for group in ['radius_dependent', 'fes', 'non_fes', 'fes_all', 'fes_bar', 'fes_protein']:
        for radius in radii:
            aggregated_data[group][radius] = {model: [] for model in models}
    
    # Initialize radius_independent with models
    aggregated_data['radius_independent']['aggregated'] = {model: [] for model in models}
    
    # Load radius-independent datasets (no radius column)
    for dataset in datasets['radius_independent']:
        file_path = Path(directory) / f"{dataset}_ml_results_summary.csv"
        if file_path.exists():
            print(f"Loading radius-independent dataset: {file_path}")
            df = pd.read_csv(file_path)
            for model in models:
                if model in df['model'].values:
                    mae_list = parse_list_column(df[df['model'] == model]['mae_test_list']).iloc[0]
                    aggregated_data['radius_independent']['aggregated'][model].extend(mae_list)
                    print(f"  Added {len(mae_list)} values for {model} in radius_independent")
                    
                    # Add to FES if it's a FES dataset
                    if dataset in datasets['fes']:
                        aggregated_data['fes']['aggregated'][model].extend(mae_list)
                        if dataset == 'FES_protein':
                            aggregated_data['fes_protein']['aggregated'][model].extend(mae_list)
    
    # Load radius-dependent datasets (with radius column)
    for dataset in datasets['radius_dependent']:
        file_path = Path(directory) / f"{dataset}_ml_results_summary.csv"
        if file_path.exists():
            print(f"Loading radius-dependent dataset: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if radius column exists
            if 'radius' not in df.columns:
                print(f"  Warning: No 'radius' column found in {dataset}, skipping...")
                continue
                
            for model in models:
                if model in df['model'].values:
                    model_df = df[df['model'] == model]
                    
                    # Aggregated across all radii
                    mae_lists = parse_list_column(model_df['mae_test_list'])
                    combined_mae = []
                    for mae_list in mae_lists:
                        combined_mae.extend(mae_list)
                    
                    aggregated_data['radius_dependent']['aggregated'][model].extend(combined_mae)
                    print(f"  Added {len(combined_mae)} aggregated values for {model} in radius_dependent")
                    
                    # Categorize into FES and non-FES
                    if dataset in datasets['fes']:
                        aggregated_data['fes']['aggregated'][model].extend(combined_mae)
                        # Handle individual FES datasets
                        dataset_key = dataset.lower()
                        if dataset_key in aggregated_data:
                            aggregated_data[dataset_key]['aggregated'][model].extend(combined_mae)
                    else:
                        aggregated_data['non_fes']['aggregated'][model].extend(combined_mae)
                    
                    # Per radius
                    for radius in radii[:-1]:  # Exclude 'aggregated'
                        radius_df = model_df[model_df['radius'] == radius]
                        if not radius_df.empty:
                            mae_list = parse_list_column(radius_df['mae_test_list']).iloc[0]
                            aggregated_data['radius_dependent'][radius][model].extend(mae_list)
                            print(f"    Added {len(mae_list)} values for {model} at radius {radius}")
                            
                            if dataset in datasets['fes']:
                                aggregated_data['fes'][radius][model].extend(mae_list)
                                # Handle individual FES datasets
                                dataset_key = dataset.lower()
                                if dataset_key in aggregated_data and radius in aggregated_data[dataset_key]:
                                    aggregated_data[dataset_key][radius][model].extend(mae_list)
                            else:
                                aggregated_data['non_fes'][radius][model].extend(mae_list)
    
    # Special handling for FES_protein (radius-independent)
    fes_protein_file = Path(directory) / "FES_protein_ml_results_summary.csv"
    if fes_protein_file.exists():
        print(f"Loading FES_protein dataset: {fes_protein_file}")
        df = pd.read_csv(fes_protein_file)
        for model in models:
            if model in df['model'].values:
                mae_list = parse_list_column(df[df['model'] == model]['mae_test_list']).iloc[0]
                # FES_protein is radius-independent, so add to aggregated only
                aggregated_data['fes_protein']['aggregated'][model].extend(mae_list)
                aggregated_data['fes']['aggregated'][model].extend(mae_list)
                print(f"  Added {len(mae_list)} values for {model} in FES_protein")
    
    # Print summary for debugging
    print(f"\nData loaded successfully:")
    print(f"Available radii: {radii}")
    print(f"Available models: {models}")
    for group, group_data in aggregated_data.items():
        print(f"\n{group}:")
        for radius_key, radius_data in group_data.items():
            non_empty_models = [m for m, v in radius_data.items() if v]
            print(f"  {radius_key}: {len(non_empty_models)} models with data")
            for model in non_empty_models:
                print(f"    {model}: {len(radius_data[model])} values")
    
    return aggregated_data, models, radii

def perform_mann_whitney_tests(data, models, group, radius, model_of_interest):
    """Perform Mann-Whitney U tests for a specific group, radius, and model of interest."""
    results = []
    comparisons = [(model_of_interest, other_model) for other_model in models if other_model != model_of_interest]
    n_comparisons = len(comparisons)
    
    for model1, model2 in comparisons:
        if data[group][radius][model1] and data[group][radius][model2]:
            stat, p_value = mannwhitneyu(
                data[group][radius][model1], data[group][radius][model2], alternative='less'
            )
            n1, n2 = len(data[group][radius][model1]), len(data[group][radius][model2])
            effect_size = 1 - (2 * stat) / (n1 * n2)
            results.append({
                'comparison': f"{model1} vs {model2}",
                'group': group,
                'radius': radius,
                'statistic': stat,
                'p_value': p_value,
                'adjusted_p_value': min(p_value * n_comparisons, 1.0),
                'effect_size': effect_size
            })
    
    return results

def compare_fes_vs_non_fes(data, models, radius):
    """Compare FES vs non-FES datasets for all models."""
    results = []
    for model in models:
        if data['fes'][radius][model] and data['non_fes'][radius][model]:
            stat, p_value = mannwhitneyu(
                data['fes'][radius][model], data['non_fes'][radius][model], alternative='less'
            )
            n1, n2 = len(data['fes'][radius][model]), len(data['non_fes'][radius][model])
            effect_size = 1 - (2 * stat) / (n1 * n2)
            results.append({
                'comparison': f"{model}: FES vs non-FES",
                'group': 'FES vs non-FES',
                'radius': radius,
                'statistic': stat,
                'p_value': p_value,
                'adjusted_p_value': min(p_value * len(models), 1.0),
                'effect_size': effect_size
            })
    
    return results

def main(directory, radii=None):
    """Main function to run Mann-Whitney U tests."""
    # Create output directory if it doesn't exist
    output_dir = Path(directory)
    output_dir.mkdir(exist_ok=True)
    
    data, models, radii = load_and_aggregate_data(directory, radii)
    
    output_file = output_dir / 'mann_whitney_results_comprehensive.txt'
    with open(output_file, 'w') as f:
        f.write("Mann-Whitney U Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Available models: {', '.join(models)}\n")
        f.write(f"Available radii: {radii}\n")
        f.write(f"Test models: {{'test1': 'GradientBoostingRegressor', 'test2': 'GradientBoostingRegressor', "
                f"'test3': 'RandomForestRegressor', 'test4': 'All models'}}\n\n")
        
        # Test 1: GradientBoostingRegressor on radius-Independent Datasets
        f.write("Test 1: GradientBoostingRegressor on radius-Independent Datasets\n")
        if 'GradientBoostingRegressor' in models:
            results_ri = perform_mann_whitney_tests(data, models, 'radius_independent', 'aggregated', 'GradientBoostingRegressor')
            if results_ri:
                for res in results_ri:
                    f.write(f"Comparison: {res['comparison']}\n")
                    f.write(f"Statistic: {res['statistic']:.2f}, p-value: {res['p_value']:.4f}, "
                            f"Adjusted p-value: {res['adjusted_p_value']:.4f}, Effect size: {res['effect_size']:.4f}\n")
                    print(f"Test 1 - {res['comparison']}: p-value={res['p_value']:.4f}, "
                          f"Adjusted p-value={res['adjusted_p_value']:.4f}, Effect size={res['effect_size']:.4f}")
            else:
                f.write("No comparisons available.\n")
                print("Test 1: No comparisons available.")
        else:
            f.write("GradientBoostingRegressor not found in models.\n")
            print("Test 1: GradientBoostingRegressor not found in models.")
        f.write("\n")
        
        # Test 2: GradientBoostingRegressor on radius-Dependent Datasets
        if 'GradientBoostingRegressor' in models:
            for radius in radii:
                f.write(f"Test 2: GradientBoostingRegressor on radius-Dependent Datasets (Radius: {radius})\n")
                results_rd = perform_mann_whitney_tests(data, models, 'radius_dependent', radius, 'GradientBoostingRegressor')
                if results_rd:
                    for res in results_rd:
                        f.write(f"Comparison: {res['comparison']}\n")
                        f.write(f"Statistic: {res['statistic']:.2f}, p-value: {res['p_value']:.4f}, "
                                f"Adjusted p-value: {res['adjusted_p_value']:.4f}, Effect size: {res['effect_size']:.4f}\n")
                        print(f"Test 2 (Radius: {radius}) - {res['comparison']}: p-value={res['p_value']:.4f}, "
                              f"Adjusted p-value={res['adjusted_p_value']:.4f}, Effect size={res['effect_size']:.4f}")
                else:
                    f.write("No comparisons available.\n")
                    print(f"Test 2 (Radius: {radius}): No comparisons available.")
                f.write("\n")
        
        # Test 3: RandomForestRegressor on Individual FES Datasets
        if 'RandomForestRegressor' in models:
            for dataset in ['fes_all', 'fes_bar', 'fes_protein']:
                if dataset in data:
                    for radius in radii:
                        if radius in data[dataset]:
                            f.write(f"Test 3: RandomForestRegressor on {dataset} (Radius: {radius})\n")
                            results_fes = perform_mann_whitney_tests(data, models, dataset, radius, 'RandomForestRegressor')
                            if results_fes:
                                for res in results_fes:
                                    f.write(f"Comparison: {res['comparison']}\n")
                                    f.write(f"Statistic: {res['statistic']:.2f}, p-value: {res['p_value']:.4f}, "
                                            f"Adjusted p-value: {res['adjusted_p_value']:.4f}, Effect size: {res['effect_size']:.4f}\n")
                                    print(f"Test 3 ({dataset}, Radius: {radius}) - {res['comparison']}: p-value={res['p_value']:.4f}, "
                                          f"Adjusted p-value={res['adjusted_p_value']:.4f}, Effect size={res['effect_size']:.4f}")
                            else:
                                f.write("No comparisons available.\n")
                                print(f"Test 3 ({dataset}, Radius: {radius}): No comparisons available.")
                            f.write("\n")
        
        # Test 4: FES vs non-FES Datasets
        for radius in radii:
            f.write(f"Test 4: FES vs non-FES Datasets (Radius: {radius})\n")
            results_fes_vs_non = compare_fes_vs_non_fes(data, models, radius)
            if results_fes_vs_non:
                for res in results_fes_vs_non:
                    f.write(f"Comparison: {res['comparison']}\n")
                    f.write(f"Statistic: {res['statistic']:.2f}, p-value: {res['p_value']:.4f}, "
                            f"Adjusted p-value: {res['adjusted_p_value']:.4f}, Effect size: {res['effect_size']:.4f}\n")
                    print(f"Test 4 (Radius: {radius}) - {res['comparison']}: p-value={res['p_value']:.4f}, "
                          f"Adjusted p-value={res['adjusted_p_value']:.4f}, Effect size={res['effect_size']:.4f}")
            else:
                f.write("No comparisons available.\n")
                print(f"Test 4 (Radius: {radius}): No comparisons available.")
            f.write("\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    directory = "input"  # Update this path
    radii = None  # Update with your radii; set to None to auto-detect
    main(directory, radii)