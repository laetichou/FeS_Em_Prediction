import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add report_tests to path
from interpretation.interpretation_final import ResultAnalyzer, OutputTee
from datetime import datetime
from scipy import stats as stats_scipy

def aggregate_radius_dependent(parent_dir: str):
    parent_dir = Path(parent_dir)
    output_dir = parent_dir / "aggregated_interpretation"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = output_dir / "aggregation_log.txt"
    original_stdout = sys.stdout
    log_stream = open(log_file, 'w')
    sys.stdout = OutputTee(original_stdout, log_stream)
    
    print(f"=== Aggregated Iron-Sulfur Redox Potential Prediction - Radius-Dependent Results ===")
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
    
    # Aggregate model performance and collect stability/trend data
    model_performance = []
    stability_data = {}  # {dataset: {model: {'mae_std_list': list, 'rmse_std_list': list, etc.}}}
    trend_data = {}     # {model: {'mae_slopes': list, 'rmse_slopes': list, etc.}}
    
    for dataset_name, results in all_results.items():
        for radius, radius_results in results.items():
            for model_name, metrics in radius_results.items():
                model_performance.append({
                    'dataset': dataset_name,
                    'radius': int(radius),
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
        
        # Collect stability data for all metrics
        if dataset_name not in stability_data:
            stability_data[dataset_name] = {}
        for model_name in radius_results:
            if model_name not in stability_data[dataset_name]:
                stability_data[dataset_name][model_name] = {
                    'mae_std_list': [], 'rmse_std_list': [], 'r2_std_list': [], 'spearman_std_list': []
                }
            for radius in results:
                if model_name in results[radius]:
                    model_metrics = results[radius][model_name]
                    stability_data[dataset_name][model_name]['mae_std_list'].append(model_metrics.get('MAE_std', 0))
                    stability_data[dataset_name][model_name]['rmse_std_list'].append(model_metrics.get('RMSE_std', 0))
                    stability_data[dataset_name][model_name]['r2_std_list'].append(model_metrics.get('R2_std', 0))
                    stability_data[dataset_name][model_name]['spearman_std_list'].append(model_metrics.get('Spearman_std', 0))
    
    # Aggregate stability data for all metrics
    aggregated_stability = {}  # {model: {'mae_std_avg': float, 'rmse_std_avg': float, etc.}}
    for dataset_name, models in stability_data.items():
        for model_name, data in models.items():
            if model_name not in aggregated_stability:
                aggregated_stability[model_name] = {
                    'mae_std_avg': 0, 'rmse_std_avg': 0, 'r2_std_avg': 0, 'spearman_std_avg': 0, 'datasets': 0
                }
            for metric in ['mae_std_list', 'rmse_std_list', 'r2_std_list', 'spearman_std_list']:
                if data[metric]:
                    metric_key = metric.replace('_list', '_avg')
                    aggregated_stability[model_name][metric_key] += np.mean(data[metric])
            aggregated_stability[model_name]['datasets'] += 1
    
    for model_name in aggregated_stability:
        if aggregated_stability[model_name]['datasets'] > 0:
            for metric in ['mae_std_avg', 'rmse_std_avg', 'r2_std_avg', 'spearman_std_avg']:
                aggregated_stability[model_name][metric] /= aggregated_stability[model_name]['datasets']
    
    # Aggregate trend data for all metrics
    for dataset_name, results in all_results.items():
        for model_name in results[next(iter(results))]:  # Sample first radius
            if model_name not in trend_data:
                trend_data[model_name] = {
                    'mae_slopes': [], 'rmse_slopes': [], 'r2_slopes': [], 'spearman_slopes': [],
                    'mae_p_values': [], 'rmse_p_values': [], 'r2_p_values': [], 'spearman_p_values': [],
                    'datasets': 0
                }
            radii = [int(r) for r in results.keys()]
            
            # Get values for each metric
            mae_values = [results[str(r)][model_name].get('MAE_mean', float('inf')) for r in radii if model_name in results[str(r)]]
            rmse_values = [results[str(r)][model_name].get('RMSE_mean', float('inf')) for r in radii if model_name in results[str(r)]]
            r2_values = [results[str(r)][model_name].get('R2_mean', 0) for r in radii if model_name in results[str(r)]]
            spearman_values = [results[str(r)][model_name].get('Spearman_mean', 0) for r in radii if model_name in results[str(r)]]
            
            if len(radii) > 3 and len(mae_values) == len(radii):  # Enough data for trend
                for metric_name, values in [('mae', mae_values), ('rmse', rmse_values), ('r2', r2_values), ('spearman', spearman_values)]:
                    slope, _, r_value, p_value, _ = stats_scipy.linregress(radii, values)
                    trend_data[model_name][f'{metric_name}_slopes'].append(slope)
                    trend_data[model_name][f'{metric_name}_p_values'].append(p_value)
                trend_data[model_name]['datasets'] += 1
    
    # Average trend data for all metrics
    for model_name in trend_data:
        if trend_data[model_name]['datasets'] > 0:
            for metric in ['mae', 'rmse', 'r2', 'spearman']:
                trend_data[model_name][f'{metric}_slope_avg'] = np.mean(trend_data[model_name][f'{metric}_slopes'])
                trend_data[model_name][f'{metric}_p_value_avg'] = np.mean(trend_data[model_name][f'{metric}_p_values'])
    
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
    all_models_without_lr.discard('LinearRegression')
    top_models = sorted(list(all_models_without_lr))
    
    # Define colors for plotting
    colors = sns.color_palette("husl", len(top_models))  # Distinct colors for models
    
    # Print comprehensive statistics
    print("\nOverall Performance Statistics Across All Radius-Dependent Datasets:")
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
        lr_stats = overall_stats['rmse_mean'].loc[model_name]
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
    
    # Find best performers for each metric
    best_mae_model = overall_stats['mae_mean']['mean'].idxmin()
    best_mae_value = overall_stats['mae_mean']['mean'].min()
    best_rmse_model = overall_stats['rmse_mean']['mean'].idxmin()
    best_rmse_value = overall_stats['rmse_mean']['mean'].min()
    best_r2_model = overall_stats['r2_mean']['mean'].idxmax()
    best_r2_value = overall_stats['r2_mean']['mean'].max()
    best_spearman_model = overall_stats['spearman_mean']['mean'].idxmax()
    best_spearman_value = overall_stats['spearman_mean']['mean'].max()
    
    # Find optimal radius per model per dataset
    optimal_df = df.loc[df.groupby(['dataset', 'model'])['mae_mean'].idxmin()]
    top_models_by_mae = optimal_df.nsmallest(5, 'mae_mean')
    
    print("\nTop 5 Models by MAE at Optimal Radii Across All Datasets:")
    print("="*60)
    for idx, row in top_models_by_mae.iterrows():
        print(f"{row['model']} ({row['dataset']}, r={row['radius']}Å): "
              f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
              f"RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f} mV, "
              f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}, "
              f"Spearman={row['spearman_mean']:.3f}±{row['spearman_std']:.3f}")
    
    # Stability and trend summary for all metrics
    print("\nStability and Trend Analysis Across All Datasets:")
    print("="*80)
    for model_name in set(aggregated_stability.keys()).union(trend_data.keys()):
        if model_name in aggregated_stability:
            mae_std_avg = aggregated_stability[model_name].get('mae_std_avg', 'N/A')
            rmse_std_avg = aggregated_stability[model_name].get('rmse_std_avg', 'N/A')
            r2_std_avg = aggregated_stability[model_name].get('r2_std_avg', 'N/A')
            spearman_std_avg = aggregated_stability[model_name].get('spearman_std_avg', 'N/A')
        else:
            mae_std_avg = rmse_std_avg = r2_std_avg = spearman_std_avg = 'N/A'
        
        if model_name in trend_data:
            mae_slope_avg = trend_data[model_name].get('mae_slope_avg', 'N/A')
            rmse_slope_avg = trend_data[model_name].get('rmse_slope_avg', 'N/A')
            r2_slope_avg = trend_data[model_name].get('r2_slope_avg', 'N/A')
            spearman_slope_avg = trend_data[model_name].get('spearman_slope_avg', 'N/A')
        else:
            mae_slope_avg = rmse_slope_avg = r2_slope_avg = spearman_slope_avg = 'N/A'
        
        print(f"{model_name}:")
        print(f"  Stability: MAE_std={mae_std_avg:.2f}, RMSE_std={rmse_std_avg:.2f}, R2_std={r2_std_avg:.3f}, Spearman_std={spearman_std_avg:.3f}")
        print(f"  Trends: MAE_slope={mae_slope_avg:.3f}, RMSE_slope={rmse_slope_avg:.3f}, R2_slope={r2_slope_avg:.4f}, Spearman_slope={spearman_slope_avg:.4f}")
    
    # Generate heatmap for stability per radius
    models = sorted(aggregated_stability.keys())
    radii = sorted({int(radius) for dataset in all_results.values() for radius in dataset.keys()})
    heatmap_data = pd.DataFrame(index=models, columns=radii)
    for model in models:
        for radius in radii:
            mae_std_values = [stability_data[ds][model]['mae_std_list'][i] 
                             for ds in stability_data 
                             for i, r in enumerate(sorted([int(k) for k in all_results[ds].keys()])) 
                             if r == radius and model in stability_data[ds] and i < len(stability_data[ds][model]['mae_std_list'])]
            heatmap_data.at[model, radius] = np.mean(mae_std_values) if mae_std_values else np.nan
    heatmap_data = heatmap_data.astype(float)
    valid_data = heatmap_data.drop('LinearRegression', errors='ignore').values
    vmin = np.nanmin(valid_data[valid_data == valid_data])
    vmax = np.nanmax(valid_data[valid_data == valid_data])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', vmin=vmin, vmax=vmax, 
                cbar_kws={'label': 'Mean MAE Std (mV)'})
    plt.xlabel('Radius (Å)')
    plt.ylabel('Model')
    plt.title('Stability (Mean MAE Std) per Radius Across All Datasets\n(LinearRegression excluded from scale)')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stability heatmap to: {output_dir / 'stability_heatmap.png'}")
    
    if 'LinearRegression' in heatmap_data.index:
        filtered_heatmap_data = heatmap_data.drop('LinearRegression')
        plt.figure(figsize=(12, 8))
        sns.heatmap(filtered_heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                    vmin=vmin, vmax=vmax, cbar_kws={'label': 'Mean MAE Std (mV)'})
        plt.xlabel('Radius (Å)')
        plt.ylabel('Model')
        plt.title('Stability (Mean MAE Std) per Radius Across All Datasets\n(LinearRegression completely excluded)')
        plt.tight_layout()
        plt.savefig(output_dir / 'stability_heatmap_no_lr.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved filtered stability heatmap to: {output_dir / 'stability_heatmap_no_lr.png'}")
    
    # Collage of MAE Distribution (Box Plots) for all 6 datasets
    datasets = ['complete_dataset', 'FES_all', 'SF4_all', 'all_cofactors_bar', 'FES_bar', 'SF4_bar']
    mae_data = {dataset: {model: [] for model in top_models} for dataset in datasets}
    for dataset_name, results in all_results.items():
        if dataset_name in datasets:
            for radius in sorted([int(r) for r in results.keys()]):
                for model_name in top_models:
                    if model_name in results[str(radius)]:
                        mae_data[dataset_name][model_name].append(results[str(radius)][model_name]['MAE_mean'])
    
    # 2x3 layout for MAE distribution
    fig_2x3, axes_2x3 = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    for i, dataset in enumerate(datasets):
        row, col = divmod(i, 3)
        ax = axes_2x3[row, col]
        data_to_plot = [mae_data[dataset][model] for model in top_models]
        bp = ax.boxplot(data_to_plot, labels=None, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f'{dataset}')
        ax.set_ylabel('MAE (mV)') if col == 0 else ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_distribution_collage_2x3.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3x2 layout for MAE distribution
    fig_3x2, axes_3x2 = plt.subplots(3, 2, figsize=(10, 15), sharey=True)
    report_datasets = ['complete_dataset', 'FES_all', 'SF4_all']
    appendix_datasets = ['all_cofactors_bar', 'FES_bar', 'SF4_bar']
    for i, dataset in enumerate(report_datasets + appendix_datasets):
        row = i // 2
        col = i % 2
        ax = axes_3x2[row, col]
        data_to_plot = [mae_data[dataset][model] for model in top_models]
        bp = ax.boxplot(data_to_plot, labels=None, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f'{dataset}')
        ax.set_ylabel('MAE (mV)') if col == 0 else ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_distribution_collage_3x2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save legend as separate image
    fig_legend, ax_legend = plt.subplots(figsize=(2, len(top_models) * 0.5))
    ax_legend.axis('off')
    handles, labels = [], []
    for model, color in zip(top_models, colors):
        handle = plt.Line2D([0], [0], color=color, lw=4, label=model)
        handles.append(handle)
        labels.append(model)
    ax_legend.legend(handles=handles, labels=labels, loc='center')
    plt.savefig(output_dir / 'legend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved legend to: {output_dir / 'legend.png'}")
    
    # Performance Trends Collages
    selected_datasets_report = ['complete_dataset', 'FES_all', 'SF4_all']
    selected_datasets_appendix = ['all_cofactors_bar', 'FES_bar', 'SF4_bar']
    
    for title, datasets_group in [('Report', selected_datasets_report), ('Appendix', selected_datasets_appendix)]:
        # 2x3 layout for performance trends
        fig_2x3, axes_2x3 = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey='row')
        for i, dataset in enumerate(datasets_group):
            row = i // 3
            col = i % 3
            ax_mae = axes_2x3[0, col]
            ax_r2 = axes_2x3[1, col]
            for model_name, color in zip(top_models, colors):
                radii = sorted([int(r) for r in all_results[dataset].keys()])
                mae_values = [all_results[dataset][str(r)].get(model_name, {}).get('MAE_mean', np.nan) for r in radii]
                r2_values = [all_results[dataset][str(r)].get(model_name, {}).get('R2_mean', np.nan) for r in radii]
                ax_mae.plot(radii, mae_values, marker='o', color=color, label=model_name if i == 0 else "")
                ax_r2.plot(radii, r2_values, marker='o', color=color, label=model_name if i == 0 else "")
            ax_mae.set_title(f'{dataset} - MAE')
            ax_r2.set_title(f'{dataset} - R²')
            ax_mae.set_ylabel('MAE (mV)') if col == 0 else ax_mae.set_ylabel('')
            ax_r2.set_ylabel('R²') if col == 0 else ax_r2.set_ylabel('')
        axes_2x3[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Radius (Å)')
        plt.tight_layout()
        plt.savefig(output_dir / f'performance_trends_{title.lower()}_2x3.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3x2 layout
        fig_3x2, axes_3x2 = plt.subplots(3, 2, figsize=(10, 15), sharex=True)
        for i, dataset in enumerate(datasets_group):
            ax_mae = axes_3x2[i, 0]
            ax_r2 = axes_3x2[i, 1]
            for model_name, color in zip(top_models, colors):
                radii = sorted([int(r) for r in all_results[dataset].keys()])
                mae_values = [all_results[dataset][str(r)].get(model_name, {}).get('MAE_mean', np.nan) for r in radii]
                r2_values = [all_results[dataset][str(r)].get(model_name, {}).get('R2_mean', np.nan) for r in radii]
                ax_mae.plot(radii, mae_values, marker='o', color=color, label=model_name if i == 0 else "")
                ax_r2.plot(radii, r2_values, marker='o', color=color, label=model_name if i == 0 else "")
            ax_mae.set_title(f'{dataset} - MAE')
            ax_r2.set_title(f'{dataset} - R²')
            ax_mae.set_ylabel('MAE (mV)')
            ax_r2.set_ylabel('R²')
        axes_3x2[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Radius (Å)')
        plt.tight_layout()
        plt.savefig(output_dir / f'performance_trends_{title.lower()}_3x2.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save report
    report_file = output_dir / 'aggregated_report.txt'
    with open(report_file, 'w') as f:
        f.write("Aggregated Iron-Sulfur Cofactor Redox Potential Prediction - Radius-Dependent Report\n")
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
        
        f.write("\nTOP 5 MODELS BY MAE AT OPTIMAL RADII:\n")
        f.write("-" * 30 + "\n")
        for idx, row in top_models_by_mae.iterrows():
            f.write(f"{row['model']} ({row['dataset']}, r={row['radius']}Å): "
                    f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                    f"RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f} mV, "
                    f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}, "
                    f"Spearman={row['spearman_mean']:.3f}±{row['spearman_std']:.3f}\n")
    
    print(f"Aggregated report saved to: {report_file}")
    sys.stdout = original_stdout
    log_stream.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate radius-dependent ML results with stability and trends')
    parser.add_argument('--parent_dir', type=str, required=True, help='Parent directory containing ML output folders')
    args = parser.parse_args()
    aggregate_radius_dependent(args.parent_dir)