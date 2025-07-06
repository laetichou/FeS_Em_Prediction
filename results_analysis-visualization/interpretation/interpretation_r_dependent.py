#!/usr/bin/env python3
"""
Enhanced utility functions for analyzing ML results and creating visualizations

Run with: python enhanced_interpretation.py --results_dir path/to/output --literature_mae 36.4
python enhanced_interpretation.py --literature_mae 36.4 --exclude_models LinearRegression --results_dir path/to/output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

class ResultAnalyzer:
    """Analyze and visualize ML training results with enhanced features"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(f"{results_dir}_interpretation")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created interpretation output directory: {self.output_dir}")
        
        self.log_file = self.output_dir / "interpretation_log.txt"
        self.original_stdout = sys.stdout
        self.log_stream = open(self.log_file, 'w')
        sys.stdout = OutputTee(self.original_stdout, self.log_stream)
        
        print(f"=== Iron-Sulfur Redox Potential Prediction - Results Interpretation ===")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Source results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        self.results = self._load_and_validate_results()

    def _load_and_validate_results(self):
        """Load and validate ML results from JSON file"""
        results_file = self.results_dir / 'ml_results.json'
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        print(f"Loading results from: {results_file}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Validate structure
        if not results or not any(isinstance(v, dict) for v in results.values()):
            raise ValueError(f"Invalid results format in {results_file}")
        return results

    def find_best_models(self, metric='MAE_mean', top_n=5):
        """Find the top N best models across all radii with validation"""
        model_performance = []
        sample_radius = next(iter(self.results))
        sample_model = next(iter(self.results[sample_radius]))
        
        print(f"Debug - Sample metrics for radius {sample_radius}, model {sample_model}:")
        for k, v in self.results[sample_radius][sample_model].items():
            print(f"  {k}: {v}")
        
        if metric not in self.results[sample_radius][sample_model]:
            available = self.results[sample_radius][sample_model].keys()
            print(f"Warning: Metric '{metric}' not found. Available: {', '.join(available)}")
            metric = next((m for m in ['MAE_mean', 'R2_mean'] if m in available), list(available)[0])
            print(f"Using fallback metric: {metric}")
        
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                try:
                    model_performance.append({
                        'radius': int(radius),
                        'model': model_name,
                        'metric_value': metrics[metric],
                        'mae_mean': metrics.get('MAE_mean', 0),
                        'mae_std': metrics.get('MAE_std', 0),
                        'r2_mean': metrics.get('R2_mean', 0),
                        'r2_std': metrics.get('R2_std', 0)
                    })
                except KeyError as e:
                    print(f"Error for radius {radius}, model {model_name}: {e}")
        
        df = pd.DataFrame(model_performance)
        ascending = True if 'MAE' in metric or 'RMSE' in metric else False
        best_models = df.nsmallest(top_n, 'metric_value') if ascending else df.nlargest(top_n, 'metric_value')
        
        print(f"\nTop {top_n} models by {metric}:")
        print("="*60)
        for idx, row in best_models.iterrows():
            print(f"{row['model']} (r={row['radius']}Å): "
                  f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                  f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}")
        return best_models

    def statistical_significance_test(self, model1_name, model1_radius, model2_name, model2_radius, alpha=0.05):
        """Perform statistical significance test between two models"""
        results1 = self.results[str(model1_radius)][model1_name]
        results2 = self.results[str(model2_radius)][model2_name]
        
        mae1_mean, mae1_std = results1['MAE_mean'], results1['MAE_std']
        mae2_mean, mae2_std = results2['MAE_mean'], results2['MAE_std']
        
        pooled_std = np.sqrt((mae1_std**2 + mae2_std**2) / 2)
        t_stat = abs(mae1_mean - mae2_mean) / (pooled_std * np.sqrt(2/10))  # Assuming 10 repeats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=18))  # df = 2*10-2
        
        is_significant = p_value < alpha
        print(f"\nStatistical Comparison ({model1_name} vs {model2_name}):")
        print(f"Model 1 (r={model1_radius}Å): MAE = {mae1_mean:.2f}±{mae1_std:.2f} mV")
        print(f"Model 2 (r={model2_radius}Å): MAE = {mae2_mean:.2f}±{mae2_std:.2f} mV")
        print(f"P-value: {p_value:.4f}, Significant: {is_significant} (α={alpha})")
        return p_value, is_significant

    def analyze_radius_trends(self, exclude_models=None):
        """Analyze and visualize model performance trends with radius for all models"""
        if exclude_models is None:
            exclude_models = []
            
        suffix = "_filtered" if exclude_models else ""
        
        sample_radius = next(iter(self.results))
        metric = 'MAE_mean' if 'MAE_mean' in self.results[sample_radius][next(iter(self.results[sample_radius]))] else list(self.results[sample_radius].values())[0].keys()[0]
        print(f"Using metric '{metric}' for trend analysis{' (filtered)' if exclude_models else ''}")
        
        # Get all unique model names
        all_models = set()
        for radius_results in self.results.values():
            all_models.update(radius_results.keys())
        
        # Filter out excluded models
        models = [model for model in all_models if model not in exclude_models]
        if not models:
            print(f"No models remain after filtering. Skipping trend analysis.")
            return
        
        print(f"Analyzing trends for {len(models)} models: {', '.join(models)}")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model_name in enumerate(sorted(models)):
            radii, values = [], []
            for radius in sorted([int(r) for r in self.results.keys()]):
                if model_name in self.results[str(radius)]:
                    radii.append(radius)
                    values.append(self.results[str(radius)][model_name][metric])
            
            if len(radii) > 3:
                slope, _, r_value, p_value, _ = stats.linregress(radii, values)
                ax.plot(radii, values, marker='o', label=f"{model_name} (slope={slope:.3f}, p={p_value:.4f})", 
                    color=colors[i])
                print(f"  {model_name}: Slope={slope:.3f}, R={r_value:.3f}, p={p_value:.4f}")
            else:
                ax.plot(radii, values, marker='o', label=model_name, color=colors[i])
                print(f"  {model_name}: Insufficient data for trend analysis")
        
        ax.set_xlabel('Radius (Å)')
        ax.set_ylabel(f'{metric} (mV)' if 'MAE' in metric else metric)
        title = 'Performance Trends Across Radii for All Models'
        if exclude_models:
            title += f" (Excluding: {', '.join(exclude_models)})"
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'radius_trends{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved radius trends plot to: {self.output_dir / f'radius_trends{suffix}.png'}")

    def analyze_model_stability(self, exclude_models=None):
        """Analyze model stability using existing results, with option to exclude models"""
        if exclude_models is None:
            exclude_models = []
        
        suffix = "_filtered" if exclude_models else ""
        
        stability_metrics = {}
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                # Skip excluded models
                if model_name in exclude_models:
                    continue
                    
                if model_name not in stability_metrics:
                    stability_metrics[model_name] = {'radius': [], 'cv_std': []}
                stability_metrics[model_name]['radius'].append(int(radius))
                stability_metrics[model_name]['cv_std'].append(metrics.get('MAE_std', 0))
        
        if not stability_metrics:
            print(f"No models remaining after filtering. Skipping stability analysis.")
            return
        
        # Create a single plot instead of two
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a colormap for consistency across plots
        models = list(stability_metrics.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, (model, data) in enumerate(stability_metrics.items()):
            # Sort by radius to ensure correct line plotting
            sorted_indices = np.argsort(data['radius'])
            radii = [data['radius'][i] for i in sorted_indices]
            cv_std = [data['cv_std'][i] for i in sorted_indices]
            
            ax.plot(radii, cv_std, marker='o', label=model, color=colors[i])
        
        title_suffix = f" (Excluding: {', '.join(exclude_models)})" if exclude_models else ""
        
        ax.set_xlabel('Radius (Å)')
        ax.set_ylabel('MAE Standard Deviation (mV)')
        ax.set_title(f'Model Stability Across Radii{title_suffix}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_stability_analysis{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved model stability analysis plot to: {self.output_dir / f'model_stability_analysis{suffix}.png'}")

    def create_detailed_comparison_plot(self, exclude_models=None):
        """Create detailed comparison plots with model exclusion option"""
        if exclude_models is None:
            exclude_models = []
        suffix = "_filtered" if exclude_models else ""
        
        plot_data = []
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                if model_name not in exclude_models:
                    plot_data.append({
                        'Radius': int(radius),
                        'Model': model_name,
                        'MAE': metrics['MAE_mean'],
                        'MAE_std': metrics['MAE_std'],
                        'R2': metrics['R2_mean'],
                        'R2_std': metrics['R2_std']
                    })
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            print("No data available for plot after filtering. Skipping.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        title = 'Comprehensive Model Performance Analysis'
        if exclude_models:
            title += f" (Excluding: {', '.join(exclude_models)})"
        fig.suptitle(title, fontsize=16)
        
        models = df['Model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model].sort_values('Radius')
            axes[0, 0].plot(model_data['Radius'], model_data['MAE'], marker='o', label=model, color=colors[i])
            axes[0, 0].fill_between(model_data['Radius'], model_data['MAE'] - model_data['MAE_std'],
                                  model_data['MAE'] + model_data['MAE_std'], alpha=0.2, color=colors[i])
            axes[0, 1].plot(model_data['Radius'], model_data['R2'], marker='s', label=model, color=colors[i])
            axes[0, 1].fill_between(model_data['Radius'], model_data['R2'] - model_data['R2_std'],
                                  model_data['R2'] + model_data['R2_std'], alpha=0.2, color=colors[i])
        
        axes[0, 0].set_xlabel('Radius (Å)')
        axes[0, 0].set_ylabel('MAE (mV)')
        axes[0, 0].set_title('Mean Absolute Error vs Radius')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('Radius (Å)')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score vs Radius')
        axes[0, 1].grid(True, alpha=0.3)
        
        best_radii = df.loc[df.groupby('Model')['MAE'].idxmin()]
        bars = axes[1, 0].bar(range(len(best_radii)), best_radii['Radius'])
        axes[1, 0].set_xticks(range(len(best_radii)))
        axes[1, 0].set_xticklabels(best_radii['Model'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Best Radius (Å)')
        axes[1, 0].set_title('Optimal Radius for Each Model')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', ha='center', va='bottom')
        
        mae_by_model = [df[df['Model'] == model]['MAE'].values for model in models]
        bp = axes[1, 1].boxplot(mae_by_model, labels=models, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_ylabel('MAE (mV)')
        axes[1, 1].set_title('MAE Distribution Across All Radii')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'detailed_analysis{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved detailed comparison plot to: {self.output_dir / f'detailed_analysis{suffix}.png'}")

    def generate_report(self):
        """Generate a comprehensive analysis report integrating all findings"""
        report_file = self.output_dir / 'analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write("Iron-Sulfur Cofactor Redox Potential Prediction - Analysis Report\n")
            f.write("="*70 + "\n\n")
            
            # Best models
            best_models = self.find_best_models(top_n=5)
            f.write("TOP 5 MODELS BY MAE:\n")
            f.write("-" * 30 + "\n")
            for idx, row in best_models.iterrows():
                f.write(f"{row['model']} (r={row['radius']}Å): MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                        f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}\n")
            f.write("\n")
            
            # Stability and trends
            f.write("PERFORMANCE TRENDS AND STABILITY:\n")
            f.write("-" * 30 + "\n")
            
            # Get all unique model names
            all_models = set()
            for radius_results in self.results.values():
                all_models.update(radius_results.keys())
            
            # Include all models in the report
            for model in sorted(all_models):
                f.write(f"{model}:\n")
                for radius in sorted(self.results.keys(), key=int):
                    if model in self.results[radius]:
                        f.write(f"  Radius {radius}Å: MAE={self.results[radius][model]['MAE_mean']:.2f} mV\n")
                f.write("\n")

        print(f"Analysis report saved to: {report_file}")

    def _get_top_features(self, importance_data):
        """Get the top features based on average importance"""
        feature_avg_importance = {}
        
        for f, d in importance_data.items():
            # Make sure we calculate the average only if we have data points
            if d['importance']:
                feature_avg_importance[f] = np.mean(d['importance'])
        
        return sorted(feature_avg_importance, key=feature_avg_importance.get, reverse=True)

class OutputTee:
    """Redirect output to both console and file"""
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream
    def write(self, message): self.console.write(message); self.file.write(message); self.file.flush()
    def flush(self): self.console.flush(); self.file.flush()

def compare_with_literature(results_dir: str, output_dir: str, literature_mae: float = None):
    """Compare results with literature values"""
    analyzer = ResultAnalyzer(results_dir)
    best_models = analyzer.find_best_models(top_n=3)
    
    print("\nComparison with Literature:")
    print("-" * 40)
    if literature_mae:
        best_mae = best_models.iloc[0]['mae_mean']
        improvement = ((literature_mae - best_mae) / literature_mae) * 100
        print(f"Literature: {literature_mae:.1f} mV, Ours: {best_mae:.1f} mV")
        print(f"Improvement: {improvement:.1f}%" if improvement > 0 else f"Worse by: {abs(improvement):.1f}%")
    flavoprotein_mae = 36.4
    best_mae = best_models.iloc[0]['mae_mean']
    comparison = best_mae / flavoprotein_mae
    print(f"\nGaluzzi et al. (flavoproteins): {flavoprotein_mae:.1f} mV, Ours: {best_mae:.1f} mV")
    print(f"Ours is {(1-comparison)*100:.1f}% better" if comparison < 1 else f"Flavoproteins better by {(comparison-1)*100:.1f}%")
    sys.stdout = analyzer.original_stdout
    analyzer.log_stream.close()

def main():
    """Main function for enhanced analysis"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ML training results')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing ML results')
    parser.add_argument('--literature_mae', type=float, default=None, help='Literature MAE value')
    parser.add_argument('--exclude_models', type=str, nargs='+', default=['LinearRegression'], help='Models to exclude')
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.results_dir)
    print("Analyzing ML results...")
    analyzer.find_best_models()
    
    # Create radius trend plots
    analyzer.analyze_radius_trends([])
    analyzer.analyze_radius_trends(args.exclude_models)
    
    # Generate stability plots
    analyzer.analyze_model_stability([])
    analyzer.analyze_model_stability(args.exclude_models)
    
    # Create detailed comparison plots
    analyzer.create_detailed_comparison_plot([])
    analyzer.create_detailed_comparison_plot(args.exclude_models)
    
    analyzer.generate_report()
    
    if args.literature_mae:
        compare_with_literature(args.results_dir, args.results_dir + '_interpretation', args.literature_mae)
    
    for file in ['ml_results.json', 'hyperparameters_*.json']:
        for src in Path(args.results_dir).glob(file):
            dst = analyzer.output_dir / src.name
            with open(src, 'r') as s, open(dst, 'w') as d:
                d.write(s.read())
            print(f"Copied {src.name} to {analyzer.output_dir}")
    print(f"Interpretation complete. Results in: {analyzer.output_dir}")
    sys.stdout = analyzer.original_stdout
    analyzer.log_stream.close()

if __name__ == "__main__":
    main()