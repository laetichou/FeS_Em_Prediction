#!/usr/bin/env python3
"""
Enhanced utility functions for analyzing ML results for radius-independent datasets

Run with: python interpretation_protein.py --results_dir path/to/output --literature_mae 36.4
python interpretation_protein.py --literature_mae 36.4 --exclude_models LinearRegression --results_dir path/to/output
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
    """Analyze and visualize ML training results for radius-independent datasets"""
    
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
        if not results or not isinstance(results, dict):
            raise ValueError(f"Invalid results format in {results_file}")
        
        # For radius-independent datasets, wrap results in a dummy radius
        # to maintain compatibility with some methods
        return {"0": results}

    def find_best_models(self, metric='MAE_mean', top_n=5):
        """Find the top N best models for radius-independent datasets"""
        model_performance = []
        
        # In radius-independent datasets, we only have one "radius" (dummy)
        radius = next(iter(self.results))
        radius_results = self.results[radius]
        
        if not radius_results:
            print("No model results found!")
            return pd.DataFrame()
        
        sample_model = next(iter(radius_results))
        print(f"Available metrics for {sample_model}:")
        for k, v in radius_results[sample_model].items():
            print(f"  {k}: {v}")
        
        if metric not in radius_results[sample_model]:
            available = radius_results[sample_model].keys()
            print(f"Warning: Metric '{metric}' not found. Available: {', '.join(available)}")
            metric = next((m for m in ['MAE_mean', 'R2_mean'] if m in available), list(available)[0])
            print(f"Using fallback metric: {metric}")
        
        for model_name, metrics in radius_results.items():
            try:
                model_performance.append({
                    'model': model_name,
                    'metric_value': metrics[metric],
                    'mae_mean': metrics.get('MAE_mean', 0),
                    'mae_std': metrics.get('MAE_std', 0),
                    'r2_mean': metrics.get('R2_mean', 0),
                    'r2_std': metrics.get('R2_std', 0)
                })
            except KeyError as e:
                print(f"Error for model {model_name}: {e}")
        
        df = pd.DataFrame(model_performance)
        ascending = True if 'MAE' in metric or 'RMSE' in metric else False
        best_models = df.nsmallest(top_n, 'metric_value') if ascending else df.nlargest(top_n, 'metric_value')
        
        print(f"\nTop {top_n} models by {metric}:")
        print("="*60)
        for idx, row in best_models.iterrows():
            print(f"{row['model']}: "
                  f"MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                  f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}")
        return best_models

    def create_performance_comparison_plot(self, exclude_models=None):
        """Create performance comparison plots with model exclusion option"""
        if exclude_models is None:
            exclude_models = []
        suffix = "_filtered" if exclude_models else ""
        
        radius = next(iter(self.results))
        models_metrics = self.results[radius]
        
        # Filter out excluded models
        models_metrics = {name: metrics for name, metrics in models_metrics.items() 
                        if name not in exclude_models}
        
        if not models_metrics:
            print("No models remaining after filtering. Skipping comparison plot.")
            return
        
        # Sort models by MAE performance
        sorted_models = sorted(models_metrics.keys(), 
                            key=lambda x: models_metrics[x]['MAE_mean'])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data for plotting
        model_names = []
        mae_values = []
        mae_errors = []
        r2_values = []
        r2_errors = []
        
        for model in sorted_models:
            metrics = models_metrics[model]
            model_names.append(model)
            mae_values.append(metrics['MAE_mean'])
            mae_errors.append(metrics['MAE_std'])
            r2_values.append(metrics['R2_mean'])
            r2_errors.append(metrics['R2_std'])
        
        # Create MAE bar chart
        y_pos = np.arange(len(model_names))
        ax1.barh(y_pos, mae_values, xerr=mae_errors, align='center', 
                capsize=5, color='skyblue', ecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_names)
        ax1.invert_yaxis()  # Labels read top-to-bottom
        ax1.set_xlabel('Mean Absolute Error (mV)')
        ax1.set_title('MAE Comparison Across Models')
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values to bars - move text above the bars with offset
        for i, v in enumerate(mae_values):
            # Position the text 10 units above each bar, shifting it higher
            text_x = v + mae_errors[i] + 10  # Position after the error bar cap
            ax1.text(text_x, i, f"{v:.2f}±{mae_errors[i]:.2f}", 
                    va='center', fontsize=9)
        
        # Create R² bar chart
        ax2.barh(y_pos, r2_values, xerr=r2_errors, align='center', 
                capsize=5, color='lightgreen', ecolor='black')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(model_names)
        ax2.invert_yaxis()  # Labels read top-to-bottom
        ax2.set_xlabel('R² Score')
        ax2.set_title('R² Comparison Across Models')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values to bars - move text above the bars with offset
        for i, v in enumerate(r2_values):
            # Position the text after the error bar with sufficient spacing
            text_x = v + r2_errors[i] + 0.1  # Position after the error bar cap
            ax2.text(text_x, i, f"{v:.3f}±{r2_errors[i]:.3f}", 
                    va='center', fontsize=9)
        
        # Adjust x-axis limits to make room for labels
        ax1_xlim = ax1.get_xlim()
        ax1.set_xlim(ax1_xlim[0], ax1_xlim[1] * 1.2)  # Extend x-axis by 20%
        
        ax2_xlim = ax2.get_xlim()
        ax2.set_xlim(ax2_xlim[0], min(ax2_xlim[1] * 1.2, 1.2))  # Extend x-axis by 20% but cap at 1.2
        
        plt.tight_layout()
        title = "Model Performance Comparison"
        if exclude_models:
            title += f" (Excluding: {', '.join(exclude_models)})"
        fig.suptitle(title, fontsize=16, y=1.05)
        
        plt.savefig(self.output_dir / f'model_performance{suffix}.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved model performance comparison to: {self.output_dir / f'model_performance{suffix}.png'}")

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
                f.write(f"{row['model']}: MAE={row['mae_mean']:.2f}±{row['mae_std']:.2f} mV, "
                        f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}\n")
            f.write("\n")
            
            # Full model comparison
            f.write("ALL MODELS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            
            # Get all models
            radius = next(iter(self.results))
            all_models = self.results[radius]
            
            # Sort models by MAE
            sorted_models = sorted(all_models.keys(), 
                                  key=lambda x: all_models[x]['MAE_mean'])
            
            for model in sorted_models:
                metrics = all_models[model]
                f.write(f"{model}:\n")
                f.write(f"  MAE: {metrics['MAE_mean']:.2f}±{metrics['MAE_std']:.2f} mV\n")
                f.write(f"  R²: {metrics['R2_mean']:.3f}±{metrics['R2_std']:.3f}\n")
                if 'Spearman_mean' in metrics:
                    f.write(f"  Spearman: {metrics['Spearman_mean']:.3f}±{metrics['Spearman_std']:.3f}\n")
                if 'Pearson_mean' in metrics:
                    f.write(f"  Pearson: {metrics['Pearson_mean']:.3f}±{metrics['Pearson_std']:.3f}\n")
                f.write("\n")

        print(f"Analysis report saved to: {report_file}")

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
    best_models = analyzer.find_best_models(top_n=1)
    
    if best_models.empty:
        print("No models available for literature comparison")
        return
        
    best_mae = best_models.iloc[0]['mae_mean']
    
    print("\nComparison with Literature:")
    print("-" * 40)
    if literature_mae:
        improvement = ((literature_mae - best_mae) / literature_mae) * 100
        print(f"Literature: {literature_mae:.1f} mV, Ours: {best_mae:.1f} mV")
        print(f"Improvement: {improvement:.1f}%" if improvement > 0 else f"Worse by: {abs(improvement):.1f}%")
    
    flavoprotein_mae = 36.4
    comparison = best_mae / flavoprotein_mae
    print(f"\nGaluzzi et al. (flavoproteins): {flavoprotein_mae:.1f} mV, Ours: {best_mae:.1f} mV")
    print(f"Ours is {(1-comparison)*100:.1f}% better" if comparison < 1 else f"Flavoproteins better by {(comparison-1)*100:.1f}%")

def main():
    """Main function for radius-independent analysis"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze ML training results for radius-independent datasets')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing ML results')
    parser.add_argument('--literature_mae', type=float, default=None, help='Literature MAE value')
    parser.add_argument('--exclude_models', type=str, nargs='+', default=['LinearRegression'], help='Models to exclude')
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.results_dir)
    print("Analyzing ML results...")
    analyzer.find_best_models()
    
    # Generate performance comparison plots
    analyzer.create_performance_comparison_plot([])
    analyzer.create_performance_comparison_plot(args.exclude_models)
    
    # Generate report
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