#!/usr/bin/env python3
"""
Machine Learning Training Script for Iron-Sulfur Cofactor Redox Potential Prediction
Based on Galuzzi et al. methodology for flavoproteins, adapted for iron-sulfur clusters.

This script performs:
1. Data loading and preprocessing for multiple radii (1-16 Å)
2. Nested cross-validation for hyperparameter optimization and model selection
3. Training of 6 different ML models
4. SHAP analysis for feature importance
5. Comprehensive results analysis and visualization

Usage: python ml_training.py --data_dir /path/to/features --output_dir /path/to/output
"""

import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# ML libraries
from sklearn.model_selection import (
    KFold, GridSearchCV, cross_val_score, 
    train_test_split, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    r2_score, explained_variance_score
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for model interpretation
try:
    import shap
    assert shap.__version__ >= "0.41.0"
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature importance analysis will be limited.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set up logging
def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class IronSulfurMLTrainer:
    """Main class for training ML models on iron-sulfur cofactor data"""
    
    def __init__(self, data_dir: str, output_dir: str, target_column: str = "redox_potential"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_column = target_column
        
        # Available radii
        self.radii = list(range(1,17))
        
        # ML models configuration
        self.models = self._setup_models()
        
        # Results storage
        self.results = {}
        self.best_models = {}
        
        # Add dictionary to store grid search parameters
        self.grid_search_params = {}
    
        
        # Setup logger
        self.logger = setup_logging(str(self.output_dir))
    
    def _create_pipeline(self, model_config: Dict, n_features: int = None) -> Pipeline:
        """Create a sklearn pipeline based on model configuration"""
        pipeline_steps = []
        
        # Add scaling if needed
        if model_config['needs_scaling']:
            pipeline_steps.append(('scaler', StandardScaler()))
        
        # Add feature selection if needed
        if model_config['needs_feature_selection'] and n_features is not None:
            k_features = min(50, n_features // 2)  # Adaptive feature selection
            pipeline_steps.append(('selector', SelectKBest(f_regression, k=k_features)))
        
        # Add the model
        pipeline_steps.append(('model', model_config['model']))
        
        return Pipeline(pipeline_steps)
        
    def _setup_models(self) -> Dict:
        """Setup ML models with their hyperparameter grids"""
        
        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},  # No hyperparameters for basic LR
                'needs_scaling': True,
                'needs_feature_selection': True
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                },
                'needs_scaling': True,
                'needs_feature_selection': True
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf'],  # dropped 'poly' for less complexity
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'needs_scaling': True,
                'needs_feature_selection': True
            },
            'GaussianProcessRegressor': {
                'model': GaussianProcessRegressor(random_state=42, n_restarts_optimizer=3),
                'params': {
                    'kernel': [
                        ConstantKernel() * RBF() + WhiteKernel(),
                        ConstantKernel() * RBF(),
                        RBF() + WhiteKernel()
                    ],
                    'alpha': [1e-10, 1e-8, 1e-6, 1e-4]
                },
                'needs_scaling': True,
                'needs_feature_selection': True
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']  # could keep only 'minkowski' for generalization
                },
                'needs_scaling': True,
                'needs_feature_selection': True
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'needs_scaling': False,
                'needs_feature_selection': False
            },
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 1.0]
                },
                'needs_scaling': False,
                'needs_feature_selection': False
            }
        }
        
        return models
    
    def load_data(self, radius: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Load feature data for a specific radius"""
        
        # Look for feature files with the radius in the name
        feature_files = list(self.data_dir.glob(f"*features*r{radius}*.csv"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found for radius {radius} in {self.data_dir}")
        
        # Use the most recent file if multiple exist
        feature_file = sorted(feature_files)[-1]
        self.logger.info(f"Loading data from {feature_file}")
        
        df = pd.read_csv(feature_file, index_col=0)
        
        # Check if target column exists
        if self.target_column not in df.columns:
            # Try common alternative names
            target_alternatives = ['redox_potential', 'Em', 'E_m', 'midpoint_potential']
            for alt in target_alternatives:
                if alt in df.columns:
                    self.target_column = alt
                    break
            else:
                raise ValueError(f"Target column '{self.target_column}' not found. Available columns: {df.columns.tolist()}")
        
        # Separate features and target
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        
        # Remove any remaining non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features for radius {radius}Å")
        self.logger.info(f"Target range: {y.min():.2f} to {y.max():.2f} mV")
        
        return X, y
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, 
                       needs_scaling: bool = True, 
                       needs_feature_selection: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the data"""
        
        # Remove features with zero variance
        variance_selector = VarianceThreshold(threshold=0)
        X_var = pd.DataFrame(
            variance_selector.fit_transform(X),
            columns=X.columns[variance_selector.get_support()],
            index=X.index
        )
        
        self.logger.info(f"Removed {X.shape[1] - X_var.shape[1]} zero-variance features")
        
        # Remove highly correlated features
        correlation_matrix = X_var.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        
        X_processed = X_var.drop(columns=high_corr_features)
        self.logger.info(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Handle missing values
        if X_processed.isnull().sum().sum() > 0:
            self.logger.warning("Found missing values, filling with median")
            X_processed = X_processed.fillna(X_processed.median())
        
        self.logger.info(f"Final feature set: {X_processed.shape[1]} features")
        
        return X_processed, y
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        pearson_corr, _ = pearsonr(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Spearman': spearman_corr,
            'Pearson': pearson_corr,
            'ExplainedVariance': explained_var
        }

    def plot_learning_curves(self, X: pd.DataFrame, y: pd.Series, model_name: str, radius: int):
            """Generate and plot learning curves for model validation."""
            from sklearn.model_selection import learning_curve
            
            model_config = self.models[model_name]
            pipeline = self._create_pipeline(model_config, n_features=X.shape[1])
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline, X, y,
                train_sizes=train_sizes,
                cv=5, n_jobs=-1,
                scoring='neg_mean_absolute_error'
            )
            
            train_mean = -np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = -np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training score', color='blue')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Examples')
            plt.ylabel('Mean Absolute Error (mV)')
            plt.title(f'Learning Curves - {model_name} (r={radius}Å)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(self.output_dir / f'learning_curve_{model_name}_r{radius}.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, radius: int):
        """Create comprehensive prediction analysis plots."""
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Scatter plot of predicted vs actual
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', label='Perfect Prediction')
        z = 1.96  # 95% confidence interval
        mae = mean_absolute_error(y_true, y_pred)
        ax1.fill_between(
            [y_true.min(), y_true.max()],
            [y_true.min() - z*mae, y_true.max() - z*mae],
            [y_true.min() + z*mae, y_true.max() + z*mae],
            alpha=0.2, color='gray', label='95% CI'
        )
        ax1.set_xlabel('Actual Em (mV)')
        ax1.set_ylabel('Predicted Em (mV)')
        ax1.set_title('Predicted vs Actual')
        ax1.legend()
        
        # 2. Error distribution
        errors = y_pred - y_true
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Prediction Error (mV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        
        # 3. Residual plot
        ax3.scatter(y_pred, errors, alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predicted Em (mV)')
        ax3.set_ylabel('Residual (mV)')
        ax3.set_title('Residual Plot')
        
        plt.suptitle(f'Prediction Analysis - {model_name} (r={radius}Å)')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f'prediction_analysis_{model_name}_r{radius}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_model_radius(self, radius: int, n_splits_outer: int = 5, 
                        n_splits_inner: int = 3, n_repeats: int = 10) -> Dict:
        """Train all models for a specific radius using nested CV"""
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Training models for radius {radius}Å")
        self.logger.info(f"{'='*50}")
        
        # Load data
        X, y = self.load_data(radius)
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y)
        
        radius_results = {}

        for model_name, model_config in self.models.items():
            self.logger.info(f"\nTraining {model_name}...")

            # Store grid search parameters
            self.grid_search_params[f"{model_name}_r{radius}"] = model_config['params']
            
            all_true = []
            all_pred = []
            all_metrics = {
                'MAE': [], 'RMSE': [], 'R2': [], 
                'Spearman': [], 'Pearson': [], 'ExplainedVariance': []
            }
            
            # Store individual fold metrics (like Em Predict script)
            fold_metrics = {
                'mae_train': [],
                'mae_test': [],
                'RMSE': [],
                'R2': [],
                'Pearson': [],
                'Spearman': [],
                'ExplainedVariance': []
            }

            for repeat in range(n_repeats):
                # Outer CV for model evaluation
                outer_cv = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42 + repeat)
                
                # Important: Preprocess data inside each fold to prevent data leakage
                X_processed, y_processed = self.preprocess_data(X, y)
                
                for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_processed)):
                    X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
                    y_train, y_test = y_processed.iloc[train_idx], y_processed.iloc[test_idx]
                    
                    # Create pipeline
                    pipeline_steps = []
                    
                    # Add scaling if needed
                    if model_config['needs_scaling']:
                        pipeline_steps.append(('scaler', StandardScaler()))
                    
                    # Add feature selection if needed
                    if model_config['needs_feature_selection']:
                        # Select top features based on univariate statistical tests
                        k_features = min(50, X_train.shape[1] // 2)  # Adaptive feature selection
                        pipeline_steps.append(('selector', SelectKBest(f_regression, k=k_features)))
                    
                    # Add the model
                    pipeline_steps.append(('model', model_config['model']))
                    
                    pipeline = Pipeline(pipeline_steps)
                    
                    # Inner CV for hyperparameter optimization
                    if model_config['params']:
                        # Adjust parameter names for pipeline
                        param_grid = {}
                        for param, values in model_config['params'].items():
                            param_grid[f'model__{param}'] = values
                        
                        inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
                        
                        grid_search = GridSearchCV(
                            pipeline, param_grid, cv=inner_cv,
                            scoring='neg_mean_absolute_error', n_jobs=-1,
                            verbose=0
                        )
                        
                        grid_search.fit(X_train, y_train)
                        best_model = grid_search.best_estimator_
                        
                        # Log best parameters for debugging
                        if repeat == 0 and fold == 0:
                            self.logger.info(f"Best parameters: {grid_search.best_params_}")
                        
                        best_model_path = self.output_dir / f"best_models" / f"{model_name}_r{radius}.pkl"
                        best_model_path.parent.mkdir(exist_ok=True)

                        # Save model and its parameters
                        model_info = {
                            'model': grid_search.best_estimator_,
                            'best_params': grid_search.best_params_,
                            'cv_results': grid_search.cv_results_,
                            'grid_search_params': model_config['params']
                        }
                        
                        with open(best_model_path, 'wb') as f:
                            pickle.dump(model_info, f)
                        
                    else:
                        # No hyperparameters to tune
                        best_model = pipeline
                        best_model.fit(X_train, y_train)

                        # Save basic model info even without hyperparameters
                        best_model_path = self.output_dir / "best_models" / f"{model_name}_r{radius}.pkl"
                        best_model_path.parent.mkdir(exist_ok=True)
                        model_info = {
                            'model': best_model,
                            'best_params': {},
                            'cv_results': None,
                            'grid_search_params': {}
                        }
                        with open(best_model_path, 'wb') as f:
                            pickle.dump(model_info, f)
                    
                    # Predict on training and test sets
                    y_train_pred = best_model.predict(X_train)
                    y_test_pred = best_model.predict(X_test)

                    # Store predictions for overall analysis
                    all_true.extend(y_test.tolist())
                    all_pred.extend(y_test_pred.tolist())
                    
                    # Calculate metrics for this fold
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_pearson = pearsonr(y_test, y_test_pred)[0]
                    test_spearman = spearmanr(y_test, y_test_pred)[0]
                    test_explained_var = explained_variance_score(y_test, y_test_pred)
                    
                    # Store individual fold metrics (like Em Predict script)
                    fold_metrics['mae_train'].append(train_mae)
                    fold_metrics['mae_test'].append(test_mae)
                    fold_metrics['RMSE'].append(test_rmse)
                    fold_metrics['R2'].append(test_r2)
                    fold_metrics['Pearson'].append(test_pearson)
                    fold_metrics['Spearman'].append(test_spearman)
                    fold_metrics['ExplainedVariance'].append(test_explained_var)
                    
                    # Evaluate and store metrics for aggregation
                    metrics = self.evaluate_model(y_test, y_test_pred)
                    
                    # Debug log for R2 values
                    self.logger.debug(f"Repeat {repeat}, Fold {fold}, R2: {metrics['R2']:.4f}")
                    
                    # Collect metrics for each fold
                    for metric_name, metric_value in metrics.items():
                        all_metrics[metric_name].append(metric_value)

            # Generate visualizations after all folds/repeats
            self.plot_learning_curves(X_processed, y_processed, model_name, radius)
            self.plot_prediction_analysis(np.array(all_true), np.array(all_pred), 
                                    model_name, radius)
                        
            # Calculate statistics across all folds and repeats
            final_metrics = {}
            for metric_name, metric_values in all_metrics.items():
                final_metrics[f'{metric_name}_mean'] = np.mean(metric_values)
                final_metrics[f'{metric_name}_std'] = np.std(metric_values)
                
                # Add more detail for R2 to help debugging
                if metric_name == 'R2':
                    self.logger.info(f"R2 values: min={np.min(metric_values):.4f}, max={np.max(metric_values):.4f}, median={np.median(metric_values):.4f}")
            
            # Add individual fold metrics lists (like Em Predict script)
            final_metrics['mae_train_list'] = fold_metrics['mae_train']
            final_metrics['mae_test_list'] = fold_metrics['mae_test']
            final_metrics['RMSE_list'] = fold_metrics['RMSE']
            final_metrics['R2_list'] = fold_metrics['R2']
            final_metrics['Pearson_list'] = fold_metrics['Pearson']
            final_metrics['Spearman_list'] = fold_metrics['Spearman']
            final_metrics['ExplainedVariance_list'] = fold_metrics['ExplainedVariance']
            
            radius_results[model_name] = final_metrics
            
            self.logger.info(f"{model_name} - MAE: {final_metrics['MAE_mean']:.2f} ± {final_metrics['MAE_std']:.2f} mV, R2: {final_metrics['R2_mean']:.4f} ± {final_metrics['R2_std']:.4f}")
        
        return radius_results
    
    def train_best_model_for_shap(self, radius: int, model_name: str) -> Tuple[Any, pd.DataFrame, pd.Series, List[str]]:
        """Train the best model on full dataset for SHAP analysis"""
        
        # Load and preprocess data
        X, y = self.load_data(radius)
        X_processed, y_processed = self.preprocess_data(X, y)
        
        model_config = self.models[model_name]
        
        # Create pipeline
        pipeline_steps = []
        
        if model_config['needs_scaling']:
            pipeline_steps.append(('scaler', StandardScaler()))
        
        if model_config['needs_feature_selection']:
            k_features = min(50, X_processed.shape[1] // 2)
            pipeline_steps.append(('selector', SelectKBest(f_regression, k=k_features)))
        
        pipeline_steps.append(('model', model_config['model']))
        
        pipeline = Pipeline(pipeline_steps)
        
        # Hyperparameter optimization on full dataset
        if model_config['params']:
            param_grid = {}
            for param, values in model_config['params'].items():
                param_grid[f'model__{param}'] = values
            
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv,
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            
            grid_search.fit(X_processed, y_processed)
            best_model = grid_search.best_estimator_

            # Save all grid search results to CSV
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df.to_csv(self.output_dir / f'grid_search_results_{model_name}_r{radius}.csv', index=False)
                        
        else:
            best_model = pipeline
            best_model.fit(X_processed, y_processed)
        
        # Get feature names after preprocessing
        if hasattr(best_model, 'named_steps') and 'selector' in best_model.named_steps:
            selector = best_model.named_steps['selector']
            selected_features = X_processed.columns[selector.get_support()].tolist()
        else:
            selected_features = X_processed.columns.tolist()
        
        return best_model, X_processed, y_processed, selected_features
    
    def perform_shap_analysis(self, radius: int, model_name: str, top_features: int = 20):
        """Perform SHAP analysis with separate analyses for feature types."""
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available, skipping feature importance analysis")
            return
        
        self.logger.info(f"Performing SHAP analysis for {model_name} at radius {radius}Å")
        
        try:
            # Train best model and get data
            best_model, X, y, feature_names = self.train_best_model_for_shap(radius, model_name)
            
            # Get actual model from pipeline
            if hasattr(best_model, 'named_steps'):
                actual_model = best_model.named_steps['model']
                # Transform data through pipeline except final model
                X_transformed = X.copy()
                for step_name, step in best_model.named_steps.items():
                    if step_name != 'model':
                        X_transformed = step.transform(X_transformed)
                
                if hasattr(X_transformed, 'toarray'):
                    X_transformed = X_transformed.toarray()
                
                if not isinstance(X_transformed, pd.DataFrame):
                    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
            else:
                actual_model = best_model
                X_transformed = X
            
            # Calculate SHAP values
            if model_name in ['RandomForestRegressor', 'GradientBoostingRegressor']:
                explainer = shap.TreeExplainer(actual_model)
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.KernelExplainer(actual_model.predict, X_transformed.iloc[:100])
                shap_values = explainer.shap_values(X_transformed.iloc[:100])
                X_transformed = X_transformed.iloc[:100]  # Match subset used for KernelExplainer
            
            # Separate features by type
            feature_groups = {
                'protein': ([f for f in feature_names if f.startswith('Protein.')], 
                        'Protein-wide Features'),
                'bar': ([f for f in feature_names if f.startswith('Bar.')],
                        'Bar Region Features'),
                'all': (feature_names, 'All Features Combined')
            }
            
            # Create plots for each feature group
            for group_name, (features, title) in feature_groups.items():
                if not features:
                    continue
                
                # Get feature indices
                feature_idx = [feature_names.index(f) for f in features]
                X_subset = X_transformed[features]
        
                if isinstance(shap_values, np.ndarray):
                    values_to_plot = shap_values[:, feature_idx]
                else:
                    values_to_plot = [shap_values[i] for i in feature_idx]
                
                # Create plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    values_to_plot,
                    X_transformed[features],
                    feature_names=features,
                    max_display=min(top_features, len(features)),
                    show=False
                )
                plt.title(f'{title}\n{model_name} (Radius {radius}Å)')
                plt.tight_layout()
                plt.savefig(
                    self.output_dir / f'shap_summary_{group_name}_{model_name}_r{radius}.png',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()
                
                # Save feature importance for this group
                feature_importance = pd.DataFrame({
                'feature': features,
                'importance': np.abs(shap_values[:, feature_idx]).mean(0)
                if isinstance(shap_values, np.ndarray)
                else np.abs(np.array([shap_values[i] for i in feature_idx])).mean(0)
            })
                
                feature_importance.to_csv(
                    self.output_dir / f'feature_importance_{group_name}_{model_name}_r{radius}.csv',
                    index=False
                )
            
            self.logger.info(f"SHAP analysis completed for {model_name} at radius {radius}Å")
            
        except Exception as e:
            self.logger.error(f"SHAP analysis failed for {model_name} at radius {radius}Å: {str(e)}")
            self.logger.error(f"Error details: {str(e)}", exc_info=True)
    
        
    def run_full_analysis(self):
        """Run the complete ML analysis pipeline"""
        
        self.logger.info("Starting comprehensive ML analysis for iron-sulfur cofactors")
        
        # Train models for all radii
        for radius in self.radii:
            self.results[radius] = self.train_model_radius(radius)
        
        # Find top 3 models overall based on MAE
        all_model_performances = []
        
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                all_model_performances.append({
                    'radius': radius,
                    'model_name': model_name,
                    'mae': metrics['MAE_mean'],
                    'metrics': metrics
                })
        
        # Sort by MAE (lower is better)
        all_model_performances.sort(key=lambda x: x['mae'])
        
        # Get top 3 models
        top_models = all_model_performances[:3]
        
        # Log best model
        best_model_info = top_models[0]
        self.logger.info(f"\nBest overall model: {best_model_info['model_name']} at radius {best_model_info['radius']}Å")
        self.logger.info(f"Best MAE: {best_model_info['mae']:.2f} mV")
        
        # Log all top 3 models
        self.logger.info("\nTop 3 performing models:")
        for i, model_info in enumerate(top_models, 1):
            self.logger.info(f"{i}. {model_info['model_name']} (r={model_info['radius']}Å) - MAE: {model_info['mae']:.2f} mV")
        
        # Perform SHAP analysis for top 3 models
        for model_info in top_models:
            radius = model_info['radius']
            model_name = model_info['model_name']
            self.logger.info(f"\nPerforming SHAP analysis for top model: {model_name} at radius {radius}Å")
            self.perform_shap_analysis(radius, model_name)
        
        # Save results - with error handling to ensure visualizations are still created
        try:
            self.save_results()
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error("Continuing with visualization generation...")
        
        # Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
        
        self.logger.info("Analysis completed!")
    
    def save_results(self):
        """Save all results and parameters to files"""
        
        # Save detailed results as JSON
        try:
            results_file = self.output_dir / 'ml_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, cls=CustomJSONEncoder)
        except IOError as e:
            self.logger.error(f"Failed to save results: {e}")
        
        # Create summary DataFrame with individual fold metrics
        summary_data = []
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                row = {
                    'radius': radius, 
                    'model': model_name,
                    'MAE_mean': metrics['MAE_mean'],
                    'MAE_std': metrics['MAE_std'],
                    'RMSE_mean': metrics['RMSE_mean'],
                    'RMSE_std': metrics['RMSE_std'],
                    'R2_mean': metrics['R2_mean'],
                    'R2_std': metrics['R2_std'],
                    'Pearson_mean': metrics['Pearson_mean'],
                    'Pearson_std': metrics['Pearson_std'],
                    'Spearman_mean': metrics['Spearman_mean'],
                    'Spearman_std': metrics['Spearman_std'],
                    'ExplainedVariance_mean': metrics['ExplainedVariance_mean'],
                    'ExplainedVariance_std': metrics['ExplainedVariance_std'],
                    # Add individual fold metrics (like Em Predict script)
                    'mae_train_list': metrics['mae_train_list'],
                    'mae_test_list': metrics['mae_test_list'],
                    'RMSE_list': metrics['RMSE_list'],
                    'R2_list': metrics['R2_list'],
                    'Pearson_list': metrics['Pearson_list'],
                    'Spearman_list': metrics['Spearman_list'],
                    'ExplainedVariance_list': metrics['ExplainedVariance_list']
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'ml_results_summary.csv', index=False)

        # Save grid search parameters
        params_file = self.output_dir / 'grid_search_parameters.json'
        with open(params_file, 'w') as f:
            json.dump(self.grid_search_params, f, indent=2, cls=CustomJSONEncoder)
        
        # Create summary of best parameters
        best_params_summary = {}
        best_models_dir = self.output_dir / "best_models"
        if best_models_dir.exists():
            for model_file in best_models_dir.glob("*.pkl"):
                with open(model_file, 'rb') as f:
                    model_info = pickle.load(f)
                    best_params_summary[model_file.stem] = {
                        'best_parameters': model_info['best_params'],
                        'grid_search_params': model_info['grid_search_params']
                    }
        
        # Save best parameters summary
        with open(self.output_dir / 'best_parameters_summary.json', 'w') as f:
            json.dump(best_params_summary, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")

    
    def create_visualizations(self):
        """Create visualization plots"""
        
        # Prepare data for plotting
        plot_data = []
        for radius, radius_results in self.results.items():
            for model_name, metrics in radius_results.items():
                plot_data.append({
                    'Radius': radius,
                    'Model': model_name,
                    'MAE': metrics['MAE_mean'],
                    'MAE_std': metrics['MAE_std'],
                    'RMSE': metrics['RMSE_mean'],
                    'RMSE_std': metrics['RMSE_std'],
                    'R2': metrics['R2_mean'],
                    'R2_std': metrics['R2_std'],
                    'Spearman': metrics['Spearman_mean'],
                    'Pearson': metrics['Pearson_mean'],
                    'ExplainedVariance': metrics['ExplainedVariance_mean']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # MAE vs Radius plot
        plt.figure(figsize=(14, 8))
        
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            plt.errorbar(model_data['Radius'], model_data['MAE'], 
                        yerr=model_data['MAE_std'], label=model, 
                        marker='o', capsize=5, capthick=2)
        
        plt.xlabel('Radius (Å)')
        plt.ylabel('Mean Absolute Error (mV)')
        plt.title('Model Performance vs Radius')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mae_vs_radius.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # R² vs Radius plot
        plt.figure(figsize=(14, 8))
        
        for model in plot_df['Model'].unique():
            model_data = plot_df[plot_df['Model'] == model]
            plt.errorbar(model_data['Radius'], model_data['R2'], 
                        yerr=model_data['R2_std'], label=model, 
                        marker='s', capsize=5, capthick=2)
        
        plt.xlabel('Radius (Å)')
        plt.ylabel('R² Score')
        plt.title('Model R² vs Radius')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'r2_vs_radius.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # # Heatmap of model performance (MAE)
        # pivot_mae = plot_df.pivot(index='Model', columns='Radius', values='MAE')
        
        # plt.figure(figsize=(12, 8))
        # sns.heatmap(pivot_mae, annot=True, fmt='.1f', cmap='viridis_r', 
        #            cbar_kws={'label': 'MAE (mV)'})
        # plt.title('Model Performance Heatmap (MAE)')
        # plt.tight_layout()
        # plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # Create heatmaps for different metrics
        metrics_to_plot = {
            'MAE': {'cmap': 'viridis_r', 'title': 'Mean Absolute Error (mV)', 'format': '.1f'},
            'RMSE': {'cmap': 'viridis_r', 'title': 'Root Mean Squared Error (mV)', 'format': '.1f'},
            'R2': {'cmap': 'viridis', 'title': 'R² Score', 'format': '.3f'},
            'Spearman': {'cmap': 'viridis', 'title': 'Spearman Correlation', 'format': '.3f'},
            'Pearson': {'cmap': 'viridis', 'title': 'Pearson Correlation', 'format': '.3f'},
            'ExplainedVariance': {'cmap': 'viridis', 'title': 'Explained Variance', 'format': '.3f'}
        }
        
        for metric, properties in metrics_to_plot.items():
            try:
                # Create pivot table for heatmap
                pivot_metric = plot_df.pivot(index='Model', columns='Radius', values=metric)
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_metric, annot=True, fmt=properties['format'], 
                        cmap=properties['cmap'], 
                        cbar_kws={'label': properties['title']})
                plt.title(f'Model Performance Heatmap - {properties["title"]}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'heatmap_{metric.lower()}.png', dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Created heatmap for {metric}")
            except Exception as e:
                self.logger.error(f"Failed to create heatmap for {metric}: {str(e)}")
        
        
        self.logger.info("Visualizations saved")

def main():
    """Main function to run the ML training pipeline"""
    
    parser = argparse.ArgumentParser(description='Train ML models for iron-sulfur redox potential prediction')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing feature CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--target_column', type=str, default='redox_potential',
                       help='Name of target column (default: redox_potential)')
    parser.add_argument('--n_repeats', type=int, default=10,
                       help='Number of repeated CV runs (default: 10)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = IronSulfurMLTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_column=args.target_column
    )
    
    # Run analysis
    try:
        trainer.run_full_analysis()
    except Exception as e:
        trainer.logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()