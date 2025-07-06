#!/usr/bin/env python3
"""
Create Feature Subsets
---------------------
Organizes feature datasets into specific subsets based on:
- Cofactor type (all, 2Fe, 4Fe)
- Feature scope (all, protein-wide, bar-region)

Will create a directory structure like:
    feature_datasets/
    ├── complete_dataset/
    ├── all_cofactors_protein/
    ├── all_cofactors_bar/
    ├── FES_all/
    ├── FES_protein/
    ├── FES_bar/
    ├── SF4_all/
    ├── SF4_protein/
    └── SF4_bar/

    
Usage: 
    cd final_stretch/feature_extraction
    python create_feature_subsets.py \
        --input_dir output/merged_with_ph_em \
        --output_dir feature_subsets
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
import shutil
import glob

class FeatureSubsetCreator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.base_output_dir = Path(output_dir)
        
        # Define dataset configurations
        self.dataset_configs = {
            'complete_dataset': {
                'cofactor_filter': None,
                'feature_type': 'all'
            },
            'all_cofactors_protein': {
                'cofactor_filter': None,
                'feature_type': 'protein'
            },
            'all_cofactors_bar': {
                'cofactor_filter': None,
                'feature_type': 'bar'
            },
            'FES_all': {
                'cofactor_filter': 'FES',
                'feature_type': 'all'
            },
            'FES_protein': {
                'cofactor_filter': 'FES',
                'feature_type': 'protein'
            },
            'FES_bar': {
                'cofactor_filter': 'FES',
                'feature_type': 'bar'
            },
            'SF4_all': {
                'cofactor_filter': 'SF4',
                'feature_type': 'all'
            },
            'SF4_protein': {
                'cofactor_filter': 'SF4',
                'feature_type': 'protein'
            },
            'SF4_bar': {
                'cofactor_filter': 'SF4',
                'feature_type': 'bar'
            }
        }
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_feature_columns(self, df: pd.DataFrame, feature_type: str) -> List[str]:
        """Get columns based on feature type."""
        # Always include these columns but will reorder later
        id_cols = ['cofactor_id']
        end_cols = ['pH', 'Em']
        struct_cols = [col for col in df.columns if col.startswith('struct.')]
        seq_cols = [col for col in df.columns if col.startswith(('seq.'))]
        
        if feature_type == 'all':
            feature_cols = [col for col in df.columns 
                        if col not in id_cols + end_cols]
        elif feature_type == 'protein':
            protein_cols = [col for col in df.columns if col.startswith('Protein.')]
            feature_cols = struct_cols + protein_cols + seq_cols
        elif feature_type == 'bar':
            bar_cols = [col for col in df.columns if col.startswith('Bar.')]
            feature_cols = struct_cols + bar_cols
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Return columns in desired order
        return id_cols + feature_cols + end_cols
    
    def create_subsets(self):
        """Create all feature subsets."""
        # First, create the base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each dataset configuration
        for dataset_name, config in self.dataset_configs.items():
            output_dir = self.base_output_dir / dataset_name
            output_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Creating dataset: {dataset_name}")
            
            # Process each radius file
            for file in self.input_dir.glob("features_r*_final_*_with_ph_em.csv"):
                # Read CSV
                df = pd.read_csv(file)
                
                # Apply cofactor filter if specified
                if config['cofactor_filter']:
                    # Extract cofactor type from cofactor_id without adding new column
                    mask = df['cofactor_id'].str.extract(r'_(SF4|FES)(?:_|$)')[0] == config['cofactor_filter']
                    df = df[mask]
                    if len(df) == 0:
                        self.logger.warning(f"No {config['cofactor_filter']} cofactors found in {file.name}")
                        continue
                
                # Extract radius from filename
                radius = int(file.name.split('_')[1][1:])

                # For complete dataset, copy the file with a new name
                if dataset_name == 'complete_dataset':
                    new_name = f"features_r{radius}_all.csv"
                    
                    # Copy with new name directly
                    shutil.copy2(file, output_dir / new_name)
                    self.logger.info(f"Created {new_name}")
                    continue
                
                # Select features
                feature_cols = self._get_feature_columns(df, config['feature_type'])
                df_subset = df[feature_cols]
                
                # Create output filename
                if config['feature_type'] == 'protein':
                    # For protein features, use simplified name that will overwrite
                    if config['cofactor_filter']:
                        output_name = f"features_{config['cofactor_filter']}_protein.csv"
                    else:
                        output_name = "features_all_protein.csv"
                else:
                    # For other feature types, keep the radius in the name
                    if config['cofactor_filter']:
                        output_name = f"features_r{radius}_{config['cofactor_filter']}"
                    else:
                        output_name = f"features_r{radius}_all"
                        
                    if config['feature_type'] != 'all':
                        output_name += f"_{config['feature_type']}"
                    
                    output_name += ".csv"
                
                # Save subset
                df_subset.to_csv(output_dir / output_name, index=False)
                self.logger.info(f"Created {output_name} with {len(df_subset)} rows")


    def cleanup_complete_dataset(self):
        """Remove original files with _with_ph_em.csv extension from complete_dataset folder."""
        complete_dataset_dir = self.base_output_dir / 'complete_dataset'
        if not complete_dataset_dir.exists():
            self.logger.warning("Complete dataset directory not found, skipping cleanup")
            return
            
        # Find and remove files with _with_ph_em.csv extension
        for file in complete_dataset_dir.glob("*_with_ph_em.csv"):
            file.unlink()
            self.logger.info(f"Removed original file: {file.name}")


def main():
    parser = argparse.ArgumentParser(description="Create feature dataset subsets")
    parser.add_argument("--input_dir", required=True, help="Directory containing merged feature CSVs")
    parser.add_argument("--output_dir", required=True, help="Base directory for feature datasets")
    args = parser.parse_args()
    
    creator = FeatureSubsetCreator(args.input_dir, args.output_dir)
    creator.create_subsets()
    creator.cleanup_complete_dataset()

if __name__ == "__main__":
    main()