#!/usr/bin/env python3
"""
Statistical Analysis of Iron-Sulfur Protein Dataset
------------------------------------------------
Analyzes protein and cofactor characteristics from structure dataset and metadata.

Usage:
    cd BEP
    python Prediction_FeS_EM/final_stretch/dataset_analysis/dataset_statistics.py \
    --pdb_dir Prediction_FeS_EM/final_structure_dataset \
    --features Prediction_FeS_EM/final_stretch/feature_extraction/output/merged_with_ph_em/features_r1_final_20250601_202817_with_ph_em.csv \
    --metadata FeS_datasets/complete_FeS_dataset_with_cofactor_id.xlsx \
    --output_dir Prediction_FeS_EM/final_stretch/dataset_analysis/output
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import PDB
from Bio.PDB import *
from typing import Dict, List, Tuple
from scipy.spatial import ConvexHull

# Cofactor to PDB ID mapping for P00209
COFACTOR_TO_PDB_ID = {
    '[3Fe-4S]': 'F3S',
    '[4Fe-4S]': 'SF4',
    '[2Fe-2S]': 'FES',
    'Fe3+': 'FE'
}

def compute_cofactor_depth(structure, cofactor_atoms):
    """Compute burial depth as distance from cofactor barycenter to protein surface."""
    ca_coords = [
        atom.coord for model in structure
        for chain in model for residue in chain
        if is_aa(residue)
        for atom in residue if atom.get_id() == "CA"
    ]
    
    if len(ca_coords) < 4:
        return 0  # Not enough points for convex hull
        
    hull = ConvexHull(np.array(ca_coords))
    bar_center = np.mean([atom.coord for atom in cofactor_atoms if atom.element == "FE"], axis=0)
    
    distances = np.linalg.norm(np.array(ca_coords)[hull.vertices] - bar_center, axis=1)
    return float(min(distances))

def compute_protein_dimensions(structure):
    """Compute 3D size metrics of the protein structure."""
    coords = np.array([
        atom.coord 
        for model in structure
        for chain in model 
        for residue in chain 
        if is_aa(residue)
        for atom in residue
    ])
    
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    dimensions = max_coords - min_coords
    
    hull = ConvexHull(coords)
    
    return {
        'x_size': float(dimensions[0]),  # Å
        'y_size': float(dimensions[1]),  # Å
        'z_size': float(dimensions[2]),  # Å
        'max_dimension': float(np.max(dimensions)),  # Å
        'volume': float(hull.volume),  # Å³
        'surface_area': float(hull.area)  # Å²
    }

def convert_to_serializable(obj):
    """Convert object and its nested contents to JSON serializable types."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    return obj

class DatasetAnalyzer:
    def __init__(self, pdb_dir: str, features_file: str, metadata_file: str, output_dir: str = None):
        """Initialize dataset analyzer.
        
        Args:
            pdb_dir: Directory containing PDB files
            features_file: Path to features CSV file
            metadata_file: Path to metadata Excel file
            output_dir: Output directory for analysis results
        """
        self.pdb_dir = Path(pdb_dir)
        self.features_df = pd.read_csv(features_file)
        self.excel_df = pd.read_excel(metadata_file)
        
        self.metadata = pd.DataFrame()
        self.metadata['cofactor_id'] = self.features_df['cofactor_id']
        
        id_parts = self.metadata['cofactor_id'].str.split('_', expand=True)
        self.metadata['UniProt'] = id_parts[0]
        
        # Add cofactor information
        self.metadata['Cofactor'] = self.metadata['cofactor_id'].map(
            self.excel_df.set_index('cofactor_id')['Cofactor']
        )
        
        # Add protein family information
        self.metadata['Family'] = self.metadata['cofactor_id'].map(
            self.excel_df.set_index('cofactor_id')['Family']
        )
        
        self.metadata['Mutation_AF'] = self.metadata['cofactor_id'].map(
            self.excel_df.set_index('cofactor_id')['Mutation_AF']
        )
        
        self.metadata['burial_depth'] = self.features_df['struct.burial_depth']
        self.metadata['Em'] = self.features_df['Em']
        self.metadata['pH'] = self.features_df['pH']
        
        self.output_dir = Path(output_dir or "dataset_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.output_dir / 'analysis.log')
        
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Dataset analysis started")
        self.logger.info(f"Output directory: {self.output_dir}")
              
    def analyze_protein_sizes(self) -> Dict:
        """Analyze protein sizes from PDB files, with exception for P00209."""
        sizes = []
        dimensions_list = []
        parser = PDB.PDBParser(QUIET=True)
        
        for _, row in self.metadata.iterrows():
            uniprot = row['UniProt']
            mutation = row['Mutation_AF']
            cofactor = row['Cofactor']
            
            # Handle P00209 exception with cofactor mapping
            if uniprot == 'P00209':
                pdb_id = COFACTOR_TO_PDB_ID.get(cofactor, cofactor)  # Map cofactor to PDB ID
                pdb_file = self.pdb_dir / f"{uniprot}_{pdb_id}.pdb"
            else:
                # Use {UniProt}.pdb for wild-type (no mutation), otherwise {UniProt}_{Mutation}.pdb
                pdb_file = self.pdb_dir / f"{uniprot}.pdb" if mutation == 'no' else self.pdb_dir / f"{uniprot}_{mutation}.pdb"
            
            try:
                structure = parser.get_structure(uniprot, pdb_file)
                
                n_residues = sum(1 for residue in structure.get_residues() 
                                if is_aa(residue))
                sizes.append(n_residues)
                
                dimensions = compute_protein_dimensions(structure)
                dimensions_list.append(dimensions)
                
            except Exception as e:
                self.logger.warning(f"Error processing {pdb_file}: {e}")
        
        sequence_stats = {
            'mean_residues': np.mean(sizes),
            'median_residues': np.median(sizes),
            'std_residues': np.std(sizes),
            'min_residues': np.min(sizes),
            'max_residues': np.max(sizes),
            'total_proteins': len(sizes)
        }
        
        dimension_stats = {
            'mean_volume': np.mean([d['volume'] for d in dimensions_list]),
            'std_volume': np.std([d['volume'] for d in dimensions_list]),
            'mean_surface': np.mean([d['surface_area'] for d in dimensions_list]),
            'std_surface': np.std([d['surface_area'] for d in dimensions_list]),
            'mean_max_dimension': np.mean([d['max_dimension'] for d in dimensions_list]),
            'std_max_dimension': np.std([d['max_dimension'] for d in dimensions_list])
        }
        
        stats = {**sequence_stats, **dimension_stats}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.histplot(sizes, bins=30, ax=ax1)
        ax1.set_title('Distribution of Protein Sizes')
        ax1.set_xlabel('Number of Residues')
        ax1.set_ylabel('Count')
        
        sns.histplot([d['max_dimension'] for d in dimensions_list], bins=30, ax=ax2)
        ax2.set_title('Distribution of Protein Maximum Diameters')
        ax2.set_xlabel('Maximum Diameter (Å)')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'protein_sizes_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def analyze_cofactor_sizes(self) -> Dict:
        """Analyze residue counts and maximum diameter per cofactor type, with exception for P00209."""
        parser = PDB.PDBParser(QUIET=True)
        cofactor_sizes = {cofactor: {'residues': [], 'max_dimension': []} 
                         for cofactor in self.metadata['Cofactor'].unique()}
        
        for _, row in self.metadata.iterrows():
            uniprot = row['UniProt']
            mutation = row['Mutation_AF']
            cofactor = row['Cofactor']
            
            # Handle P00209 exception with cofactor mapping
            if uniprot == 'P00209':
                pdb_id = COFACTOR_TO_PDB_ID.get(cofactor, cofactor)  # Map cofactor to PDB ID
                pdb_file = self.pdb_dir / f"{uniprot}_{pdb_id}.pdb"
            else:
                # Use {UniProt}.pdb for wild-type (no mutation), otherwise {UniProt}_{Mutation}.pdb
                pdb_file = self.pdb_dir / f"{uniprot}.pdb" if mutation == 'no' else self.pdb_dir / f"{uniprot}_{mutation}.pdb"
            
            if not pdb_file.exists():
                self.logger.warning(f"PDB file not found for {uniprot}")
                continue
                
            try:
                structure = parser.get_structure(uniprot, pdb_file)
                n_residues = sum(1 for residue in structure.get_residues() 
                                if is_aa(residue))
                dimensions = compute_protein_dimensions(structure)
                
                cofactor_sizes[cofactor]['residues'].append(n_residues)
                cofactor_sizes[cofactor]['max_dimension'].append(dimensions['max_dimension'])
                
            except Exception as e:
                self.logger.warning(f"Error processing {pdb_file}: {e}")
        
        stats = []
        for cofactor, data in cofactor_sizes.items():
            residues = data['residues']
            max_dims = data['max_dimension']
            if residues:  # Only include if data exists
                stats.append({
                    'cofactor': cofactor,
                    'residue_count_mean': np.mean(residues),
                    'residue_count_median': np.median(residues),
                    'residue_count_std': np.std(residues),
                    'max_dimension_mean': np.mean(max_dims),
                    'max_dimension_median': np.median(max_dims),
                    'max_dimension_std': np.std(max_dims)
                })
        
        # Plot residue count distribution by cofactor type
        plt.figure(figsize=(10, 6))
        valid_data = self.metadata.merge(
            pd.DataFrame([
                {
                    'UniProt': row['UniProt'],
                    'Cofactor': row['Cofactor'],
                    'Mutation_AF': row['Mutation_AF'],
                    'residue_count': sum(1 for residue in PDB.PDBParser(QUIET=True).get_structure(
                        row['UniProt'], 
                        self.pdb_dir / f"{row['UniProt']}_{COFACTOR_TO_PDB_ID.get(row['Cofactor'], row['Cofactor']) if row['UniProt'] == 'P00209' else (row['Mutation_AF'] if row['Mutation_AF'] != 'no' else '')}.pdb"
                    ).get_residues() if is_aa(residue))
                }
                for _, row in self.metadata.iterrows() 
                if (self.pdb_dir / f"{row['UniProt']}_{COFACTOR_TO_PDB_ID.get(row['Cofactor'], row['Cofactor']) if row['UniProt'] == 'P00209' else (row['Mutation_AF'] if row['Mutation_AF'] != 'no' else '')}.pdb").exists()
            ]), on=['UniProt', 'Cofactor', 'Mutation_AF'], how='left'
        ).dropna(subset=['residue_count'])
        if len(valid_data) > 0:
            sns.boxplot(data=valid_data, x='Cofactor', y='residue_count')
            plt.title('Residue Count Distribution by Cofactor Type')
            plt.xticks(rotation=45)
            plt.ylabel('Number of Residues')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'residue_count_by_cofactor.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot maximum diameter distribution by cofactor type
        plt.figure(figsize=(10, 6))
        valid_data = self.metadata.merge(
            pd.DataFrame([
                {
                    'UniProt': row['UniProt'],
                    'Cofactor': row['Cofactor'],
                    'Mutation_AF': row['Mutation_AF'],
                    'max_dimension': compute_protein_dimensions(PDB.PDBParser(QUIET=True).get_structure(
                        row['UniProt'], 
                        self.pdb_dir / f"{row['UniProt']}_{COFACTOR_TO_PDB_ID.get(row['Cofactor'], row['Cofactor']) if row['UniProt'] == 'P00209' else (row['Mutation_AF'] if row['Mutation_AF'] != 'no' else '')}.pdb"
                    ))['max_dimension']
                }
                for _, row in self.metadata.iterrows() 
                if (self.pdb_dir / f"{row['UniProt']}_{COFACTOR_TO_PDB_ID.get(row['Cofactor'], row['Cofactor']) if row['UniProt'] == 'P00209' else (row['Mutation_AF'] if row['Mutation_AF'] != 'no' else '')}.pdb").exists()
            ]), on=['UniProt', 'Cofactor', 'Mutation_AF'], how='left'
        ).dropna(subset=['max_dimension'])
        if len(valid_data) > 0:
            sns.boxplot(data=valid_data, x='Cofactor', y='max_dimension')
            plt.title('Maximum Diameter Distribution by Cofactor Type')
            plt.xticks(rotation=45)
            plt.ylabel('Maximum Diameter (Å)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'max_dimension_by_cofactor.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {'cofactor_sizes': stats}
    
    def analyze_protein_detailed_stats(self) -> Dict:
        """Analyze residue count and maximum dimension per unique protein."""
        parser = PDB.PDBParser(QUIET=True)
        protein_stats = []
        
        # Iterate over unique UniProt IDs
        for uniprot in self.metadata['UniProt'].unique():
            # Prefer wild-type (Mutation == 'no') if available, else take first occurrence
            protein_rows = self.metadata[self.metadata['UniProt'] == uniprot]
            row = protein_rows[protein_rows['Mutation_AF'] == 'no'].iloc[0] if not protein_rows[protein_rows['Mutation_AF'] == 'no'].empty else protein_rows.iloc[0]
            
            mutation = row['Mutation_AF']
            cofactor = row['Cofactor']
            
            # Handle P00209 exception with cofactor mapping
            if uniprot == 'P00209':
                pdb_id = COFACTOR_TO_PDB_ID.get(cofactor, cofactor)
                pdb_file = self.pdb_dir / f"{uniprot}_{pdb_id}.pdb"
            else:
                # Use {UniProt}.pdb for wild-type, otherwise {UniProt}_{Mutation}.pdb
                pdb_file = self.pdb_dir / f"{uniprot}.pdb" if mutation == 'no' else self.pdb_dir / f"{uniprot}_{mutation}.pdb"
            
            if not pdb_file.exists():
                self.logger.warning(f"PDB file not found for {uniprot}: {pdb_file}")
                continue
                
            try:
                structure = parser.get_structure(uniprot, pdb_file)
                n_residues = sum(1 for residue in structure.get_residues() 
                                if is_aa(residue))
                max_dimension = compute_protein_dimensions(structure)['max_dimension']
                
                protein_stats.append({
                    'UniProt': uniprot,
                    'residue_count': n_residues,
                    'max_dimension': max_dimension
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing {pdb_file}: {e}")
        
        return {'protein_detailed_stats': protein_stats}
    
    def analyze_cofactors(self) -> Dict:
        """Analyze cofactor types and their properties."""
        cofactor_stats = {
            'total_cofactors': len(self.metadata),
            'counts': self.metadata['Cofactor'].value_counts().to_dict(),
            'proportions': self.metadata['Cofactor'].value_counts(normalize=True).to_dict(),
            'burial_depth': {
                'mean': self.metadata['burial_depth'].mean(),
                'median': self.metadata['burial_depth'].median(),
                'std': self.metadata['burial_depth'].std(),
                'min': self.metadata['burial_depth'].min(),
                'max': self.metadata['burial_depth'].max()
            }
        }
            
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.metadata, x='Cofactor')
        plt.title('Distribution of Cofactor Types')
        plt.xticks(rotation=45)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cofactor_dist.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        counts = list(cofactor_stats['counts'].values())
        cofactor_names = list(cofactor_stats['counts'].keys())
        total = sum(counts)
        labels = [f'{name}\n({count}, {(count/total)*100:.1f}%)' for name, count in zip(cofactor_names, counts)]
        
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=labels,
            autopct='',
            startangle=140,
            colors=plt.cm.Set3(range(len(counts)))
        )
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        plt.title('Cofactor Type Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'cofactor_pie_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        valid_data = self.metadata.dropna(subset=['burial_depth'])
        if len(valid_data) > 0:
            sns.boxplot(data=valid_data, x='Cofactor', y='burial_depth')
            plt.title('Burial Depth Distribution by Cofactor Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'burial_depth_by_cofactor.png', dpi=300, bbox_inches='tight')
        else:
            self.logger.warning("No valid burial depth data for boxplot")
        plt.close()
        
        return cofactor_stats

    def analyze_protein_families(self) -> Dict:
        """Analyze protein families, focusing on HiPIPs and Rieske proteins."""
        
        # Get unique proteins with their cofactor type and family
        unique_proteins = self.metadata[['UniProt', 'Cofactor', 'Family']].drop_duplicates()
        
        # Analysis for 4Fe-4S (SF4/HiPIP)
        sf4_proteins = unique_proteins[unique_proteins['Cofactor'] == '[4Fe-4S]']
        hipip_proteins = sf4_proteins[sf4_proteins['Family'] == 'HiPIP']
        
        sf4_count = len(sf4_proteins)
        hipip_count = len(hipip_proteins)
        other_sf4_count = sf4_count - hipip_count
        
        # Analysis for 2Fe-2S (FES/Rieske)
        fes_proteins = unique_proteins[unique_proteins['Cofactor'] == '[2Fe-2S]']
        rieske_proteins = fes_proteins[fes_proteins['Family'] == 'Rieske protein']
        
        fes_count = len(fes_proteins)
        rieske_count = len(rieske_proteins)
        other_fes_count = fes_count - rieske_count
        
        # Create pie chart for 4Fe-4S (HiPIP proportion)
        if sf4_count > 0:
            plt.figure(figsize=(8, 8))
            hipip_percent = (hipip_count / sf4_count) * 100
            other_sf4_percent = (other_sf4_count / sf4_count) * 100
            
            labels = [
                f'HiPIP\n({hipip_count}, {hipip_percent:.1f}%)',
                f'Other [4Fe-4S]\n({other_sf4_count}, {other_sf4_percent:.1f}%)'
            ]
            
            wedges, texts, autotexts = plt.pie(
                [hipip_count, other_sf4_count],
                labels=labels,
                autopct='',
                startangle=90,
                colors=['#FF9671', '#FFC75F']
            )
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            
            plt.title('HiPIP Proportion Among [4Fe-4S] Proteins', fontsize=14, fontweight='bold', pad=20)
            plt.savefig(self.output_dir / 'hipip_proportion.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create pie chart for 2Fe-2S (Rieske proportion)
        if fes_count > 0:
            plt.figure(figsize=(8, 8))
            rieske_percent = (rieske_count / fes_count) * 100
            other_fes_percent = (other_fes_count / fes_count) * 100
            
            labels = [
                f'Rieske\n({rieske_count}, {rieske_percent:.1f}%)',
                f'Other [2Fe-2S]\n({other_fes_count}, {other_fes_percent:.1f}%)'
            ]
            
            wedges, texts, autotexts = plt.pie(
                [rieske_count, other_fes_count],
                labels=labels,
                autopct='',
                startangle=90,
                colors=['#845EC2', '#B39CD0']
            )
            
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            
            plt.title('Rieske Proportion Among [2Fe-2S] Proteins', fontsize=14, fontweight='bold', pad=20)
            plt.savefig(self.output_dir / 'rieske_proportion.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'sf4_total': sf4_count,
            'hipip_count': hipip_count,
            'hipip_percent': (hipip_count / sf4_count * 100) if sf4_count > 0 else 0,
            'fes_total': fes_count,
            'rieske_count': rieske_count,
            'rieske_percent': (rieske_count / fes_count * 100) if fes_count > 0 else 0
        }

    
    def analyze_conditions(self) -> Dict:
        """Analyze experimental conditions (Em and pH)."""
        # Calculate overall Em statistics
        condition_stats = {
            'Em': {
                'mean': self.metadata['Em'].mean(),
                'median': self.metadata['Em'].median(),
                'std': self.metadata['Em'].std(),
                'min': self.metadata['Em'].min(),
                'max': self.metadata['Em'].max()
            },
            'pH': {
                'mean': self.metadata['pH'].mean(),
                'median': self.metadata['pH'].median(),
                'std': self.metadata['pH'].std(),
                'min': self.metadata['pH'].min(),
                'max': self.metadata['pH'].max()
            }
        }

        # Calculate Em statistics per cofactor type
        cofactor_em_stats = {}
        for cofactor in self.metadata['Cofactor'].unique():
            subset = self.metadata[self.metadata['Cofactor'] == cofactor]['Em']
            cofactor_em_stats[cofactor] = {
                'mean': subset.mean(),
                'median': subset.median(),
                'std': subset.std(),
                'min': subset.min(),
                'max': subset.max(),
                'count': len(subset)
            }
        
        # Save Em statistics to CSV
        em_stats_rows = []
        # Add overall statistics
        em_stats_rows.append({
            'Cofactor': 'All',
            'Count': len(self.metadata),
            'Mean': condition_stats['Em']['mean'],
            'Median': condition_stats['Em']['median'],
            'Std': condition_stats['Em']['std'],
            'Min': condition_stats['Em']['min'],
            'Max': condition_stats['Em']['max']
        })
        
        # Add per-cofactor statistics
        for cofactor, stats in cofactor_em_stats.items():
            em_stats_rows.append({
                'Cofactor': cofactor,
                'Count': stats['count'],
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max']
            })
        
        # Save to CSV
        em_stats_df = pd.DataFrame(em_stats_rows)
        em_stats_df.to_csv(self.output_dir / 'redox_potential_statistics.csv', index=False)
        self.logger.info("Redox potential statistics saved to redox_potential_statistics.csv")
        
        # Plotting Em and pH distributions
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.metadata, x='Em', bins=30)
        plt.title('Distribution of Redox Potentials')
        plt.xlabel('Em (mV)')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'em_dist.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.metadata, x='Cofactor', y='Em')
        plt.title('Redox Potential Distribution by Cofactor Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'em_by_cofactor.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.metadata, x='pH', bins=20)
        plt.title('Distribution of pH Values')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ph_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add cofactor Em stats to the return values
        condition_stats['Em_per_cofactor'] = cofactor_em_stats

        return condition_stats
    
    def analyze_mutations(self) -> Dict:
        """Analyze mutation statistics."""
        mutation_stats = {
            'total_entries': len(self.metadata),
            'wild_type': sum(self.metadata['Mutation_AF'] == 'no'),
            'mutant': sum(self.metadata['Mutation_AF'] != 'no'),
        }
        mutation_stats['proportion_mutant'] = mutation_stats['mutant'] / mutation_stats['total_entries'] * 100
                
        wt_count = mutation_stats['wild_type']
        mut_count = mutation_stats['mutant']
        total = mutation_stats['total_entries']
        wt_percent = (wt_count / total) * 100
        mut_percent = (mut_count / total) * 100
        
        labels = [
            f'Wild Type\n({wt_count}, {wt_percent:.1f}%)',
            f'Mutant\n({mut_count}, {mut_percent:.1f}%)'
        ]

        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            [wt_count, mut_count],
            labels=labels,
            autopct='',
            startangle=90,
            colors=['lightblue', 'lightcoral']
        )
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        plt.title('Proportion of Mutant Entries', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'mutation_proportion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return mutation_stats
    
    def analyze_correlations(self) -> Dict:
        """Analyze correlations between numerical variables."""
        numerical_cols = ['Em', 'pH', 'burial_depth']
        corr_matrix = self.metadata[numerical_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix.to_dict()

    def analyze_cofactor_statistics(self) -> Dict:
        """Analyze statistics grouped by cofactor type."""
        stats = []
        for cofactor in self.metadata['Cofactor'].unique():
            subset = self.metadata[self.metadata['Cofactor'] == cofactor]
            stats.append({
                'cofactor': cofactor,
                'count': len(subset),
                'em_mean': subset['Em'].mean(),
                'em_std': subset['Em'].std(),
                'em_median': subset['Em'].median(),
                'burial_mean': subset['burial_depth'].mean(),
                'burial_std': subset['burial_depth'].std(),
                'burial_median': subset['burial_depth'].median(),
                'ph_mean': subset['pH'].mean(),
                'ph_std': subset['pH'].std(),
                'ph_median': subset['pH'].median()
            })
        return {'cofactor_stats': stats}

    def analyze_protein_coverage(self) -> Dict:
        """Analyze how many unique proteins are in the dataset."""
        unique_proteins = self.metadata['UniProt'].nunique()
        proteins_per_cofactor = self.metadata.groupby('Cofactor')['UniProt'].nunique()
        
        coverage_stats = {
            'unique_proteins': unique_proteins,
            'proteins_per_cofactor': proteins_per_cofactor.to_dict(),
            'avg_cofactors_per_protein': len(self.metadata) / unique_proteins
        }
        
        # Keep the original bar plot
        plt.figure(figsize=(10, 6))
        ax = proteins_per_cofactor.plot(kind='bar')
        plt.title('Number of Unique Proteins per Cofactor Type')
        plt.xlabel('Cofactor Type')
        plt.ylabel('Number of Unique Proteins')
        plt.xticks(rotation=45)
        for i, v in enumerate(proteins_per_cofactor.values):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'proteins_per_cofactor_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create pie chart with both count and percentage shown
        plt.figure(figsize=(10, 8))
        counts = proteins_per_cofactor.values
        cofactor_names = proteins_per_cofactor.index
        total = unique_proteins  # Use the total unique proteins as denominator for percentages
        
        # Create labels with count and percentage
        labels = [f'{name}\n({count}, {(count/total)*100:.1f}%)' for name, count in zip(cofactor_names, counts)]
        
        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            counts,
            labels=labels,
            autopct='',  # No percentage inside the pie - already in the labels
            startangle=140,
            colors=plt.cm.Set3(range(len(counts)))
        )
        
        # Style the labels
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        
        plt.title('Unique Proteins per Cofactor Type', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'protein_coverage_pie_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return coverage_stats
        
        return coverage_stats
         
    def _create_simple_report(self, results: Dict):
        """Create a simple text report of the analysis."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("IRON-SULFUR PROTEIN DATASET ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 20)
        report_lines.append(f"Total proteins analyzed: {results['protein_sizes']['total_proteins']}")
        report_lines.append(f"Total cofactors: {results['cofactors']['total_cofactors']}")
        report_lines.append(f"Unique proteins: {results['protein_coverage']['unique_proteins']}")
        report_lines.append("")
        
        report_lines.append("PROTEIN SIZES")
        report_lines.append("-" * 15)
        ps = results['protein_sizes']
        report_lines.append(f"Mean residues: {ps['mean_residues']:.1f}")
        report_lines.append(f"Median residues: {ps['median_residues']:.1f}")
        report_lines.append(f"Min residues: {ps['min_residues']:.0f}")
        report_lines.append(f"Max residues: {ps['max_residues']:.0f}")
        report_lines.append(f"Average maximum diameter: {ps['mean_max_dimension']:.1f} Å")
        report_lines.append(f"Average surface area: {ps['mean_surface']:.1f} Å²")
        report_lines.append("")
        
        report_lines.append("COFACTOR DISTRIBUTION")
        report_lines.append("-" * 22)
        for cofactor, count in results['cofactors']['counts'].items():
            proportion = results['cofactors']['proportions'][cofactor] * 100
            report_lines.append(f"{cofactor}: {count} ({proportion:.1f}%)")
        report_lines.append("")

        if 'protein_families' in results:
            report_lines.append("PROTEIN FAMILY ANALYSIS")
            report_lines.append("-" * 22)
            pf = results['protein_families']
            
            # 4Fe-4S/HiPIP information
            report_lines.append(f"[4Fe-4S] proteins: {pf['sf4_total']}")
            report_lines.append(f"  HiPIP proteins: {pf['hipip_count']} ({pf['hipip_percent']:.1f}%)")
            
            # 2Fe-2S/Rieske information
            report_lines.append(f"[2Fe-2S] proteins: {pf['fes_total']}")
            report_lines.append(f"  Rieske proteins: {pf['rieske_count']} ({pf['rieske_percent']:.1f}%)")
            report_lines.append("")
        
        report_lines.append("COFACTOR SIZE STATISTICS")
        report_lines.append("-" * 25)
        for stat in results['cofactor_sizes']['cofactor_sizes']:
            report_lines.append(f"{stat['cofactor']}:")
            report_lines.append(f"  Residue Count - Mean: {stat['residue_count_mean']:.1f}, "
                              f"Median: {stat['residue_count_median']:.1f}, "
                              f"Std Dev: {stat['residue_count_std']:.1f}")
            report_lines.append(f"  Max Diameter - Mean: {stat['max_dimension_mean']:.1f} Å, "
                              f"Median: {stat['max_dimension_median']:.1f} Å, "
                              f"Std Dev: {stat['max_dimension_std']:.1f} Å")
        report_lines.append("")
        
        report_lines.append("BURIAL DEPTH STATISTICS")
        report_lines.append("-" * 25)
        bd = results['cofactors']['burial_depth']
        report_lines.append(f"Mean: {bd['mean']:.2f} Å")
        report_lines.append(f"Median: {bd['median']:.2f} Å")
        report_lines.append(f"Std Dev: {bd['std']:.2f} Å")
        report_lines.append(f"Range: {bd['min']:.2f} - {bd['max']:.2f} Å")
        report_lines.append("")
        
        report_lines.append("EXPERIMENTAL CONDITIONS")
        report_lines.append("-" * 23)
        report_lines.append("Redox Potential (Em):")
        em = results['conditions']['Em']
        report_lines.append(f"  Mean: {em['mean']:.1f} mV")
        report_lines.append(f"  Median: {em['median']:.1f} mV")
        report_lines.append(f"  Std Dev: {em['std']:.1f} mV")
        report_lines.append(f"  Range: {em['min']:.1f} - {em['max']:.1f} mV")
        report_lines.append("")
        report_lines.append("pH:")
        ph = results['conditions']['pH']
        report_lines.append(f"  Mean: {ph['mean']:.1f}")
        report_lines.append(f"  Median: {ph['median']:.1f}")
        report_lines.append(f"  Std Dev: {ph['std']:.1f}")
        report_lines.append(f"  Range: {ph['min']:.1f} - {ph['max']:.1f}")
        report_lines.append("")
        
        report_lines.append("MUTATION ANALYSIS")
        report_lines.append("-" * 17)
        mut = results['mutations']
        report_lines.append(f"Wild type entries: {mut['wild_type']}")
        report_lines.append(f"Mutant entries: {mut['mutant']}")
        report_lines.append(f"Proportion mutant: {mut['proportion_mutant']:.1f}%")
        report_lines.append("")
        
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        self._save_detailed_csvs(results)
        
        self.logger.info("Simple text report created: analysis_report.txt")

    def _save_detailed_csvs(self, results: Dict):
        """Save detailed statistics as CSV files for easy analysis."""
        cofactor_stats = pd.DataFrame(results['cofactor_statistics']['cofactor_stats'])
        cofactor_sizes = pd.DataFrame(results['cofactor_sizes']['cofactor_sizes'])
        merged_cofactor_stats = cofactor_stats.merge(
            cofactor_sizes[['cofactor', 'residue_count_mean', 'residue_count_median', 'residue_count_std', 
                           'max_dimension_mean', 'max_dimension_median', 'max_dimension_std']],
            on='cofactor', how='left'
        )
        merged_cofactor_stats.to_csv(self.output_dir / 'cofactor_detailed_stats.csv', index=False)
        
        # Save protein detailed stats
        protein_stats = pd.DataFrame(results['protein_detailed_stats']['protein_detailed_stats'])
        if not protein_stats.empty:
            protein_stats.to_csv(self.output_dir / 'protein_detailed_stats.csv', index=False)
        
        summary_data = []
        ps = results['protein_sizes']
        summary_data.append(['Protein Residues', 'Mean', ps['mean_residues']])
        summary_data.append(['Protein Residues', 'Median', ps['median_residues']])
        summary_data.append(['Protein Residues', 'Std Dev', ps['std_residues']])
        summary_data.append(['Protein Max Diameter', 'Mean (Å)', ps['mean_max_dimension']])
        summary_data.append(['Protein Surface', 'Mean (Å²)', ps['mean_surface']])
        
        bd = results['cofactors']['burial_depth']
        summary_data.append(['Burial Depth', 'Mean (Å)', bd['mean']])
        summary_data.append(['Burial Depth', 'Median (Å)', bd['median']])
        summary_data.append(['Burial Depth', 'Std Dev (Å)', bd['std']])
        
        em = results['conditions']['Em']
        summary_data.append(['Redox Potential', 'Mean (mV)', em['mean']])
        summary_data.append(['Redox Potential', 'Median (mV)', em['median']])
        summary_data.append(['Redox Potential', 'Std Dev (mV)', em['std']])
        
        ph = results['conditions']['pH']
        summary_data.append(['pH', 'Mean', ph['mean']])
        summary_data.append(['pH', 'Median', ph['median']])
        summary_data.append(['pH', 'Std Dev', ph['std']])
        
        summary_df = pd.DataFrame(summary_data, columns=['Property', 'Statistic', 'Value'])
        summary_df.to_csv(self.output_dir / 'summary_statistics.csv', index=False)
        
        cofactor_counts = pd.DataFrame([
            {'Cofactor': k, 'Count': v, 'Proportion': results['cofactors']['proportions'][k]}
            for k, v in results['cofactors']['counts'].items()
        ])
        cofactor_counts.to_csv(self.output_dir / 'cofactor_counts.csv', index=False)
        
        coverage_data = []
        coverage_data.append(['Total unique proteins', results['protein_coverage']['unique_proteins']])
        coverage_data.append(['Avg cofactors per protein', results['protein_coverage']['avg_cofactors_per_protein']])
        for cofactor, count in results['protein_coverage']['proteins_per_cofactor'].items():
            coverage_data.append([f'Proteins with {cofactor}', count])
        
        coverage_df = pd.DataFrame(coverage_data, columns=['Metric', 'Value'])
        coverage_df.to_csv(self.output_dir / 'protein_coverage.csv', index=False)
        
        self.logger.info("Detailed CSV files created in output directory")
    
    def run_analysis(self):
        """Run complete dataset analysis."""
        self.logger.info("Starting dataset analysis...")
        
        results = {
            'protein_sizes': self.analyze_protein_sizes(),
            'cofactors': self.analyze_cofactors(),
            'conditions': self.analyze_conditions(),
            'mutations': self.analyze_mutations(),
            'cofactor_statistics': self.analyze_cofactor_statistics(),
            'cofactor_sizes': self.analyze_cofactor_sizes(),
            'protein_coverage': self.analyze_protein_coverage(),
            'protein_detailed_stats': self.analyze_protein_detailed_stats(),
            'protein_families': self.analyze_protein_families()  # Add the new analysis here
        }
        
        serializable_results = convert_to_serializable(results)
        
        import json
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self._create_simple_report(results)
        self._save_detailed_csvs(results)
        self.logger.info("Analysis results saved to JSON and CSV files")
        
        self.logger.info("Analysis completed successfully")
        return results

def main():
    parser = argparse.ArgumentParser(description="Analyze iron-sulfur protein dataset")
    parser.add_argument("--pdb_dir", required=True, help="Directory containing PDB files")
    parser.add_argument("--features", required=True, 
                       help="CSV file with features (including Em, pH, burial depth)")
    parser.add_argument("--metadata", required=True,
                       help="Excel file with mutation information")
    parser.add_argument("--output_dir", help="Output directory for analysis results")
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.pdb_dir, args.features, args.metadata, args.output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()