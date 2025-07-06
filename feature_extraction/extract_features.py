#!/usr/bin/env python3
"""
Feature Extraction Script for Iron-Sulfur Proteins
------------------------------------------------
Extracts and organizes features from PDB files containing iron-sulfur clusters:

Categories:
- counts: Amino acid counts (Bar region vs whole Protein)
- properties: Physicochemical properties from tableAmm
- means: Mean values of properties in each region
- structure: Structural features (burial depth)
- sequence: CTD-based sequence features

Usage: python extract_features.py --log DEBUG
"""

import os
import logging
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from PyBioMed.PyProtein import CTD
from datetime import datetime
from scipy.spatial import ConvexHull, distance_matrix

from utils_structure import get_all_FeS_clusters, get_baricenter, get_atoms_coord
from utils import (
    initialize_amino_acid_counts,
    count_amino_acids,
    specific_feature,
)

# --- CONFIGURATION ---
OUTPUT_PATH = "Prediction_FeS_EM/final_stretch/feature_extraction/output"
PDB_DIR = "Prediction_FeS_EM/final_structure_dataset"
TABLE_AMM_PATH = "Prediction_FeS_EM/final_stretch/feature_extraction/data/tableAmm.txt"
LIST_BAR = np.arange(1,17)  
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]
HIPIP_IDS = [
    "P00260", "P00261", "P00262", "P00263", "P00264", "P00265", "P00266",
    "P04168", "P04169", "P33678", "P38524", "P38589", "P38941", "P80882"
]

RIESKE_IDS = [
    "D3FQ82", "O52396", "P37332", "Q02762", "Q46136", "Q9HH83"
]

# Feature organization
FEATURE_CATEGORIES = {
    'counts': {
        'description': 'Amino acid counts in Bar/Protein regions',
        'prefix': 'count',
        'examples': ['Bar.count.ALA', 'Protein.count.CYS']
    },
    'properties': {
        'description': 'Properties from tableAmm (sums)',
        'prefix': 'prop',
        'examples': ['Bar.prop.Volume', 'Protein.prop.Hydrophobicity']
    },
    'means': {
        'description': 'Mean property values per region',
        'prefix': 'mean',
        'examples': ['Bar.mean.Volume', 'Protein.mean.Hydrophobicity']
    },
    'structure': {
        'description': 'Structural features',
        'prefix': 'struct',
        'examples': ['burial_depth', 'iron_count', 'is_rieske', 'is_hipip']
    },
    'sequence': {
        'description': 'CTD sequence-based features',
        'prefix': 'seq',
        'examples': ['seq.CTDC_PolarizabilityC1', 'seq.CTDT_PolarizabilityT12']
    }
}

# Feature metadata template
FEATURE_METADATA = {
    'categories': FEATURE_CATEGORIES,
    'date_generated': datetime.now().isoformat(),
    'description': 'Feature definitions and organization for iron-sulfur protein analysis'
}


def create_feature_documentation(table_amm: pd.DataFrame) -> dict:
    """Create comprehensive documentation for all features."""
    documentation = {
        'metadata': FEATURE_METADATA,
        'features': {},
        'property_descriptions': {
            'Volume': 'Amino acid volume',
            'Log(Solubility (m at 20ºC))': 'Logarithm of solubility in moles/L at 20ºC',
            'Hydrophobicity': 'Hydrophobicity scale value',
            'Isoelectric point': 'pH at which amino acid carries no net charge',
            'P(helix)': 'Helix propensity',
            'nOats': 'Number of oxygen atoms in side chain',
            'nSats': 'Number of sulfur atoms in side chain',
            'nNats': 'Number of nitrogen atoms in side chain',
            'Steric hindrance': 'Steric parameter value',
            'nH-bonds': 'Number of potential hydrogen bonds',
            'Flexibility': 'Flexibility parameter'
        }
    }
    
    # Document amino acid count features
    for scope in ['Bar', 'Protein']:
        for aa in AMINO_ACIDS:
            feature_name = f"{scope}.count.{aa}"
            documentation['features'][feature_name] = {
                'category': 'counts',
                'description': f'Count of {aa} residues in the {scope.lower()} region',
                'type': 'integer',
                'range': '[0, inf)'
            }
    
    # Document property features
    for scope in ['Bar', 'Protein']:
        for col in table_amm.columns:
            # Property sums
            prop_name = f"{scope}.prop.{col}"
            documentation['features'][prop_name] = {
                'category': 'properties',
                'description': (f'Sum of {documentation["property_descriptions"].get(col, col)} '
                              f'for all residues in {scope.lower()} region'),
                'type': 'float'
            }
            
            # Property means
            mean_name = f"{scope}.mean.{col}"
            documentation['features'][mean_name] = {
                'category': 'means',
                'description': (f'Average {documentation["property_descriptions"].get(col, col)} '
                              f'per residue in {scope.lower()} region'),
                'type': 'float'
            }
    
    # Document structural features
    documentation['features'].update({
        'struct.burial_depth': {
            'category': 'structure',
            'description': 'Minimum distance from cofactor barycenter to protein surface',
            'type': 'float',
            'units': 'Angstroms'
        },
        'struct.iron_count': {
            'category': 'structure',
            'description': 'Number of iron atoms in the cofactor',
            'type': 'integer',
            'range': '[1, 4]'
        },
        'struct.is_rieske': {
            'category': 'structure',
            'description': 'Binary indicator for Rieske proteins (1 if Rieske, 0 otherwise)',
            'type': 'integer',
            'range': '[0, 1]'
        },
        'struct.is_hipip': {
            'category': 'structure',
            'description': 'Binary indicator for High-Potential Iron Protein (1 if HiPIP, 0 otherwise)',
            'type': 'integer',
            'range': '[0, 1]'
        }
    })
    
    # Document CTD features
    ctd_descriptions = {
        'Polarizability': 'Distribution of atomic polarizabilities',
        'SolventAccessibility': 'Distribution of solvent accessibility values',
        'SecondaryStr': 'Secondary structure propensity distribution',
        'Charge': 'Distribution of charge states',
        'Polarity': 'Distribution of polarity values',
        'NormalizedVDWV': 'Distribution of normalized van der Waals volumes',
        'Hydrophobicity': 'Distribution of hydrophobicity values'
    }
    
    # Add CTD feature documentation
    for prop, desc in ctd_descriptions.items():
        # Composition features
        for i in range(1, 4):
            feature_name = f"seq.CTDC_{prop}C{i}"
            documentation['features'][feature_name] = {
                'category': 'sequence',
                'description': f'{desc} - Fraction of amino acids in group {i}',
                'type': 'float',
                'range': '[0, 1]'
            }
        
        # Transition features
        for i in range(1, 3):
            for j in range(i+1, 4):
                feature_name = f"seq.CTDT_{prop}T{i}{j}"
                documentation['features'][feature_name] = {
                    'category': 'sequence',
                    'description': f'{desc} - Frequency of transitions between groups {i} and {j}',
                    'type': 'float',
                    'range': '[0, 1]'
                }
    
    return documentation


# --- SETUP AND UTILITIES ---
def setup_logging(output_path: str, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with file and console output."""
    os.makedirs(output_path, exist_ok=True)
    
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [
        logging.FileHandler(f"{output_path}/extract_features.log", mode="w"),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class FeatureTracker:
    """Track feature extraction progress and metadata."""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata_path = os.path.join(output_path, "metadata")
        os.makedirs(self.metadata_path, exist_ok=True)
        self.tracking_data = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tracking_file = f"{output_path}/processing_tracker_{self.timestamp}.csv"
        
    def add_entry(self, structure_id: str, cofactor_id: str, cofactor_type: str, 
             features: dict, sequence: str = None, radius: float = None):
            """Add a processing entry with feature statistics."""
            # Count features by category based on the defined prefixes
            feature_counts = {
                'counts': sum(1 for k in features if 'count.' in k),
                'properties': sum(1 for k in features if 'prop.' in k),
                'means': sum(1 for k in features if 'mean.' in k),
                'structure': sum(1 for k in features if 'struct.' in k),
                'sequence': sum(1 for k in features if 'seq.' in k)
            }
            
            entry = {
                "structure_id": structure_id,
                "cofactor_id": cofactor_id,
                "cofactor_type": cofactor_type,
                "sequence_length": len(sequence) if sequence else 0,
                "num_features": len(features),
                "num_features_by_category": feature_counts,
                "radius": radius,
                "timestamp": datetime.now().isoformat()
            }
            self.tracking_data.append(entry)
    
    def save(self):
        """Save tracking data to CSV with feature category statistics.
            Also save tracking data and feature documentation."""
        df = pd.DataFrame(self.tracking_data)
        df.to_csv(self.tracking_file, index=False)
        
        # Save feature metadata to folder
        metadata_file = os.path.join(self.metadata_path, f"feature_metadata_{self.timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(FEATURE_METADATA, f, indent=2)

        # Save feature documentation
        feature_doc = create_feature_documentation(table_amm)
        doc_file = os.path.join(self.metadata_path, f"feature_documentation_{self.timestamp}.json")
        with open(doc_file, 'w') as f:
            json.dump(feature_doc, f, indent=2)

def validate_features(features: dict) -> list:
    """Validate extracted features for common issues."""
    issues = []
    
    # Check for missing values
    missing = [k for k, v in features.items() if pd.isna(v)]
    if missing:
        issues.append(f"Missing values in features: {missing}")
    
    # Check for invalid counts (negative values)
    invalid_counts = [
        k for k, v in features.items() 
        if k.startswith('count.') and isinstance(v, (int, float)) and v < 0
    ]
    if invalid_counts:
        issues.append(f"Negative values in count features: {invalid_counts}")
    
    # Check for unreasonable ratios (Bar counts > Protein counts)
    for aa in AMINO_ACIDS:
        bar_count = features.get(f"Bar.count.{aa}", 0)
        total_count = features.get(f"Protein.count.{aa}", 0)
        if bar_count > total_count:
            issues.append(f"Bar count exceeds protein count for {aa}")
    
    return issues

def organize_features(raw_features: dict) -> dict:
    """Organize features into categories with consistent naming."""
    organized = {}
    
    # Organize amino acid counts
    for scope in ['Bar', 'Protein']:
        for aa in AMINO_ACIDS:
            key = f"{scope}.{aa}"
            if key in raw_features:
                organized[f"{scope}.count.{aa}"] = raw_features[key]
    
    # Organize properties and means
    for scope in ['Bar', 'Protein']:
        for col in table_amm.columns:
            prop_key = f"{scope}.{col}"
            if prop_key in raw_features:
                organized[f"{scope}.prop.{col}"] = raw_features[prop_key]
                mean_key = f"{scope}.mean.{col}"
                if mean_key in raw_features:
                    organized[mean_key] = raw_features[mean_key]
    
    # Add structural features
    if 'burial_depth' in raw_features:
        organized['struct.burial_depth'] = raw_features['burial_depth']
    
    # Add sequence (CTD) features with proper prefixes
    for k, v in raw_features.items():
        if k.startswith(('CTDC_', 'CTDT_')):
            organized[f"seq.{k}"] = v
    
    return organized


def get_protein_sequence(structure):
    """Extract and concatenate protein sequences from ALL chains."""
    from Bio.PDB import PPBuilder
    
    ppb = PPBuilder()
    full_sequence = []
    
    for chain in structure[0]:  # Use only first model
        if any(is_aa(residue) for residue in chain):
            chain_sequence = ["".join(str(pp.get_sequence()) 
                            for pp in ppb.build_peptides(chain))]
            if chain_sequence:
                full_sequence.extend(chain_sequence)
    
    return "".join(full_sequence) if full_sequence else None

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
    bar_center = np.array(get_baricenter(cofactor_atoms, atom_names=["FE"]))
    
    if bar_center.shape != (3,):
        raise ValueError(f"Invalid barycenter shape: {bar_center.shape}")
        
    distances = np.linalg.norm(np.array(ca_coords)[hull.vertices] - bar_center, axis=1)
    return float(min(distances))

def extract_features(structure, cofactor_info, bar_radius, structure_id, logger):
    """Extract all features for a single cofactor."""
    residue, cof_type, atoms, depth = cofactor_info
    chain = residue.get_parent()
    
    # Get barycenter and check validity
    bar_center = get_baricenter(atoms, atom_names=["FE"])
    if bar_center is None:
        raise ValueError("Could not compute cofactor barycenter")
    
    # Initialize feature dictionary with categories
    features = {cat: {} for cat in FEATURE_CATEGORIES}
    
    # 1. Extract sequence and CTD features
    sequence = get_protein_sequence(structure)
    if sequence:
        try:
            ctd_comp = CTD.CalculateC(sequence)
            ctd_trans = CTD.CalculateT(sequence)
            
            features['sequence'].update({
                f"seq.CTDC_{k}": v for k, v in ctd_comp.items()
            })
            features['sequence'].update({
                f"seq.CTDT_{k}": v for k, v in ctd_trans.items()
            })
        except Exception as e:
            logger.error(f"CTD calculation failed: {e}")
    
    # 2. Count amino acids and compute properties
    aa_counts = initialize_amino_acid_counts(AMINO_ACIDS)
    aa_counts = count_amino_acids(
        count_dict=aa_counts,
        structure=structure,
        bar_center=bar_center,
        bar_radius=bar_radius,
        aa_names=AMINO_ACIDS
    )
    
    # 3. Compute property sums and means
    for scope in ['Bar', 'Protein']:
        total = sum(aa_counts[f"{scope}.{aa}"] for aa in AMINO_ACIDS) or 1
        
        # Add counts to features
        features['counts'].update({
            f"{scope}.count.{aa}": aa_counts[f"{scope}.{aa}"]
            for aa in AMINO_ACIDS
        })
        
        # Add properties
        for col in table_amm.columns:
            prop_sum = sum(
                table_amm[col][aa] * aa_counts[f"{scope}.{aa}"]
                for aa in AMINO_ACIDS
            )
            features['properties'][f"{scope}.prop.{col}"] = prop_sum
            features['means'][f"{scope}.mean.{col}"] = prop_sum / total
    
    # 4. Add structural features
    # Iron count
    fe_atoms = [atom for atom in atoms if atom.element == "FE"]
    # Protein type
    uniprot_id = structure_id.split('_')[0]

    features['structure'].update({
        'struct.burial_depth': depth,
        'struct.iron_count': len(fe_atoms),
        'struct.is_rieske': 1 if uniprot_id in RIESKE_IDS else 0,
        'struct.is_hipip': 1 if uniprot_id in HIPIP_IDS else 0
    })
    
    # Flatten feature dictionary
    flat_features = {}
    for category, category_features in features.items():
        flat_features.update(category_features)
    
    return flat_features

def main():
    """Main execution function."""
    # Parse arguments and setup logging
    parser = argparse.ArgumentParser(description="Extract features from PDB files.")
    parser.add_argument("--log", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    
    logger = setup_logging(OUTPUT_PATH, args.log)
    tracker = FeatureTracker(OUTPUT_PATH)
    
    # Load amino acid property table
    global table_amm
    table_amm = pd.read_csv(TABLE_AMM_PATH, sep="\t", index_col=1)
    table_amm.index = [el.upper() for el in table_amm.index]
    table_amm = table_amm.iloc[:, 1:]
    
    # Process each radius
    for bar in LIST_BAR:
        logger.info(f"Processing features for radius {bar}Å")
        features_list = []
        metadata_list = []
        
        # Process each PDB file
        for pdb_file in tqdm(os.listdir(PDB_DIR)):
            if not pdb_file.endswith(".pdb"):
                continue
                
            try:
                # Process structure and extract features
                structure_id = pdb_file.replace(".pdb", "")
                structure = PDBParser(QUIET=True).get_structure(
                    structure_id, 
                    os.path.join(PDB_DIR, pdb_file)
                )
                
                # Get clusters and compute depths
                clusters = get_all_FeS_clusters(structure)
                clusters_with_depth = [
                    (*cluster, compute_cofactor_depth(structure, cluster[2]))
                    for cluster in clusters
                ]

                # Count the number of each cofactor type
                type_counts = {typ: sum(1 for x in clusters_with_depth if x[1] == typ) 
                            for typ in set(x[1] for x in clusters_with_depth)}
                cofactor_type_indices = {}
                sf4_clusters = [x for x in clusters_with_depth if x[1] == "SF4"]
                
                labels_by_index = {}
                
                # Special handling for P18187_P238C
                if structure_id == "P18187_P238C" and len(sf4_clusters) == 3:
                    centers = [get_baricenter(c[2], atom_names=["FE"]) for c in sf4_clusters]
                    d01 = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
                    d02 = np.linalg.norm(np.array(centers[0]) - np.array(centers[2]))
                    d12 = np.linalg.norm(np.array(centers[1]) - np.array(centers[2]))
                    dists = [d01 + d02, d01 + d12, d02 + d12]
                    intermediate_idx = np.argmin(dists)
                    remaining = [i for i in range(3) if i != intermediate_idx]
                    depths = [sf4_clusters[i][3] for i in remaining]
                    sorted_by_depth = sorted(zip(depths, remaining), reverse=True)
                    labels_by_index[intermediate_idx] = "intermediate"
                    labels_by_index[sorted_by_depth[0][1]] = "proximal"
                    labels_by_index[sorted_by_depth[1][1]] = "distal"
                
                # Process each cofactor with proper labeling
                for idx, cofactor_info in enumerate(clusters_with_depth):
                    residue, cof_type, atoms, depth = cofactor_info
                    
                    if cof_type not in cofactor_type_indices:
                        cofactor_type_indices[cof_type] = 0
                    
                    # Determine cofactor label
                    label = ""
                    if structure_id == "P18187_P238C" and cof_type == "SF4" and idx in labels_by_index:
                        label = labels_by_index[idx]
                    elif cof_type == "SF4" and type_counts["SF4"] == 2:
                        label = ["proximal", "distal"][cofactor_type_indices[cof_type]]
                    elif cof_type == "F3S" and type_counts.get("SF4", 0) == 2 and type_counts.get("F3S", 0) == 1:
                        label = "intermediate"
                    
                    # Create cofactor ID
                    cof_id = f"{structure_id}_{cof_type}_{label}" if label else f"{structure_id}_{cof_type}"
                    cofactor_type_indices[cof_type] += 1
                    
                    features = extract_features(structure, cofactor_info, bar, structure_id, logger)
                    features['cofactor_id'] = cof_id  # Add cofactor_id to features
                    
                    # Add to results
                    features_list.append(features)
                    metadata_list.append({
                        "structure_id": structure_id,
                        "cofactor_id": cof_id,
                        "cofactor_type": cof_type,
                        "burial_depth": depth,
                        "radius": bar
                    })
                    
                    # Update tracker
                    tracker.add_entry(
                        structure_id=structure_id,
                        cofactor_id=cof_id,
                        cofactor_type=cof_type,
                        features=features,
                        radius=bar
                    )
                
            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}")
                continue
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = os.path.join(OUTPUT_PATH, "metadata")
        os.makedirs(metadata_path, exist_ok=True)
        
        df_features = pd.DataFrame(features_list)
        cols = ['cofactor_id'] + [col for col in df_features.columns if col != 'cofactor_id']  # Ensure cofactor_id is first column before setting as index
        df_features = df_features[cols]
        df_features.set_index('cofactor_id', inplace=True)  # Set cofactor_id as index
        df_metadata = pd.DataFrame(metadata_list)
        
        df_features.to_csv(f"{OUTPUT_PATH}/features_r{bar}_final_{timestamp}.csv")
        df_metadata.to_csv(f"{metadata_path}/metadata_r{bar}_final_{timestamp}.csv")
        
    # Save tracker and feature metadata
    tracker.save()
    logger.info("Feature extraction completed successfully")

if __name__ == "__main__":
    main()