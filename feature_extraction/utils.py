# utils.py
import numpy as np
from Bio.PDB.Polypeptide import is_aa

def initialize_amino_acid_counts(aa_names):
    """
    Create a dictionary for counting amino acids across different regions.
    Keys include 'Protein.ALA', 'Bar.ALA', etc.
    """
    counts = {}
    for aa in aa_names:
        for scope in ["Protein", "Bar"]:
            counts[f"{scope}.{aa}"] = 0
    return counts


def count_amino_acids(count_dict, structure, bar_center, bar_radius, aa_names):
    """
    Count amino acids within bar_radius of cofactor center from ALL protein chains.
    
    Parameters:
    - count_dict: Dictionary to store counts
    - structure: Biopython Structure object
    - bar_center: Center point of the cofactor (Fe barycenter)
    - atom_coords: Dict of cofactor atom coordinates (unused here but kept for compatibility)
    - bar_radius: Radius for local environment analysis (in Ångströms)
    - aa_names: List of valid amino acid names
    
    Returns updated count_dict
    """
    model = structure[0]  # Use first model only
    
    for chain in model:
        # Skip non-protein chains (optional)
        if not any(is_aa(residue) for residue in chain):
            continue
            
        for residue in chain:
            if not is_aa(residue):
                continue
                
            aa = residue.resname
            if aa not in aa_names:
                continue

            # Get CA atom position (primary reference point for other residues)
            if "CA" not in residue:
                continue
            ca_pos = residue["CA"].coord

            # Count in full protein
            count_dict[f"Protein.{aa}"] += 1

            # Count if within bar_radius of cofactor center
            if np.linalg.norm(ca_pos - bar_center) <= bar_radius:
                count_dict[f"Bar.{aa}"] += 1

    return count_dict


def count_amino_acids_old(count_dict, chain, bar_center, atom_coords, ref_point, bar_radius, ring_radius, residues_dict, aa_names):
    """
    Count amino acids within bar_radius (Bar) and ring_radius (Ring) of the cofactor center.
    Also count amino acids across the full protein (Protein).
    """
    for residue in chain:
        if not hasattr(residue, "resname"):
            continue
        aa = residue.resname
        if aa not in aa_names:
            continue

        res_id = residue.id[1]

        # Position of CA atom
        ca = residue["CA"].coord if "CA" in residue else None
        if ca is None:
            continue

        # Always count full protein
        count_dict[f"Protein.{aa}"] += 1

        # Within Bar radius
        if np.linalg.norm(ca - bar_center) <= bar_radius:
            count_dict[f"Bar.{aa}"] += 1

        # Within Ring radius (closer to each atom)
        for coord in atom_coords.values():
            if np.linalg.norm(ca - coord) <= ring_radius:
                count_dict[f"Ring.{aa}"] += 1
                break

    return count_dict, None, None  # No N5-based outputs needed

def specific_feature(count_dict, prefix="", mean=True, total=1):
    """
    Add aggregate features such as average hydrophobicity, charge, etc. for a region.
    Mean is calculated by dividing totals by `total` (e.g., number of residues in region).
    """
    for key in list(count_dict.keys()):
        if key.startswith(prefix):
            if mean:
                count_dict[f"{key}.mean"] = count_dict[key] / total
    return count_dict