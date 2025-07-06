# utils_structure.py

from Bio.PDB import PDBParser
import numpy as np

FE_S_COF_NAMES = ["FES", "SF4", "3FE", "4FE", "F3S", "FE2", "FE4", "FE"]

def get_all_FeS_clusters(structure):
    """
    Identify all Fe-S cluster cofactors in the structure.
    Returns a list of (residue, cofactor_type, list_of_atoms)
    """
    clusters = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in FE_S_COF_NAMES:
                    atoms = list(residue.get_atoms())
                    if atoms:
                        clusters.append((residue, resname, atoms))

    return clusters

def get_baricenter(atoms, atom_names=None):
    coords = [
        atom.coord for atom in atoms
        if atom_names is None or any(an in atom.get_name() for an in atom_names)
    ]
    coords = np.array(coords)
    return coords.mean(axis=0) if len(coords) > 0 else None

def get_atoms_coord(atoms):
    """
    Return a dictionary of atom name to coordinates.
    """
    return {atom.get_name(): atom.coord for atom in atoms}