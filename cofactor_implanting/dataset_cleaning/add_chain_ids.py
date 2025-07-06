import os

def add_chain_id_if_missing(pdb_line, default_chain_id="A"):
    """Add chain ID if missing, but preserve existing chain IDs."""
    if pdb_line.startswith("ATOM") or pdb_line.startswith("HETATM"):
        # Ensure line is at least 22 characters long (chain ID is at position 21)
        pdb_line = pdb_line.rstrip().ljust(22)
        current_chain = pdb_line[21].strip()
        if not current_chain:  # Only add default if chain ID is empty
            pdb_line = pdb_line[:21] + default_chain_id + pdb_line[22:]
    return pdb_line

def process_pdb_file(input_path, output_path, default_chain_id="A"):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    fixed_lines = [add_chain_id_if_missing(line, default_chain_id) for line in lines]
    with open(output_path, 'w') as f:
        f.write('\n'.join(fixed_lines) + '\n')

def verify_chain_ids(pdb_path):
    """Check for duplicate or unexpected chain IDs in a PDB file."""
    chains = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21].strip()
                if chain_id:  # Only count non-empty chain IDs
                    chains.add(chain_id)
    return chains

def process_pdb_folder(input_folder, output_folder, default_chain_id="A"):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdb'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Process the file
            process_pdb_file(input_path, output_path, default_chain_id)
            
            # Verify the output
            chains = verify_chain_ids(output_path)
            print(f"Processed {filename} → Chains found: {chains}")
            
            if len(chains) > 1 and default_chain_id in chains:
                print(f"  Warning: Multiple chains found including default chain {default_chain_id}")
                
# Example usage
if __name__ == "__main__":
    input_folder = "Prediction_FeS_EM/final_structure_dataset_crystal"       # Replace with your actual folder
    output_folder = "Prediction_FeS_EM/final_structure_dataset_crystal_with_ids"    # Replace or leave as-is to create 'fixed_pdbs'
    process_pdb_folder(input_folder, output_folder)
