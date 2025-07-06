import os

# Define input and output directories
input_dir = "AlphaFilling_cofactors/extra_data"
output_dir = "AlphaFilling_cofactors/extra_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Keywords for water and sodium ions in PDB files
residues_to_remove = {'HOH', 'WAT', 'H2O', 'NA', 'Na+', 'SOD'}

# Loop through all PDB files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".pdb"):
        input_path = os.path.join(input_dir, filename)
        base_name = filename.rsplit(".pdb", 1)[0]
        new_filename = f"{base_name}_no_water.pdb"
        output_path = os.path.join(output_dir, new_filename)

        removed_count = 0
        output_lines = []

        with open(input_path, 'r') as infile:
            for line in infile:
                if line.startswith(("ATOM", "HETATM", "ANISOU")):
                    resname = line[17:20].strip()
                    if resname not in residues_to_remove:
                        output_lines.append(line)
                    else:
                        removed_count += 1
                else:
                    output_lines.append(line)

        if removed_count > 0:
            with open(output_path, 'w') as outfile:
                outfile.writelines(output_lines)
            print(f"✅ {new_filename} written — removed {removed_count} residue lines")
        else:
            print(f"ℹ️ {filename} unchanged — no water/ions removed")
