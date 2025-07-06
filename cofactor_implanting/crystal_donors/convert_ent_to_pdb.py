import os
import gemmi

input_dir = "AlphaFilling_cofactors/extra_data"
output_dir = "AlphaFilling_cofactors/extra_data"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".ent"):
        ent_path = os.path.join(input_dir, filename)
        pdb_name = filename.replace(".ent", ".pdb")
        pdb_path = os.path.join(output_dir, pdb_name)

        try:
            structure = gemmi.read_structure(ent_path)
            structure.write_pdb(pdb_path)
            print(f"Converted: {filename} ➝ {pdb_name}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")
