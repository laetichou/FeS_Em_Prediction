import os
import subprocess

'''To run:
cd BEP
python AlphaFilling_cofactors/prepare_alphafill_inputs.py
'''

'''
This script converts the minimized mutant structure .gro files to .pdb
And places them into input_structures folder, to use in transplant_cofactors.py

If you want to RErun this script, make sure to correctly delete input_structures folder first
With `rm -rf AlphaFilling_cofactors/input_structures` in Terminal 
'''

# === Config ===
mutant_dir = "Energy_minimization/tleap_runs"
output_dir = "AlphaFilling_cofactors/input_structures"
os.makedirs(output_dir, exist_ok=True)

# === Convert minimized GRO files to PDB ===
for name in os.listdir(mutant_dir):
    print(f"Found {len(os.listdir(mutant_dir))} mutant directories:")
    print(os.listdir(mutant_dir))
    folder = os.path.join(mutant_dir, name)
    gro_file = os.path.join(folder, f"{name}_minim.gro")
    pdb_out = os.path.join(output_dir, f"{name}_minimized.pdb")

    if not os.path.exists(gro_file):
        print(f"GRO file missing for {name}, skipping.")
        continue

    print(f"Converting {gro_file} to PDB...")

    result = subprocess.run(
        ["gmx", "editconf", "-f", gro_file, "-o", pdb_out],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"FAILED to convert {name}: {result.stderr}")
    else:
        print(f"Saved: {pdb_out}")

