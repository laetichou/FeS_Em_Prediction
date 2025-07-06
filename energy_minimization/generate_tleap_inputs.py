import os
import shutil
import pandas as pd
from pymol import cmd

# === Clean previous outputs ===
shutil.rmtree("Energy_minimization/tleap_runs", ignore_errors=True)

# === Config ===
pdb_folder = "protonated_structures"
csv_path = "Energy_minimization/solvation_parameters.csv"
template_path = "Energy_minimization/Optimization_step_files/tleapOrders"
output_base = "Energy_minimization/tleap_runs"

# === Load data ===
df = pd.read_csv(csv_path)

with open(template_path, "r") as f:
    template = f.read()

os.makedirs(output_base, exist_ok=True)

for _, row in df.iterrows():
    name = row["Protein"]
    na_count = int(row["Na+"])
    cl_count = int(row["Cl-"])

    folder = os.path.join(output_base, name)
    os.makedirs(folder, exist_ok=True)

    pqr_src = os.path.join(pdb_folder, f"{name}_Structure.pqr")
    pdb_dst = os.path.join(folder, f"{name}_pqr.pdb")

    if not os.path.exists(pqr_src):
        print(f"PQR missing for {name}, skipping.")
        continue

    # === Strip hydrogens using PyMOL ===
    cmd.reinitialize()
    cmd.load(pqr_src, "prot")
    cmd.remove("hydrogen")
    cmd.save(pdb_dst, "prot")
    print(f"Hydrogens removed and saved: {pdb_dst}")

    # === Create tleap.in with updated ion counts ===
    tleap_script = template.replace("your_name", f"{name}")
    tleap_script = tleap_script.replace("addIonsRand X Na+ 0", f"addIonsRand X Na+ {na_count}")
    tleap_script = tleap_script.replace("addIonsRand X Cl- 0", f"addIonsRand X Cl- {cl_count}")

    tleap_path = os.path.join(folder, "tleap.in")
    with open(tleap_path, "w") as out:
        out.write(tleap_script)

    print(f"Wrote tleap.in and cleaned .pdb for {name}")
