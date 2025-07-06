import os
import pandas as pd

pqr_folder = "protonated_structures"
output_csv = "Energy_minimization/solvation_parameters.csv"

data = []

for filename in os.listdir(pqr_folder):
    if not filename.endswith(".pqr"):
        continue

    path = os.path.join(pqr_folder, filename)
    total_charge = 0.0

    with open(path, "r") as file:
        for line in file:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                try:
                    charge = float(parts[8])  # Column for charge in .pqr files
                    total_charge += charge
                except (IndexError, ValueError):
                    continue

    # Round to nearest integer for ion calculation
    net_charge = round(total_charge)
    name = filename.replace("_Structure.pqr", "")

    if net_charge > 0:
        na_count = 0
        cl_count = net_charge
    elif net_charge < 0:
        na_count = -net_charge
        cl_count = 0
    else:
        na_count = cl_count = 0

    data.append({
        "Protein": name,
        "Charge": net_charge,
        "Na+": int(na_count),
        "Cl-": int(cl_count)
    })

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
