import os
import csv
from pymol import cmd

'''Run with:
source ~/.zshrc
cd BEP
pymol -cq AlphaFilling_cofactors/transplant_cofactors.py
'''

'''What this script does:
- Assumes every .pdb in input_structures/ is a mutant minimized structure
- Extracts the UniProt ID and mutation from the filename (format: {UniProt}_{Mutation}_minimized.pdb)
- Loads the corresponding AlphaFilled WT structure from alphafilled_wild_types/
- Aligns the mutant to the WT structure (with fallback and chain fixes)
- Transfers HETATM records (cofactors)
- Saves to alphafilled_mutants/{UniProt}_{Mutation}_filled.pdb
- Logs failures and alignment RMSD
'''

# === Config ===
input_folder = "AlphaFilling_cofactors/input_structures"
alphafill_folder = "AlphaFilling_cofactors/alphafilled_wild_types"  # <-- updated
output_folder = "AlphaFilling_cofactors/alphafilled_mutants"
os.makedirs(output_folder, exist_ok=True)

missing_log = os.path.join(output_folder, "missing_alphafill_for_mutants.txt")
rmsd_log_path = os.path.join(output_folder, "alignment_rmsd_log.csv")

# === Init logs ===
missing = []
rmsd_log = []

# === Loop through minimized mutant PDBs ===
for filename in os.listdir(input_folder):
    if not filename.endswith(".pdb"):
        continue

    try:
        base_name = filename.replace("_minimized.pdb", "")
        uniprot, mutation = base_name.split("_", 1)
    except ValueError:
        print(f"⚠️ Skipping malformed filename: {filename}")
        continue

    input_path = os.path.join(input_folder, filename)

    # Try to find transplanted or original file
    alphafill_pdb = None
    possible_files = [
        os.path.join(alphafill_folder, f"{uniprot}_transplanted.pdb"),
        os.path.join(alphafill_folder, f"{uniprot}.pdb")
    ]
    for file in possible_files:
        if os.path.exists(file):
            alphafill_pdb = file
            break

    if not alphafill_pdb:
        print(f"❌ AlphaFilled WT structure not found for {uniprot}")
        missing.append(f"{uniprot}_{mutation}")
        rmsd_log.append([uniprot, mutation, "", "", "AlphaFill missing"])
        continue

    print(f"🔁 Transplanting cofactors for {uniprot}_{mutation}...")

    cmd.reinitialize()
    cmd.load(input_path, "mutant")
    cmd.load(alphafill_pdb, "alphafill")

    cmd.alter("mutant", "chain='A'")
    cmd.sort()

    fallback_used = "No"
    rmsd = ""

    try:
        result = cmd.align("mutant", "alphafill")
        rmsd = result[0]
        if rmsd == -1 or rmsd > 10:
            print(f"⚠️ RMSD too high ({rmsd:.2f}), retrying CA-only...")
            fallback_used = "Yes"
            result = cmd.align("mutant and name CA", "alphafill and name CA")
            rmsd = result[0]
            if rmsd == -1:
                raise ValueError("Fallback CA-only alignment failed")

    except Exception as e:
        print(f"❌ Alignment failed for {uniprot}_{mutation}: {e}")
        missing.append(f"{uniprot}_{mutation} (alignment failed)")
        rmsd_log.append([uniprot, mutation, "", fallback_used, "Alignment failed"])
        continue

    # Transplant cofactors
    cmd.select("cofactors", "alphafill and hetatm")
    cmd.create("mutant_with_cofactors", "mutant")
    cmd.create("cofactors_only", "cofactors")
    cmd.create("final", "mutant_with_cofactors or cofactors_only")

    # Save structure
    output_file = os.path.join(output_folder, f"{uniprot}_{mutation}_filled.pdb")
    cmd.save(output_file, "final")
    print(f"✅ Saved: {output_file}")

    # Log successful alignment
    rmsd_log.append([uniprot, mutation, f"{rmsd:.3f}", fallback_used, "Success"])

# === Write missing entries
if missing:
    with open(missing_log, "w") as f:
        for entry in missing:
            f.write(entry + "\n")
    print(f"\n📝 Logged missing/failed entries to: {missing_log}")

# === Write RMSD log
with open(rmsd_log_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["UniProt", "Mutation", "RMSD", "FallbackUsed", "Status"])
    writer.writerows(rmsd_log)

print(f"🧾 Alignment RMSDs saved to: {rmsd_log_path}")
