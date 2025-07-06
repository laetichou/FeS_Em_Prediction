import os
from pymol import cmd
from datetime import datetime
import csv

'''Run with:
source ~/.zshrc
cd BEP
pymol -cq AlphaFilling_cofactors/manual_cofactor_transplant.py
'''

# === User Config ===

# Input files
DONOR_FILE = "AlphaFilling_cofactors/late_stage_data_fixes/Q9X115_filled.pdb"
TARGET_FILE = "AlphaFilling_cofactors/late_stage_data_fixes/Q9X1L9_empty.pdb"
OUTPUT_FILE = "AlphaFilling_cofactors/late_stage_data_fixes/Q9X1L9_filled.pdb"

# Toggle for alignment strategy
USE_LOCAL_ALIGNMENT = False  # Set to False to use full-backbone alignment

# Only used if USE_LOCAL_ALIGNMENT is True
DONOR_SELECTION = "resi 9+12+15+19+38+41+50+54"
TARGET_SELECTION = "resi 9+12+15+19+38+41+50+54"

# Cofactor residue names to extract (modify as needed)
COFACTOR_RESN = ["SF4"]  # ["FES", "SF4", "FE", "FE2", "F3S"]

# Log file path
LOG_FILE = "AlphaFilling_cofactors/manual_transplant_log.csv"

# === Start PyMOL
cmd.reinitialize()
cmd.load(DONOR_FILE, "donor")
cmd.load(TARGET_FILE, "target")

alignment_type = "Local" if USE_LOCAL_ALIGNMENT else "Global"
status = "Started"
rmsd = "N/A"
cofactor_list = []

try:
    if USE_LOCAL_ALIGNMENT:
        rmsd = cmd.align(f"donor and ({DONOR_SELECTION}) and name CA",
                         f"target and ({TARGET_SELECTION}) and name CA")[0]
    else:
        rmsd = cmd.align("donor and name CA", "target and name CA")[0]
    print(f"✅ Alignment RMSD: {rmsd:.2f} Å")
except Exception as e:
    status = f"Alignment failed: {e}"
    print(f"❌ {status}")
    rmsd = "N/A"

# === Extract cofactors
if status == "Started":
    cofactor_sel = " or ".join([f"resn {r}" for r in COFACTOR_RESN])
    cmd.select("cofactors", f"donor and hetatm and ({cofactor_sel})")
    if cmd.count_atoms("cofactors") == 0:
        status = "No cofactors found"
        print("⚠️ No cofactors found for transfer.")
    else:
        cmd.create("merged", "target")
        cmd.create("to_merge", "cofactors")
        cmd.create("merged", "merged or to_merge")
        cmd.save(OUTPUT_FILE, "merged")
        status = "Success"
        cofactor_list = list({atom.resn for atom in cmd.get_model('cofactors').atom})
        print(f"💾 Transplanted structure saved to: {OUTPUT_FILE}")

# === Logging
# Prepare log row
log_fields = [
    "Timestamp", "Donor_File", "Target_File", "Output_File",
    "Alignment_Type", "RMSD", "Cofactors", "Status"
]

log_row = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    os.path.basename(DONOR_FILE),
    os.path.basename(TARGET_FILE),
    os.path.basename(OUTPUT_FILE),
    alignment_type,
    rmsd,
    ",".join(cofactor_list) if cofactor_list else "None",
    status
]

# Write to CSV
file_exists = os.path.exists(LOG_FILE)
with open(LOG_FILE, "a", newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(log_fields)
    writer.writerow(log_row)

print(f"\n📝 Log entry written to: {LOG_FILE}")


cmd.quit()