import os
from pymol import cmd
import re

'''Run with:
source ~/.zshrc
cd BEP
pymol -cq AlphaFilling_cofactors/transplant_cofactors_from_crystal.py
'''

# === Directories ===
af_dir = "AlphaFilling_cofactors/alphafill_pdb"
donor_dir = "AlphaFilling_cofactors/crystal_donors_pdb"
output_dir = "AlphaFilling_cofactors/alphafilled_wild_types"
log_path = "AlphaFilling_cofactors/alphafilled_wild_types/transplant_logs.txt"
os.makedirs(output_dir, exist_ok=True)

# === Start fresh
cmd.reinitialize()

with open(log_path, "w") as log:
    log.write("UniProt_ID\tDonor_Type\tCofactors\tRMSD\tStatus\n")

# === Loop over AlphaFill files
for filename in os.listdir(af_dir):
    if filename.endswith("_empty.pdb"):
        uniprot_id = filename.replace("_empty.pdb", "")
        af_path = os.path.join(af_dir, filename)

        # Try donor from crystal or NMR
        donor_types = ["crystal", "nmr"]
        donor_loaded = False
        for dtype in donor_types:
            donor_filename = f"{uniprot_id}_{dtype}.pdb"
            donor_path = os.path.join(donor_dir, donor_filename)
            if os.path.exists(donor_path):
                donor_loaded = True
                break

        if not donor_loaded:
            print(f"⚠️ No donor found for {uniprot_id}")
            with open(log_path, "a") as log:
                log.write(f"{uniprot_id}\tNone\tNone\tNone\tNo donor found\n")
            continue

        # Load structures
        cmd.load(af_path, "target")
        cmd.load(donor_path, "donor")

        # Align full backbone
        try:
            rmsd = cmd.align("donor and name CA", "target and name CA")[0]
        except Exception as e:
            print(f"❌ Alignment failed for {uniprot_id}: {e}")
            with open(log_path, "a") as log:
                log.write(f"{uniprot_id}\t{dtype}\tNone\tNone\tAlignment failed\n")
            cmd.delete("all")
            continue

        # Select and copy cofactors from donor
        cofactors = []
        cmd.select("ligands", "donor and hetatm and not resn HOH")
        model = cmd.get_model("ligands")
        if not model.atom:
            print(f"⚠️ No cofactors found in {uniprot_id} donor")
            with open(log_path, "a") as log:
                log.write(f"{uniprot_id}\t{dtype}\tNone\t{rmsd:.2f}\tNo cofactors found\n")
            cmd.delete("all")
            continue

        for atom in model.atom:
            cofactors.append(atom.resn.strip())

        cofactor_list = ",".join(sorted(set(cofactors)))

        # Save final merged model
        merged_name = f"{uniprot_id}_transplanted"
        output_path = os.path.join(output_dir, f"{merged_name}.pdb")

        cmd.create(merged_name, "target")
        cmd.create("to_merge", "ligands")
        cmd.create(merged_name, f"{merged_name} or to_merge")
        cmd.save(output_path, merged_name)

        print(f"✅ Transplanted: {uniprot_id} ({cofactor_list}) RMSD={rmsd:.2f}")
        with open(log_path, "a") as log:
            log.write(f"{uniprot_id}\t{dtype}\t{cofactor_list}\t{rmsd:.2f}\tSuccess\n")

        cmd.delete("all")
