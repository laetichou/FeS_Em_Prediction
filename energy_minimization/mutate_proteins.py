import pandas as pd
import os
from pymol import cmd, stored
from Bio.Data.IUPACData import protein_letters_1to3


'''
To run this file:
    Open Terminal
    Navigate to the BEP folder with this script using `cd /path/to/your/script`
    Run the script with PyMol using `pymol -cq Energy_minimization/mutate_proteins.py`

Note that this last command works only because I have added PyMol to my Terminal PATH using:
    `open -a TextEdit ~/.zshrc`
    Pasting this line at the bottom: `export PATH="/Applications/PyMOL.app/Contents/MacOS:$PATH"`
    Reloading the shell with: `source ~/.zshrc`
'''

# === File locations ===
excel_path = "FeS_datasets/complete_FeS_dataset_with_AF_mutations.xlsx"  # Your Excel dataset
structure_folder = "FeS_WildType_structures"  # Folder where WT structures reside
mutant_folder = "FeS_mutant_structures"
os.makedirs(mutant_folder, exist_ok=True)  # Make mutant structures folder
log_path = "Energy_minimization/mutation_log_pymol.csv"  # Output log file

# === Load Excel dataset ===
df = pd.read_excel(excel_path)

# === Initialize log list ===
log_data = []

# == Map single-letter to three-letter AA codes
def one_to_three(aa):
    try:
        return protein_letters_1to3[aa.upper()].capitalize()
    except KeyError:
        raise ValueError(f"Unknown amino acid code: {aa}")

# === Process each protein row ===
for index, row in df.iterrows():
    uniprot_id = row["UniProt"]
    mutation = row["Mutation_AF"]

    # Skip wild-type entries
    if mutation == "no" or pd.isna(mutation):
        log_data.append({"UniProt ID": uniprot_id, 
                         "Mutation": "no", 
                         "Status": "Skipped", 
                         "Message": "Wild-type entry"})
        continue

    try:
        # Set file paths
        mutation_clean = mutation.replace(",", "_").replace(" ", "")
        wt_file = os.path.join(structure_folder, f"{uniprot_id}_WT_Structure.pdb")
        mut_file = os.path.join(mutant_folder, f"{uniprot_id}_{mutation_clean}_Structure.pdb")

        if not os.path.exists(wt_file):
            log_data.append({"UniProt ID": uniprot_id, 
                             "Mutation": mutation, 
                             "Status": "Failed", 
                             "Message": "WT structure file not found"})
            continue

        # Load WT structure
        cmd.reinitialize()
        cmd.load(wt_file, "protein")

        # Split multiple mutations, they are separated by an underscore
        mutations = mutation.split("_")

        for mut in mutations:
            old_aa = mut[0]
            pos = int(mut[1:-1])
            new_aa = mut[-1]
            new_resname = one_to_three(new_aa).upper()

            selection = f"chain A and resi {pos}"
            stored.resn = ""
            cmd.iterate(selection, "stored.resn = resn")

            if stored.resn:
                expected_three = one_to_three(old_aa).upper()
                if stored.resn.upper() != expected_three:
                    message = f"Residue mismatch at pos {pos}: expected {expected_three}, found {stored.resn}"
                    raise ValueError(message)
            else:
                raise ValueError(f"Residue {pos} not found in chain A")

            cmd.wizard("mutagenesis")
            cmd.get_wizard().do_select(selection)
            cmd.get_wizard().set_mode(new_resname)
            cmd.get_wizard().apply()
            cmd.set_wizard()

        # Save the final multi-mutant structure
        print(f"Saving to: {mut_file}")
        cmd.save(mut_file, "protein")
        log_data.append({"UniProt ID": uniprot_id, "Mutation": mutation, 
                         "Status": "Success", "Message": f"{len(mutations)} mutations applied and saved"})

    except Exception as e:
        log_data.append({"UniProt ID": uniprot_id, "Mutation": mutation, 
                         "Status": "Failed", "Message": str(e)})

# === Write the log to a CSV file ===
log_df = pd.DataFrame(log_data)
log_df.to_csv(log_path, index=False)
print(f"Log saved to {log_path}")

# Quit PyMOL after processing
cmd.quit()
