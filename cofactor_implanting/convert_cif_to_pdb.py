import os
import gemmi

# === Paths ===
input_dir = "report_tests"
output_dir = "report_tests"
log_path = os.path.join(output_dir, "conversion_errors.txt")
os.makedirs(output_dir, exist_ok=True)

# === Start error log ===
with open(log_path, "w") as log:
    log.write("AlphaFill CIF to PDB Conversion Errors\n")
    log.write("="*50 + "\n")

# === Loop through .cif files ===
for filename in os.listdir(input_dir):
    if filename.endswith(".cif"):
        cif_path = os.path.join(input_dir, filename)
        pdb_filename = filename.replace(".cif", ".pdb")
        pdb_path = os.path.join(output_dir, pdb_filename)

        try:
            doc = gemmi.cif.read_file(cif_path)
            block = doc.sole_block()
            structure = gemmi.make_structure_from_block(block)
            structure.write_pdb(pdb_path)
            print(f"Converted: {filename} ➝ {pdb_filename}")
        except Exception as e:
            error_msg = f"Failed to convert {filename}: {e}"
            print(error_msg)
            with open(log_path, "a") as log:
                log.write(error_msg + "\n")

print("\n Conversion complete. Errors (if any) are logged in:")
print(f"   {log_path}")