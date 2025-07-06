#!/bin/bash

# Run this script with bash Energy_minimization/assign_protonation.sh
# make sure you conda activate protonation-env first

# === Config ===
EXCEL_FILE="FeS_datasets/complete_FeS_dataset_with_AF_mutations.xlsx"
INPUT_FOLDER="FeS_mutant_structures"
OUTPUT_FOLDER="protonated_structures"
CSV_FILE="temp_dataset.csv"
LOG_FILE="Energy_minimization/protonation_log.csv"
ENV_NAME="protonation-env"

mkdir -p "$OUTPUT_FOLDER"

# Set this to "true" to skip existing .pqr files, "false" to reprocess all
SKIP_EXISTING="true"

# === Convert Excel to CSV ===
in2csv "$EXCEL_FILE" > "$CSV_FILE"  # Requires csvkit

# === Initialize log ===
echo "UniProt ID,Mutation,pH,Status,Message" > "$LOG_FILE"

# === Loop over CSV ===
csvcut -c UniProt,Mutation_AF,pH "$CSV_FILE" | tail -n +2 | \
while IFS=, read -r uniprot_id mutation pH; do
    if [[ "$mutation" == "Mutation_AF" || "$mutation" == "no" ]]; then
        continue
    fi

    pdb_in="${INPUT_FOLDER}/${uniprot_id}_${mutation}_Structure.pdb"
    pqr_out="${OUTPUT_FOLDER}/${uniprot_id}_${mutation}_Structure.pqr"

    echo "Looking for file: $pdb_in"

    if [[ ! -f "$pdb_in" ]]; then
        echo "$uniprot_id,$mutation,$pH,Skipped,Missing PDB file" >> "$LOG_FILE"
        continue
    fi

    if [[ -z "$pH" ]]; then
        echo "$uniprot_id,$mutation,$pH,Skipped,Missing pH value" >> "$LOG_FILE"
        continue
    fi

    # Skip existing .pqr files if SKIP_EXISTING is true
    if [[ "$SKIP_EXISTING" == "true" && -f "$pqr_out" ]]; then
        echo "$uniprot_id,$mutation,$pH,Skipped,PQR file already exists" >> "$LOG_FILE"
        continue
    fi

    echo "Processing $pdb_in at pH $pH..."

    if mamba run -n "$ENV_NAME" pdb2pqr --ff=PARSE --with-ph="$pH" "$pdb_in" "$pqr_out"; then
        echo "$uniprot_id,$mutation,$pH,Success,Protonated and saved" >> "$LOG_FILE"
    else
        echo "$uniprot_id,$mutation,$pH,Failed,pdb2pqr failed" >> "$LOG_FILE"
    fi
done

# Optional cleanup
rm -f "$CSV_FILE"

echo "Protonation complete. Log written to $LOG_FILE. Output in $OUTPUT_FOLDER."