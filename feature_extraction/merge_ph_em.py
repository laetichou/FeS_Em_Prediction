import pandas as pd
import os
import glob

# --- Configuration ---
metadata_file = 'FeS_datasets/complete_FeS_dataset_with_cofactor_id.xlsx'
feature_dir = 'Prediction_FeS_EM/final_stretch/feature_extraction/output'  
output_dir = 'Prediction_FeS_EM/final_stretch/feature_extraction/output/merged_with_ph_em'
log_file = os.path.join(output_dir, 'dropped_rows_log.txt')

os.makedirs(output_dir, exist_ok=True)

# Load metadata
meta_df = pd.read_excel(metadata_file)[['cofactor_id', 'pH', 'Em']]  # dont forget to rename column from E0 to Em in _with_cofactor_id.xlsx file

# Find all relevant feature CSVs
csv_files = glob.glob(os.path.join(feature_dir, 'features_r*_final_*.csv'))

# Open log file
with open(log_file, 'w') as log:
    log.write("Dropped rows due to missing pH or Em values:\n\n")

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if 'cofactor_id' not in df.columns:
            raise ValueError(f"'cofactor_id' column not found in {file_path}")

        df['cofactor_id'] = df['cofactor_id']

        # Merge metadata
        df_merged = df.merge(meta_df, on='cofactor_id', how='left')

        # Identify and log missing values
        missing = df_merged[df_merged['pH'].isna() | df_merged['Em'].isna()]
        if not missing.empty:
            log.write(f"File: {os.path.basename(file_path)}\n")
            for cid in missing['cofactor_id']:
                log.write(f"  Missing data for: {cid}\n")
            log.write("\n")

        # Drop rows with missing pH or Em
        df_cleaned = df_merged.dropna(subset=['pH', 'Em'])

        # Save cleaned file
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_with_ph_em.csv'))
        df_cleaned.to_csv(output_path, index=False)

print(f"✅ Enriched {len(csv_files)} feature CSVs with pH and Em.")
print(f"📂 Output saved to: {output_dir}")