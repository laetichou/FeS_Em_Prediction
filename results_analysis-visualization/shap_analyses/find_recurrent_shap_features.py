import pandas as pd
import os
from collections import Counter

# Configuration
DATASET_TYPE = "SF4"  # Dataset type (e.g., SF4, FES)
SUBSET = "all_features"  # Subset type (e.g., all_features, bar_features, protein_features)
SHAP_DIR = "shap_files"  # Directory with SHAP CSV files
OUTPUT_DIR = "output"  # Output directory for CSV file
TOP_N_FEATURES = 10  # Number of top features to consider
MIN_COMBINATIONS = 3  # Minimum number of combinations for recurrence

# Feature descriptions (update as needed)
FEATURE_DESCRIPTIONS = {
    "seq.CTDC__HydrophobicityC3": "Composition Transition Distribution of hydrophobicity (C3)",
    "struct.is_hipip": "Indicator for HiPIP structure",
    "struct.burial_depth": "Depth of burial of the protein structure",
    "Protein.mean.P(sheet)": "Average sheet propensity of protein residues",
    "Protein.mean.P(helix) x Flex.": "Product of helix propensity and flexibility of protein residues",
    "pH": "Acidity level of the environment",
    "Bar.prop.Hydrophobicity": "Proportion of hydrophobicity of bar residues",
}

def load_shap_files(directory, subset):
    """Load SHAP CSV files and return a list of (model_name, dataframe) tuples."""
    files = [f for f in os.listdir(directory) if f.startswith(f"feature_importance_{subset}")]
    dataframes = []
    for file in files:
        model_name = file.replace(f"feature_importance_{subset}_", "").replace(".csv", "")
        df = pd.read_csv(os.path.join(directory, file))
        dataframes.append((model_name, df))
    return dataframes

def get_top_features(df, n):
    """Extract top N features from a dataframe sorted by importance."""
    return df.sort_values(by="importance", ascending=False).head(n)["feature"].tolist()

def find_recurrent_features(dataframes, top_n, min_combinations):
    """Find features that appear in at least min_combinations of the top N features."""
    all_top_features = []
    for _, df in dataframes:
        top_features = get_top_features(df, top_n)
        all_top_features.extend(top_features)
    
    feature_counts = Counter(all_top_features)
    recurrent_features = [feat for feat, count in feature_counts.items() if count >= min_combinations]
    return sorted(recurrent_features)

def generate_csv_table(recurrent_features, dataset_type, subset):
    """Generate CSV table for recurrent features."""
    data = []
    for feature in recurrent_features:
        description = FEATURE_DESCRIPTIONS.get(feature, "Description not provided")
        all_features = "✓" if subset == "all_features" else ""
        bar_features = "" if subset != "only_bar_features" else "✓"
        protein_features = "" if subset != "only_protein_features" else "✓"
        data.append({
            "Feature": feature,
            "Description": description,
            "All Features": all_features,
            "Only Bar Features": bar_features,
            "Only Protein Features": protein_features
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load SHAP files
    dataframes = load_shap_files(SHAP_DIR, SUBSET)
    if len(dataframes) < MIN_COMBINATIONS:
        raise ValueError(f"Need at least {MIN_COMBINATIONS} CSV files for recurrence analysis.")
    
    # Find recurrent features
    recurrent_features = find_recurrent_features(dataframes, TOP_N_FEATURES, MIN_COMBINATIONS)
    
    # Generate CSV table
    csv_df = generate_csv_table(recurrent_features, DATASET_TYPE, SUBSET)
    
    # Save to CSV file
    output_file = os.path.join(OUTPUT_DIR, f"{DATASET_TYPE.lower()}_table.csv")
    csv_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
