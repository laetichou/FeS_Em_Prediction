import pandas as pd
import numpy as np
import os

# Load the CSV file
csv_path = "AlphaFilling_cofactors/manual_transplant_log.csv"
df = pd.read_csv(csv_path)

# Convert RMSD column to numeric, non-numeric values become NaN
df['RMSD'] = pd.to_numeric(df['RMSD'], errors='coerce')

# Remove NaN values (entries where RMSD failed or wasn't calculated)
rmsd_values = df['RMSD'].dropna().values

# Identify outlier (RMSD > 10)
outlier_threshold = 3
normal_values = rmsd_values[rmsd_values <= outlier_threshold]
outliers = rmsd_values[rmsd_values > outlier_threshold]

# Calculate statistics with all values
mean_all = np.mean(rmsd_values)
std_all = np.std(rmsd_values)
median_all = np.median(rmsd_values)

# Calculate statistics without outliers
mean_normal = np.mean(normal_values)
std_normal = np.std(normal_values)
median_normal = np.median(normal_values)

# Create output directory if it doesn't exist
output_dir = "report_tests/results"
os.makedirs(output_dir, exist_ok=True)

# Save to text file
with open(f"{output_dir}/rmsd_statistics.txt", "w") as f:
    f.write("=== RMSD Statistics (Å) ===\n")
    
    f.write(f"\nWith all values (n={len(rmsd_values)}):\n")
    f.write(f"Mean: {mean_all:.3f}\n")
    f.write(f"Std Dev: {std_all:.3f}\n")
    f.write(f"Median: {median_all:.3f}\n")
    f.write(f"Range: {np.min(rmsd_values):.3f} - {np.max(rmsd_values):.3f}\n")
    
    f.write(f"\nWithout outliers > {outlier_threshold}Å (n={len(normal_values)}):\n")
    f.write(f"Mean: {mean_normal:.3f}\n")
    f.write(f"Std Dev: {std_normal:.3f}\n")
    f.write(f"Median: {median_normal:.3f}\n")
    f.write(f"Range: {np.min(normal_values):.3f} - {np.max(normal_values):.3f}\n")
    
    if len(outliers) > 0:
        f.write(f"\nOutliers removed: {len(outliers)}\n")
        for outlier in outliers:
            f.write(f"  - {outlier:.3f}Å\n")

# Save to CSV file
results_dict = {
    'Statistic': ['Count', 'Mean', 'Std Dev', 'Median', 'Min', 'Max'],
    'With Outliers': [
        len(rmsd_values),
        mean_all,
        std_all,
        median_all,
        np.min(rmsd_values),
        np.max(rmsd_values)
    ],
    'Without Outliers': [
        len(normal_values),
        mean_normal,
        std_normal,
        median_normal,
        np.min(normal_values),
        np.max(normal_values)
    ]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv(f"{output_dir}/rmsd_statistics.csv", index=False)

# Also print to console
print(f"Results saved to {output_dir}/rmsd_statistics.txt and {output_dir}/rmsd_statistics.csv")
print("\n=== RMSD Statistics (Å) ===")
print(f"\nWith all values (n={len(rmsd_values)}):")
print(f"Mean: {mean_all:.3f}")
print(f"Std Dev: {std_all:.3f}")
print(f"Median: {median_all:.3f}")
print(f"Range: {np.min(rmsd_values):.3f} - {np.max(rmsd_values):.3f}")

print(f"\nWithout outliers > {outlier_threshold}Å (n={len(normal_values)}):")
print(f"Mean: {mean_normal:.3f}")
print(f"Std Dev: {std_normal:.3f}")
print(f"Median: {median_normal:.3f}")
print(f"Range: {np.min(normal_values):.3f} - {np.max(normal_values):.3f}")

if len(outliers) > 0:
    print(f"\nOutliers removed: {len(outliers)}")
    for outlier in outliers:
        print(f"  - {outlier:.3f}Å")