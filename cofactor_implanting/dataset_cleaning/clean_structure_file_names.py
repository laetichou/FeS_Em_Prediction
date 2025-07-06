# # --- Renaming without overwriting ---
# import os

# folder_path = 'Prediction_FeS_EM/final_structure_dataset_crystal'
# seen = set()

# for filename in os.listdir(folder_path):
#     if filename.endswith('.pdb'):
#         new_name = filename.replace('_no_water', '').replace('_transplanted', '')

#         if new_name in seen:
#             print(f'⚠️  Conflict: {filename} -> {new_name} would overwrite a file.')
#         else:
#             seen.add(new_name)
#             old_file = os.path.join(folder_path, filename)
#             new_file = os.path.join(folder_path, new_name)
#             os.rename(old_file, new_file)
#             print(f'Renamed: {filename} -> {new_name}')


# --- Rename with overwriting ---
import os
import glob

# Set your folder path
folder_path = "Prediction_FeS_EM/final_structure_dataset_crystal_no-anisou"

# Change to that directory
os.chdir(folder_path)

# Find all _no_water.pdb files
for file in glob.glob("*_no_water.pdb"):
    # Create the new filename by removing _no_water
    new_name = file.replace("_no_water", "")
    
    # Remove the original file if it exists
    if os.path.exists(new_name):
        os.remove(new_name)
    
    # Rename the _no_water file to the original name
    os.rename(file, new_name)

print("Renaming and overwriting complete.")
