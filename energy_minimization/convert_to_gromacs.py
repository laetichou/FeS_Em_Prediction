import os
import parmed as pmd

''' To run:
cd Energy_minimization
mamba activate parmed_env
python convert_to_gromacs.py
'''

base_dir = "tleap_runs"

for name in os.listdir(base_dir):
    folder = os.path.join(base_dir, name)
    top_path = os.path.join(folder, f"{name}.top")
    crd_path = os.path.join(folder, f"{name}.crd")

    if not (os.path.exists(top_path) and os.path.exists(crd_path)):
        print(f"Missing files for {name}, skipping.")
        continue

    print(f"Converting {name} to GROMACS format...")

    try:
        amber = pmd.load_file(top_path, crd_path)
        amber.save(os.path.join(folder, f"{name}_gromacs.top"), format="gromacs")
        amber.save(os.path.join(folder, f"{name}.gro"), format="gro")
        print(f"Saved: {name}.gro and {name}_gromacs.top")
    except Exception as e:
        print(f"Failed to convert {name}: {e}")
