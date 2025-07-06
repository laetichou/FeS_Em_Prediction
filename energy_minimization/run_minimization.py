import os
import subprocess
import shutil

'''To run:
cd Energy_minimization
mamba activate gromacs_env
python3 run_minimization.py
'''

base_dir = "tleap_runs"
minim_mdp = "Optimization_step_files/minim.mdp"

# Check that the MDP file exists
if not os.path.exists(minim_mdp):
    raise FileNotFoundError(f"Could not find minimization MDP file at {minim_mdp}")

for name in os.listdir(base_dir):
    folder = os.path.join(base_dir, name)
    gro_path = os.path.join(folder, f"{name}.gro")
    top_path = os.path.join(folder, f"{name}_gromacs.top")
    mdp_dst = os.path.join(folder, "minim.mdp")

    if not (os.path.exists(gro_path) and os.path.exists(top_path)):
        print(f"Missing GRO or TOP file for {name}, skipping.")
        continue

    # Copy MDP into folder (if needed)
    if not os.path.exists(mdp_dst):
        shutil.copy(minim_mdp, mdp_dst)

    print(f"Running energy minimization for {name}...")

    # File name prefix for GROMACS output
    prefix = f"{name}_minim"

    # Run grompp
    grompp = subprocess.run(
        ["gmx", "grompp", "-f", "minim.mdp", "-c", f"{name}.gro", "-p", f"{name}_gromacs.top", "-o", f"{prefix}.tpr"],
        cwd=folder,
        capture_output=True,
        text=True
    )
    with open(os.path.join(folder, f"{prefix}_grompp.log"), "w") as log:
        log.write(grompp.stdout)
        log.write("\n--- stderr ---\n")
        log.write(grompp.stderr)

    if grompp.returncode != 0:
        print(f"grompp failed for {name}, check {prefix}_grompp.log")
        continue

    # Run mdrun
    mdrun = subprocess.run(
        ["gmx", "mdrun", "-v", "-deffnm", prefix],
        cwd=folder,
        capture_output=True,
        text=True
    )
    with open(os.path.join(folder, f"{prefix}_mdrun.log"), "w") as log:
        log.write(mdrun.stdout)
        log.write("\n--- stderr ---\n")
        log.write(mdrun.stderr)

    if mdrun.returncode == 0:
        print(f"Minimization complete for {name}")
    else:
        print(f"Minimization failed for {name}, check {prefix}_mdrun.log")
