import os
import subprocess

# to run from terminal: `python run_tleap_all.py`
# make sure you have mamba activated ambertools_env

base_dir = "Energy_minimization/tleap_runs"
env_name = "ambertools_env"

for name in os.listdir(base_dir):
    folder = os.path.join(base_dir, name)
    tleap_in = os.path.join(folder, "tleap.in")

    if not os.path.exists(tleap_in):
        continue

    print(f"Running tleap for {name}...")
    result = subprocess.run(
        ["tleap", "-f", "tleap.in"],
        cwd=folder,
        capture_output=True,
        text=True
    )

    # Save stdout/stderr
    with open(os.path.join(folder, "tleap.log"), "w") as log:
        log.write(result.stdout)
        log.write("\n--- stderr ---\n")
        log.write(result.stderr)

    if result.returncode == 0:
        print(f"tleap completed for {name}")
    else:
        print(f"tleap failed for {name}, see tleap.log")
