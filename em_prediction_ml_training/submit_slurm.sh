#!/bin/bash
#SBATCH --job-name=FeS_ml_all_MAE_list
#SBATCH --account=research-as-bn
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=36:00:00
#SBATCH --output=FeS_ml_all_MAE_list_%j.out
#SBATCH --error=FeS_ml_all_MAE_list_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Load required modules (adjust for your system)
module load 2023r1  # Load the software stack
module load python/3.9
module load miniconda3

# Initialize conda and activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate redox-env

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Define dataset configurations
declare -a DATASETS=(
    "complete_dataset" 
    "all_cofactors_protein"
    "all_cofactors_bar"
    "FES_all"
    "FES_protein"
    "FES_bar"
    "SF4_all"
    "SF4_protein"
    "SF4_bar"
)

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
echo "Created results directory: ${RESULTS_DIR}"

# Verify required files and directories exist
echo "Checking required files and directories..."
for dataset in "${DATASETS[@]}"; do
    if [ ! -d "data/${dataset}" ]; then
        echo "ERROR: Dataset directory not found: data/${dataset}"
        exit 1
    fi
done

# Check for ML training script only
if [ ! -f "scripts/ml_training_MAE_list.py" ]; then
    echo "ERROR: Required script not found: scripts/ml_training_MAE_list.py"
    exit 1
fi

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=== Processing dataset: $dataset ==="
    echo "Started at: $(date)"
    
    # Create dataset results directory
    dataset_dir="${RESULTS_DIR}/${dataset}"
    mkdir -p "${dataset_dir}"
    
    # Determine which ML training script to use based on dataset type
    if [[ $dataset == *"_protein"* ]]; then
        # Use protein-specific script for protein datasets
        python scripts/ml_training_protein_MAE_list.py \
            --data_dir "data/${dataset}" \
            --output_dir "${dataset_dir}" \
            --target_column "Em" \
            --n_repeats 10 \
            --log_level INFO
    else
        # Use radius-dependent script for other datasets
        python scripts/ml_training_MAE_list.py \
            --data_dir "data/${dataset}" \
            --output_dir "${dataset_dir}" \
            --target_column "Em" \
            --n_repeats 10 \
            --log_level INFO
    fi
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed ML training for $dataset"
        
        # Compress dataset results
        tar -czf "${dataset_dir}.tar.gz" -C "${RESULTS_DIR}" "${dataset}"
        echo "Results compressed to ${dataset_dir}.tar.gz"
    else
        echo "ERROR: ML training failed for $dataset"
    fi
    
    echo "Finished at: $(date)"
    echo "----------------------------------------"
done

# Compress entire run results
tar -czf "${RESULTS_DIR}.tar.gz" "${RESULTS_DIR}"
echo "Complete run results compressed to: ${RESULTS_DIR}.tar.gz"

# Create final summary
echo ""
echo "=== Final Summary ==="
echo "ML training completed at: $(date)"
echo "Results are in: ${RESULTS_DIR}"
echo "Compressed results: ${RESULTS_DIR}.tar.gz"

# Clean up
conda deactivate