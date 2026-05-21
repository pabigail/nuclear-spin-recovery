#!/bin/bash
# =============================================================================
#  submit_sequential.slurm
#  Perlmutter (NERSC) – sequential mode
#
#  All 5 RJMCMC trajectories run one after another in a single job.
#  Output is written to $OUTPUT_DIR/results_combined.pkl.
#
#  Estimated wall time: ~2 h per trajectory × 5 = ~10 h.
#  Requesting 12 h gives a comfortable buffer.
#
#  Usage:
#    sbatch submit_sequential.slurm
#
#  To test quickly with shorter chains (change NUM_TRIALS in the script):
#    sbatch --qos=debug --time=00:30:00 submit_sequential.slurm
# =============================================================================

#SBATCH --job-name=rjmcmc_sequential
#SBATCH --account=YOUR_ACCOUNT          # ← replace with your NERSC project account
#SBATCH --constraint=cpu                # CPU nodes on Perlmutter
#SBATCH --qos=regular                   # use 'debug' for ≤30 min test runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # numpy/BLAS can use extra threads
#SBATCH --time=12:00:00                # HH:MM:SS  (increase if 25k steps takes longer)
#SBATCH --output=logs/sequential_%j.out
#SBATCH --error=logs/sequential_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@institution.edu   # ← replace with your email

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

# Create log directory if it does not exist
mkdir -p logs

# ── Option A: use a conda environment (recommended) ──────────────────────────
#   Replace 'my_rjmcmc_env' with your actual conda environment name.
module load conda
conda activate my_rjmcmc_env

# ── Option B: use a venv / pip install ───────────────────────────────────────
# module load python
# source /path/to/your/venv/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR="$SCRATCH/spin_bath_inference"   # working directory on Lustre scratch
OUTPUT_DIR="${WORK_DIR}/inference_results"

# Copy working files to scratch if not already there
# (edit the source path to match where your code lives)
mkdir -p "$WORK_DIR"
# rsync -av --update /path/to/your/code/ "$WORK_DIR/"

cd "$WORK_DIR" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $(hostname)"
echo "  Start    : $(date)"
echo "  Work dir : $WORK_DIR"
echo "  Output   : $OUTPUT_DIR"
echo "========================================================"

python run_cluster_inference.py \
    --mode sequential \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo "========================================================"
echo "  Finish   : $(date)"
echo "  Exit code: $EXIT_CODE"
echo "========================================================"

exit $EXIT_CODE
