#!/bin/bash
# =============================================================================
#  submit_parallel.slurm
#  Perlmutter (NERSC) – parallel SLURM array mode
#
#  Submits two dependent jobs:
#
#    Job A  (array, --array=0-4):
#      5 independent tasks, one per trajectory.
#      Each task runs on its own CPU allocation and saves traj_N.pkl.
#      Estimated wall time per task: ~2 h (25k steps).
#
#    Job B  (merge, --dependency=afterok:$ARRAY_JOB_ID):
#      Runs only after ALL array tasks succeed.
#      Combines traj_N.pkl files into results_combined.pkl.
#
#  Usage:
#    bash submit_parallel.slurm
#      (this script calls sbatch internally and chains the dependency)
#
#  Or submit manually:
#    ARRAY_JOB_ID=$(sbatch --parsable submit_array.slurm)
#    sbatch --dependency=afterok:$ARRAY_JOB_ID submit_merge.slurm
#
#  To test with a single trajectory:
#    sbatch --array=0 --qos=debug --time=00:30:00 submit_parallel.slurm
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Shared settings (edit these)
# ─────────────────────────────────────────────────────────────────────────────

ACCOUNT="YOUR_ACCOUNT"                    # ← NERSC project account
EMAIL="YOUR_EMAIL@institution.edu"        # ← your email for notifications
WORK_DIR="$SCRATCH/spin_bath_inference"   # ← working directory on Lustre scratch
OUTPUT_DIR="${WORK_DIR}/inference_results"
CONDA_ENV="my_rjmcmc_env"               # ← your conda environment name

# ─────────────────────────────────────────────────────────────────────────────
# This script submits Job A (array) and Job B (merge) in one shot.
# Run it with:  bash submit_parallel.slurm
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$WORK_DIR/logs"

# ── Job A: array of 5 trajectory tasks ───────────────────────────────────────
ARRAY_JOB_ID=$(sbatch --parsable \
    --job-name=rjmcmc_array \
    --account="$ACCOUNT" \
    --constraint=cpu \
    --qos=regular \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time=04:00:00 \
    --array=0-4 \
    --output="${WORK_DIR}/logs/array_%A_%a.out" \
    --error="${WORK_DIR}/logs/array_%A_%a.err" \
    --mail-type=FAIL \
    --mail-user="$EMAIL" \
    --wrap="
        set -euo pipefail
        module load conda
        conda activate ${CONDA_ENV}
        cd ${WORK_DIR}
        echo '--- Array task \$SLURM_ARRAY_TASK_ID started on \$(hostname) at \$(date) ---'
        python run_cluster_inference.py \
            --mode single \
            --trajectory-id \$SLURM_ARRAY_TASK_ID \
            --output-dir ${OUTPUT_DIR}
        echo '--- Array task \$SLURM_ARRAY_TASK_ID finished at \$(date) ---'
    "
)

if [[ -z "$ARRAY_JOB_ID" ]]; then
    echo "ERROR: failed to submit array job."
    exit 1
fi
echo "Submitted array job: $ARRAY_JOB_ID  (tasks 0–4)"

# ── Job B: merge – runs only after all array tasks succeed ───────────────────
MERGE_JOB_ID=$(sbatch --parsable \
    --job-name=rjmcmc_merge \
    --account="$ACCOUNT" \
    --constraint=cpu \
    --qos=regular \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time=00:15:00 \
    --dependency=afterok:"$ARRAY_JOB_ID" \
    --output="${WORK_DIR}/logs/merge_%j.out" \
    --error="${WORK_DIR}/logs/merge_%j.err" \
    --mail-type=END,FAIL \
    --mail-user="$EMAIL" \
    --wrap="
        set -euo pipefail
        module load conda
        conda activate ${CONDA_ENV}
        cd ${WORK_DIR}
        echo '--- Merge job started on \$(hostname) at \$(date) ---'
        python run_cluster_inference.py \
            --mode merge \
            --output-dir ${OUTPUT_DIR}
        echo '--- Merge job finished at \$(date) ---'
        echo 'Combined output: ${OUTPUT_DIR}/results_combined.pkl'
    "
)

if [[ -z "$MERGE_JOB_ID" ]]; then
    echo "ERROR: failed to submit merge job."
    exit 1
fi
echo "Submitted merge job:  $MERGE_JOB_ID  (depends on afterok:$ARRAY_JOB_ID)"
echo ""
echo "Track progress:"
echo "  squeue -u \$USER"
echo "  tail -f ${WORK_DIR}/logs/array_${ARRAY_JOB_ID}_0.out"
echo ""
echo "Output will be written to:"
echo "  ${OUTPUT_DIR}/results_combined.pkl"


# =============================================================================
#  ALTERNATIVE: single-node multiprocessing (5 workers on one node)
#  Uncomment the block below and comment out the array/merge blocks above
#  if you prefer one job with 5 parallel Python processes on the same node.
# =============================================================================
#
# sbatch \
#     --job-name=rjmcmc_mp \
#     --account="$ACCOUNT" \
#     --constraint=cpu \
#     --qos=regular \
#     --nodes=1 \
#     --ntasks=1 \
#     --cpus-per-task=40 \
#     --time=04:00:00 \
#     --output="${WORK_DIR}/logs/mp_%j.out" \
#     --error="${WORK_DIR}/logs/mp_%j.err" \
#     --mail-type=END,FAIL \
#     --mail-user="$EMAIL" \
#     --wrap="
#         set -euo pipefail
#         module load conda
#         conda activate ${CONDA_ENV}
#         cd ${WORK_DIR}
#         echo '--- Multiprocessing job started on \$(hostname) at \$(date) ---'
#         python run_cluster_inference.py \
#             --mode multiprocessing \
#             --output-dir ${OUTPUT_DIR}
#         echo '--- Finished at \$(date) ---'
#     "
