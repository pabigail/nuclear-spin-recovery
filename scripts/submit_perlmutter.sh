#!/bin/bash
#SBATCH -A m5305
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --array=0-19
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -D /global/cfs/cdirs/m5305/pabigail/claude-rewrite
#SBATCH -J nsr-ensemble
#SBATCH -o logs/%A_%a.out

# One ensemble per task. `shared` rather than `regular`: a single ensemble is
# minutes on one core, and a CPU node has 128 -- an exclusive allocation would
# idle 127 of them per task.
set -euo pipefail

srun -n 1 python scripts/run_ensemble.py \
    --config configs/nv_ensemble.toml \
    --ensemble "$SLURM_ARRAY_TASK_ID" \
    --out runs/"$SLURM_ARRAY_JOB_ID"

# When the array completes, pool it:
#   python scripts/merge_ensembles.py --config configs/nv_ensemble.toml \
#       --out runs/<jobid>
