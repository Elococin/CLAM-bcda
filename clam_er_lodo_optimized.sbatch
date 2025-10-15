#!/bin/bash
#SBATCH -A qtim
#SBATCH -p rtx8000
#SBATCH -o /autofs/space/crater_001/projects/bc_domain_adaptation_nicole/logs/%x-%j.out
#SBATCH -e /autofs/space/crater_001/projects/bc_domain_adaptation_nicole/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4    # 4 CPUs per 1 GPU
#SBATCH --mem=16G            # Appropriate for 4 CPUs
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

set -euo pipefail
JOB_NAME="clam_er_${HELDOUT}_fullrun"
echo "Setting job name to ${JOB_NAME}"
echo "===== Starting CLAM ER ${HELDOUT}-heldout Full Run ====="
echo "Host: $(hostname) | JobID: ${SLURM_JOB_ID}"
echo "Resource allocation: 1 GPU, 4 CPUs, 16GB RAM"
date

# --- Env setup ---
export PYENV_ROOT=/autofs/space/crater_001/tools/pyenv
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$($PYENV_ROOT/bin/pyenv init -)" || true
eval "$($PYENV_ROOT/bin/pyenv virtualenv-init -)" || true
pyenv activate bcda-312 || export PYTHON=/autofs/space/crater_001/tools/pyenv/versions/bcda-312/bin/python
PYTHON=${PYTHON:-$(which python)}
echo "Using Python: $PYTHON"
$PYTHON --version

# --- Define directories dynamically ---
RESULTS_DIR=/autofs/space/crater_001/projects/bc_domain_adaptation_nicole/results/clam_sb_baseline/lodo_${HELDOUT}_fullrun
SPLIT_DIR=/autofs/space/crater_001/projects/bc_domain_adaptation_nicole/splits/er_lodo/${HELDOUT}_heldout

# --- Create results directory ---
mkdir -p ${RESULTS_DIR}

# --- GPU monitoring setup ---
echo "=== Starting GPU monitoring ==="
nvidia-smi
nvidia-smi dmon -s pucvmet -d 60 > ${RESULTS_DIR}/gpu_utilization.log &
GPU_MON_PID=$!

# --- Repo setup ---
export PYTHONPATH=/homes/9/jy1101/CLAM-bcda:${PYTHONPATH:-}
cd /homes/9/jy1101/CLAM-bcda || exit 1

# --- Run CLAM with optimized parameters ---
echo "=== Starting CLAM training with GPU optimization ==="
echo "Model: big, embed_dim: 768 (CONCH), B: 64"
PYTHONUNBUFFERED=1 $PYTHON main.py \
  --config /homes/9/jy1101/CLAM-bcda/clam_er_optimized.yaml \
  --task ER_status \
  --model_size big \
  --embed_dim 768 \
  --B 64 \
  --bag_weight 0.7 \
  --split_dir ${SPLIT_DIR} \
  --results_dir ${RESULTS_DIR} \
  --drop_out 0.25 \
  --early_stopping

# --- Stop monitoring ---
kill $GPU_MON_PID 2>/dev/null || true

# --- GPU utilization summary ---
echo "=== GPU Utilization Summary ==="
if [ -f ${RESULTS_DIR}/gpu_utilization.log ]; then
    echo "Average GPU utilization:"
    tail -20 ${RESULTS_DIR}/gpu_utilization.log
else
    echo "GPU log not found"
fi

date
echo "===== Job Finished ====="

# --- Auto-log run ---
python3 /autofs/space/crater_001/projects/bc_domain_adaptation_nicole/scripts/append_experiment_log.py \
  /autofs/space/crater_001/projects/bc_domain_adaptation_nicole/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \
  ${HELDOUT} \
  ${RESULTS_DIR} \
  ${SPLIT_DIR} \
  "Optimized ${HELDOUT} run - 4CPU/1GPU"
echo "===== Logged run to experiment log ====="
