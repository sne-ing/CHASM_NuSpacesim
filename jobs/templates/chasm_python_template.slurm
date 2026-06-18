#!/bin/bash
#SBATCH --job-name=chasm_job
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory before cd: $(pwd)"

cd /uufs/chpc.utah.edu/common/home/u1530851/CHASM_NuSpacesim

source /uufs/chpc.utah.edu/common/home/u1530851/vens/chasm/bin/activate

echo "Python:"
which python
python --version

echo "Running script:"
python path/to/your_script.py

echo "Job finished at: $(date)"
