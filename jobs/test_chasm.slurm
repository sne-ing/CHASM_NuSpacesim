#!/bin/bash
#SBATCH --job-name=test_chasm
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/test_chasm_%j.out
#SBATCH --error=logs/test_chasm_%j.err

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

cd /uufs/chpc.utah.edu/common/home/u1530851/CHASM_NuSpacesim

source /uufs/chpc.utah.edu/common/home/u1530851/vens/chasm/bin/activate

echo "Python path:"
which python
python --version

echo "Testing CHASM import:"
python - <<'PY'
import socket
import sys
import CHASM

print("Compute node:", socket.gethostname())
print("Python executable:", sys.executable)
print("CHASM module:", CHASM)
PY

echo "Job finished at: $(date)"
