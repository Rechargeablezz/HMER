#!/bin/bash
# JSUB -q gpu
# JSUB -gpgpu 7
# JSUB -n 4
# JSUB -e jsub_logs/error.%J
# JSUB -o jsub_logs/output.%J
source /apps/software/anaconda3/etc/profile.d/conda.sh

conda activate can
unset PYTHONPATH

python train.py --dataset CROHME2