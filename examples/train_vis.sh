#!/usr/bin/env bash

sbatch --job-name=imm \
        --output=data/logs_vis/out.txt \
        --error=data/logs_vis/err.txt \
        --nodes=1 \
        --gres=gpu:8 \
        --time=40:00:00 \
        --cpus-per-task 48 \
        --partition=learnfair \
        --wrap="srun python scripts/train.py --configs configs/paths/vis.yaml configs/experiments/vis-10pts.yaml --ngpus 8"