#!/usr/bin/env bash
# Initialize Conda for bash (so conda commands work in scripts)
eval "$(conda shell.bash hook)"

# Activate the environment named "cuda-env"
conda activate cuda-env

# Now you can run commands that depend on cuda-env, for example:
python --version