#!/bin/bash

# Initialize conda.
source /data/anasynth_nonbp/manaconda3/etc/profile.d/conda.sh

# Activate environment.
conda activate linares_env

echo "Starting training..."

# Run training script.
python runner.py --config test_configuration.yaml

echo "Training finished! :)"