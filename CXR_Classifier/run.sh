#!/bin/bash

# Train model
echo Commencing model training..
python src/main.py "$@"

# Fine-tune
# echo Commencing fine-tuning..
# python src/main.py \
#   fine_tune=True \
#   load_checkpoint="checkpoints/model.pt" \
#   optimizer.lr=0.0001
