#!/bin/bash

# Create a conda env called "ml":
mamba create --prefix ~/scratch/ml -y python=3.10
mamba activate ~/scratch/ml

# Install PyTorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia

# Install some other usual libraries
mamba install -y -c conda-forge numpy matplotlib scipy pandas \
    scikit-learn ipykernel jupyter notebook pytest

# Install libraries for LLM fine-tuning
mamba install -y accelerate peft bitsandbytes transformers trl

# install sentence-piece
mamba install -y -c conda-forge sentencepiece

# Install libraries for clustering


# Install utils
pip install -e .

