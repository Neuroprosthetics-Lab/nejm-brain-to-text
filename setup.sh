#!/bin/bash

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment with Python 3.10
conda create -n b2txt25 python=3.10 -y

# Activate the new environment
conda activate b2txt25

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install additional packages
# TODO: remove redis
pip install \
    redis==5.2.1 \
    jupyter==1.1.1 \
    numpy==2.1.2 \
    pandas==2.3.0 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    tqdm==4.67.1 \
    g2p_en==2.1.0 \
    h5py==3.13.0 \
    omegaconf==2.3.0 \
    editdistance==0.8.1 \
    -e . \
    huggingface-hub==0.33.1 \
    transformers==4.53.0 \
    tokenizers==0.21.2 \
    accelerate==1.8.1 \
    bitsandbytes==0.46.0

echo
echo "Setup complete! Verify it worked by activating the conda environment with the command 'conda activate b2txt25'."
echo
