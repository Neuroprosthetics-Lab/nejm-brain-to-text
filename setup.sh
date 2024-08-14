#!/bin/bash
source ~/miniconda3/bin/activate

# Create a new conda environment with the name "tf-gpu" and Python 3.9
conda create -n tf-gpu python=3.9 -y

# Activate the conda environment
conda activate tf-gpu

# Install tensorflow-gpu version 2.10.0
conda install -c conda-forge tensorflow-gpu=2.10.0 -y

# Install numpy, scikit-learn
conda install numpy==1.26.0 scikit-learn==1.3.0 -y

# Install omegaconf, pyyaml, redis, matplotlib, jupyter, transformers, g2p_en
pip install omegaconf==2.3.0 pyyaml==6.0.1 redis==5.0.1 matplotlib==3.8.1 jupyter==1.0.0 transformers==4.35.0 g2p_en==2.1.0 coloredlogs==15.0.1 numba==0.58.1

# install punctuation model
pip install deepmultilingualpunctuation==1.0.1

# install local repository
pip install -e .

# install lm-decoder
cd LanguageModelDecoder/runtime/server/x86
python setup.py install