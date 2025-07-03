#!/bin/bash

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "This script must be run from the root directory of the project."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/build directory does not exist
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "The language_model/runtime/server/x86/build directory already exists. Please remove it before running this script."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/fc_base directory does not exist
if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "The language_model/runtime/server/x86/fc_base directory already exists. Please remove it before running this script."
    exit 1
fi

# make sure CMake is installed
if ! command -v cmake &> /dev/null; then
    echo "CMake is not installed. Please install CMake >= 3.14 before running this script with 'sudo apt-get install cmake'."
    exit 1
fi

# make sure gcc is installed
if ! command -v gcc &> /dev/null; then
    echo "GCC is not installed. Please install GCC >= 10.1 before running this script with 'sudo apt-get install build-essential'."
    exit 1
fi

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment with Python 3.9
conda create -n b2txt25_lm python=3.9 -y

# Activate the new environment
conda activate b2txt25_lm

# Upgrade pip
pip install --upgrade pip

# Install additional packages
pip install \
    torch==1.13.1 \
    redis==5.0.6 \
    jupyter==1.1.1 \
    numpy==1.24.4 \
    matplotlib==3.9.0 \
    scipy==1.11.1 \
    scikit-learn==1.6.1 \
    tqdm==4.66.4 \
    g2p_en==2.1.0 \
    omegaconf==2.3.0 \
    huggingface-hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.41.1

# cd to the language model directory and install the language model
cd language_model/runtime/server/x86
python setup.py install

# cd back to the root directory
cd ../../../..

echo
echo "Setup complete! Verify it worked by activating the conda environment with the command 'conda activate b2txt25_lm'."
echo
