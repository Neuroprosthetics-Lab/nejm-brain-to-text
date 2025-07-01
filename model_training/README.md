# Model Training

This directory contains code and resources for training the brain-to-text RNN model. This model is largely based on the architecture described in the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), but also contains modifications to improve performance, efficiency, and usability.

## Setup

1. Install the required conda environment by following the instructions in the root `README.md` file. This will set up the necessary dependencies for running the model training and evaluation code.

2. Download the dataset from Dryad: [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). Place the downloaded data in the `data` directory.

## Training

To train the baseline RNN model, run the following command:
```bash
python train_model.py
```

## Evaluation

To evaluate the model, run:
```bash
python evaluate_model.py
```