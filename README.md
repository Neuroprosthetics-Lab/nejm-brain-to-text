# An Accurate and Rapidly Calibrating Speech Neuroprosthesis
*The New England Journal of Medicine* (2024)

Nicholas S. Card, Maitreyee Wairagkar, Carrina Iacobacci,
Xianda Hou, Tyler Singer-Clark, Francis R. Willett,
Erin M. Kunz, Chaofei Fan, Maryam Vahdati Nia,
Darrel R. Deo, Aparna Srinivasan, Eun Young Choi,
Matthew F. Glasser, Leigh R. Hochberg,
Jaimie M. Henderson, Kiarash Shahlaie,
Sergey D. Stavisky*, and David M. Brandman*.

<span style="font-size:0.8em;">\* denotes co-senior authors</span>

![Speech neuroprosthesis overview](b2txt_methods_overview.png)

## Overview
This repository contains the code and data necessary to reproduce the results of the paper ["*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), *N Eng J Med*](https://www.nejm.org/doi/full/10.1056/NEJMoa2314132).

The code is written in Python, and the data can be downloaded from Dryad, [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.dncjsxm85). Please download this data and place it in the `data` directory before running the code. Be sure to unzip `t15_copyTask_neuralData.zip` and `t15_pretrained_rnn_baseline.zip`.

The code is organized into four main directories: `utils`, `analyses`, `data`, and `model_training`:
- The `utils` directory contains utility functions used throughout the code.
- The `analyses` directory contains the code necessary to reproduce results shown in the main text and supplemental appendix.
- The `data` directory contains the data necessary to reproduce the results in the paper. Download it from Dryad using the link above and place it in this directory.
- The `model_training` directory contains the code necessary to train and evaluate the brain-to-text model. See the README.md in that folder for more detailed instructions.
- The `language_model` directory contains the ngram language model implementation and a pretrained 1gram language model. Pretrained 3gram and 5gram language models can be downloaded [here](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) (`languageModel.tar.gz` and `languageModel_5gram.tar.gz`). See the `README.md` in this directory for more information.

## Dependencies
- The code has only been tested on Ubuntu 22.04 with two NVIDIA RTX 4090 GPUs.
- We recommend using a conda environment to manage the dependencies. To install miniconda, follow the instructions [here](https://docs.anaconda.com/miniconda/miniconda-install/).
- Redis is required for communication between python processes. To install redis on Ubuntu:
    - https://redis.io/docs/getting-started/installation/install-redis-on-linux/
    - In terminal:
        ```bash
        curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
        
        echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
        
        sudo apt-get update
        sudo apt-get install redis
        ```
    - Turn of autorestarting for the redis server in terminal:
        - `sudo systemctl disable redis-server`
- `CMake >= 3.14` and `gcc >= 10.1` are required for the ngram language model installation. You can install these on linux with `sudo apt-get install build-essential`.

## Python environment setup for model training and evaluation
To create a conda environment with the necessary dependencies, run the following command from the root directory of this repository:
```bash
./setup.sh
```

## Python environment setup for ngram language model and OPT rescoring
We use an ngram language model plus rescoring via the [Facebook OPT 6.7b](https://huggingface.co/facebook/opt-6.7b) LLM. A pretrained 1gram language model is included in this repository at `language_model/pretrained_language_models/openwebtext_1gram_lm_sil`. Pretrained 3gram and 5gram language models are available for download [here](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) (`languageModel.tar.gz` and `languageModel_5gram.tar.gz`). Note that the 3gram model requires ~60GB of RAM, and the 5gram model requires ~300GB of RAM. Furthermore, OPT 6.7b requires a GPU with at least ~12.4 GB of VRAM to load for inference.

Our Kaldi-based ngram implementation requires a different version of torch than our model training pipeline, so running the ngram language models requires an additional seperate python conda environment. To create this conda environment, run the following command from the root directory of this repository. For more detailed instructions, see the README.md in the `language_model` subdirectory.
```bash
./setup_lm.sh
```