# Model Training & Evaluation
This directory contains code and resources for training the brain-to-text RNN model. This model is largely based on the architecture described in the paper "*An Accurate and Rapidly Calibrating Speech Neuroprosthesis*" by Card et al. (2024), but also contains modifications to improve performance, efficiency, and usability.

A pretrained baseline RNN model is included in the [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85), as is the neural data required to train that model. The code for training the same model is included here.

All model training and evaluation code was tested on a computer running Ubuntu 22.04 with two RTX 4090's and 512 GB of RAM.

## Setup
1. Install the required `b2txt25` conda environment by following the instructions in the root `README.md` file. This will set up the necessary dependencies for running the model training and evaluation code.

2. Download the dataset from Dryad: [Dryad Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85). Place the downloaded data in the `data` directory. Be sure to unzip `t15_copyTask_neuralData.zip` and `t15_pretrained_rnn_baseline.zip`.

## Training
To train the baseline RNN model, run the following command from the `model_training` directory:
```bash
conda activate b2txt25
python train_model.py
```
The model will train for 120,000 mini-batches (~3.5 hours on an RTX 4090) and should achieve an aggregate phoneme error rate of 10.1% on the validation partition. We note that the number of training batches and specific model hyperparameters may not be optimal here, and this baseline model is only meant to serve as an example. See `rnn_args.yaml` for a list of all hyperparameters.

## Evaluation
### Start redis server
To evaluate the model, first start a redis server in terminal with:
```bash
redis-server
```

### Start language model
Next, start the ngram language model in a seperate terminal window. For example, the 1gram language model can be started using the command below. Note that the 1gram model has no gramatical structure built into it. Details on downloading pretrained 3gram and 5gram language models and running them can be found in the README.md in the `language_model` directory.
To run the 1gram language model from the root directory of this repository:
```bash
conda activate b2txt_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

### Evaluate
Finally, run the `evaluate_model.py` script to load the pretrained baseline RNN, use it for inference on the heldout val or test sets to get phoneme logits, pass them through the language model via redis to get word predictions, and then save the predicted sentences to a .txt file in the format required for competition submission.
```bash
conda activate b2txt25
python evaluate_model.py --model_path ../data/t15_pretrained_rnn_baseline --data_dir ../data/t15_copyTask_neuralData --eval_type test --gpu_number 1
```

### Shutdown redis
When you're done, you can shutdown the redis server from any terminal using `redis-cli shutdown`.