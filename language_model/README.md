# Pretrained ngram language models
A pretrained 1gram language model is included in this repository at `language_model/pretrained_language_models/openwebtext_1gram_lm_sil`. Pretrained 3gram and 5gram language models are available for download [here](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) (`languageModel.tar.gz` and `languageModel_5gram.tar.gz`) and should likewise be placed in the `language_model/pretrained_language_models/` directory. Note that the 3gram model requires ~60GB of RAM, and the 5gram model requires ~300GB of RAM. Furthermore, OPT 6.7b requires a GPU with at least ~12.4 GB of VRAM to load for inference.

# Dependencies
```
CMake >= 3.14
gcc >= 10.1
pytorch == 1.13.1
```
To install CMake and gcc on Ubuntu, simply run:
```bash
sudo apt-get install build-essential
```

# Install language model python package
Use the `setup_lm.sh` script in the root directory of this repository to create the `b2txt_lm` conda env and install the `lm-decoder` package to it. Before install, make sure that there is no `build` or `fc_base` directory in your `language_model/runtime/server/x86` directory, as this may cause the build to fail.


# Using a pretrained ngram language model
The `language-model-standalone.py` script included here is made to work with the `evaluate_model.py` script in the `model_training` directory.  `language-model-standalone.py` will do the following when run: 
1. Initialize `opt-6.7b` it on the specified gpu (`--gpu_number` arg). The first time you run the script, it will automatically download `opt-6.7b` from huggingface.
2. Initialize the ngram language model (specified with the `--lm_path` arg)
3. Connect to the `localhost` redis server (or a different server, specified by the `--redis_ip` and `--redis_port` args)
4. Wait to receive phoneme logits via redis, and then make word predictions and pass them back via redis.


### `language-model-standalone.py` input args
See the bottom of the `language-model-standalone.py` script for a full list of input args.


### run a 1gram model
To run the 1gram language model from the root directory of this repository:
```bash
conda activate b2txt_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

### run a 3gram model
To run the 3gram language model from the root directory of this repository (requires ~60GB RAM):
```bash
conda activate b2txt_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_3gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

### run a 5gram model
To run the 5gram language model from the root directory of this repository (requires ~300GB of RAM):
```bash
conda activate b2txt_lm
python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_5gram_lm_sil --rescore --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0
```

# Build a new phoneme-to-words ngram language model from scratch
1. First, build binaries for building the language model:
    1. Build SRILM:
      ```bash
      cd srilm-1.7.3
      export SRILM=$PWD
      make MAKE_PIC=yes World
      make cleanest
      export PATH=$PATH:$PWD/bin/i686-m64
      ```

    2. Build openfst and other stuff:
      ```bash
      cd runtime/server/x86
      mkdir build
      cd build
      cmake ..
      make -j8
      ```

2. Build ngram LM:
  ```bash
  cd ./examples/speech/s0/
  run.sh output_dir dict_path train_corpus sil_prob formatted_train_corpus prune_threshold order
  ```


