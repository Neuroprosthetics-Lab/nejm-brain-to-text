import os
import sys
import torch
import numpy as np
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from rnn_model import GRUDecoder
from evaluate_model_helpers import *

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory (relative to the current working directory).')
parser.add_argument('--data_dir', type=str, default='../data/t15_copyTask_neuralData',
                    help='Path to the dataset directory (relative to the current working directory).')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" for validation set, "test" for test set. '
                         'If "test", ground truth is not available.')
parser.add_argument('--gpu_number', type=int, default=1,
                    help='GPU number to use for RNN model inference. Set to -1 to use CPU.')
args = parser.parse_args()

# paths to model and data directories
# Note: these paths are relative to the current working directory
model_path = args.model_path
data_dir = args.data_dir

# define evaluation type
eval_type = args.eval_type  # can be 'val' or 'test'. if 'test', ground truth is not available

# load model args
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# set up gpu device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    if gpu_number >= torch.cuda.device_count():
        raise ValueError(f'GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
    device = f'cuda:{gpu_number}'
    device = torch.device(device)
    print(f'Using {device} for model inference.')
else:
    if gpu_number >= 0:
        print(f'GPU number {gpu_number} requested but not available.')
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# define model
model = GRUDecoder(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'], 
    n_days = len(model_args['dataset']['sessions']),
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

# load model weights
checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
# rename keys to not start with "module." (happens if model was saved with DataParallel)
for key in list(checkpoint['model_state_dict'].keys()):
    checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
    checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
model.load_state_dict(checkpoint['model_state_dict'])  

# add model to device
model.to(device) 

# set model to eval mode
model.eval()

# load data for each session
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')

        data = load_h5py_file(eval_file)
        test_data[session] = data

        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')
print(f'Total number of {eval_type} trials: {total_test_trials}')
print()


# put neural data through the pretrained model to get phoneme predictions (logits)
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():

        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            # get neural input for the trial
            neural_input = data['neural_features'][trial]

            # add batch dimension
            neural_input = np.expand_dims(neural_input, axis=0)

            # convert to torch tensor
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

            # run decoding step
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)

            pbar.update(1)
pbar.close()


# convert logits to phoneme sequences and print them out
for session, data in test_data.items():
    data['pred_seq'] = []
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        # remove blanks (0)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # remove consecutive duplicates
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        # convert to phonemes
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        # add to data
        data['pred_seq'].append(pred_seq)

        # print out the predicted sequences
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]

            print(f'Sentence label:      {sentence_label}')
            print(f'True sequence:       {" ".join(true_seq)}')
        print(f'Predicted Sequence:  {" ".join(pred_seq)}')
        print()


# language model inference via redis
# make sure that the standalone language model is running on the localhost redis ip
# see README.md for instructions on how to run the language model
r = redis.Redis(host='localhost', port=6379, db=0)
remote_lm_input_stream = 'remote_lm_input'
remote_lm_output_partial_stream = 'remote_lm_output_partial'
remote_lm_output_final_stream = 'remote_lm_output_final'

remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_finalizing_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_updating_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {
    'session': [],
    'block': [],
    'trial': [],
    'true_sentence': [],
    'pred_sentence': [],
}

# loop through all trials and put logits into the remote language model to get text predictions
# note: this takes ~15-20 minutes to run on the entire test split with the 5-gram LM + OPT rescoring (RTX 4090)
with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            # get trial logits and rearrange them for the LM
            logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]

            # reset language model
            remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(r, remote_lm_done_resetting_lastEntrySeen)
            
            '''
            # update language model parameters
            remote_lm_done_updating_lastEntrySeen = update_remote_lm_params(
                r,
                remote_lm_done_updating_lastEntrySeen,
                acoustic_scale=0.35,
                blank_penalty=90.0,
                alpha=0.55,
            )
            '''

            # put logits into LM
            remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                r,
                remote_lm_input_stream,
                remote_lm_output_partial_stream,
                remote_lm_output_partial_lastEntrySeen,
                logits,
            )

            # finalize remote LM
            remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                r,
                remote_lm_output_final_stream,
                remote_lm_output_final_lastEntrySeen,
            )

            # get the best candidate sentence
            best_candidate_sentence = lm_out['candidate_sentences'][0]

            # store results
            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
            lm_results['pred_sentence'].append(best_candidate_sentence)

            # update progress bar
            pbar.update(1)
pbar.close()


# if using the validation set, lets calculate the aggregate word error rate (WER)
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0

    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        print(f'WER: {ed} / {100 * len(true_sentence.split())} = {ed / len(true_sentence.split()):.2f}%')
        print()

    print(f'Total true sentence length: {total_true_length}')
    print(f'Total edit distance: {total_edit_distance}')
    print(f'Aggregate Word Error Rate (WER): {100 * total_edit_distance / total_true_length:.2f}%')


# write predicted sentences to a text file. put a timestamp in the filename (YYYYMMDD_HHMMSS)
output_file = os.path.join(model_path, f'baseline_rnn_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.txt')
with open(output_file, 'w') as f:
    for i in range(len(lm_results['pred_sentence'])):
        f.write(f"{remove_punctuation(lm_results['pred_sentence'][i])}\n")