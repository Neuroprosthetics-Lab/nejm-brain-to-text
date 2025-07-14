import torch
import numpy as np
import h5py
import time
import re

from data_augmentations import gauss_smooth

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]

def _extract_transcription(input):
    endIdx = np.argwhere(input == 0)[0, 0]
    trans = ''
    for c in range(endIdx):
        trans += chr(input[c])
    return trans

def load_h5py_file(file_path, b2txt_csv_df):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
        'corpus': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            # match this trial up with the csv to get the corpus name
            year, month, day = session.split('.')[1:]
            date = f'{year}-{month}-{day}'
            row = b2txt_csv_df[(b2txt_csv_df['Date'] == date) & (b2txt_csv_df['Block number'] == block_num)]
            corpus_name = row['Corpus'].values[0]

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
            data['corpus'].append(corpus_name)
    return data

def rearrange_speech_logits_pt(logits):
    # original order is [BLANK, phonemes..., SIL]
    # rearrange so the order is [BLANK, SIL, phonemes...]
    logits = np.concatenate((logits[:, :, 0:1], logits[:, :, -1:], logits[:, :, 1:-1]), axis=-1)
    return logits

# single decoding step function.
# smooths data and puts it through the model.
def runSingleDecodingStep(x, input_layer, model, model_args, device):

    # Use autocast for efficiency
    with torch.autocast(device_type = "cuda", enabled = model_args['use_amp'], dtype = torch.bfloat16):

        x = gauss_smooth(
            inputs = x, 
            device = device,
            smooth_kernel_std = model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size = model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding = 'valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x = x,
                day_idx = torch.tensor([input_layer], device=device),
                states = None, # no initial states
                return_state = True,
            )

    # convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    # # original order is [BLANK, phonemes..., SIL]
    # # rearrange so the order is [BLANK, SIL, phonemes...]
    # logits = rearrange_speech_logits_pt(logits)

    return logits

def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence

def get_current_redis_time_ms(redis_conn):
    t = redis_conn.time()
    return int(t[0]*1000 + t[1]/1000)


######### language model helper functions ##########

def reset_remote_language_model(
        r,
        remote_lm_done_resetting_lastEntrySeen,
    ):
    
    r.xadd('remote_lm_reset', {'done': 0})
    time.sleep(0.001)
    # print('Resetting remote language model before continuing...')
    remote_lm_done_resetting = []
    while len(remote_lm_done_resetting) == 0:
        remote_lm_done_resetting = r.xread(
            {'remote_lm_done_resetting': remote_lm_done_resetting_lastEntrySeen},
            count=1,
            block=10000,
        )
        if len(remote_lm_done_resetting) == 0:
            print(f'Still waiting for remote lm reset from ts {remote_lm_done_resetting_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_resetting[0][1]:
        remote_lm_done_resetting_lastEntrySeen = entry_id
        # print('Remote language model reset.')

    return remote_lm_done_resetting_lastEntrySeen


def update_remote_lm_params(
        r,
        remote_lm_done_updating_lastEntrySeen,
        acoustic_scale=0.35,
        blank_penalty=90.0,
        alpha=0.55,
    ):
    
    # update remote lm params
    entry_dict = {
        # 'max_active': max_active,
        # 'min_active': min_active,
        # 'beam': beam,
        # 'lattice_beam': lattice_beam,
        'acoustic_scale': acoustic_scale,
        # 'ctc_blank_skip_threshold': ctc_blank_skip_threshold,
        # 'length_penalty': length_penalty,
        # 'nbest': nbest,
        'blank_penalty': blank_penalty,
        'alpha': alpha,
        # 'do_opt': do_opt,
        # 'rescore': rescore,
        # 'top_candidates_to_augment': top_candidates_to_augment,
        # 'score_penalty_percent': score_penalty_percent,
        # 'specific_word_bias': specific_word_bias,
    }

    r.xadd('remote_lm_update_params', entry_dict)
    time.sleep(0.001)
    remote_lm_done_updating = []
    while len(remote_lm_done_updating) == 0:
        remote_lm_done_updating = r.xread(
            {'remote_lm_done_updating_params': remote_lm_done_updating_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_done_updating) == 0:
            print(f'Still waiting for remote lm to update parameters from ts {remote_lm_done_updating_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_done_updating[0][1]:
        remote_lm_done_updating_lastEntrySeen = entry_id
        # print('Remote language model params updated.')

    return remote_lm_done_updating_lastEntrySeen


def send_logits_to_remote_lm(
        r,
        remote_lm_input_stream,
        remote_lm_output_partial_stream,
        remote_lm_output_partial_lastEntrySeen,
        logits,
    ):
    
    # put logits into remote lm and get partial output
    r.xadd(remote_lm_input_stream, {'logits': np.float32(logits).tobytes()})
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_partial_stream: remote_lm_output_partial_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm partial output from ts {remote_lm_output_partial_lastEntrySeen}...')
    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_partial_lastEntrySeen = entry_id
        decoded = entry_data[b'lm_response_partial'].decode()

    return remote_lm_output_partial_lastEntrySeen, decoded


def finalize_remote_lm(
        r,
        remote_lm_output_final_stream,
        remote_lm_output_final_lastEntrySeen,
    ):
    
    # finalize remote lm
    r.xadd('remote_lm_finalize', {'done': 0})
    time.sleep(0.005)
    remote_lm_output = []
    while len(remote_lm_output) == 0:
        remote_lm_output = r.xread(
            {remote_lm_output_final_stream: remote_lm_output_final_lastEntrySeen},
            block=10000,
            count=1,
        )
        if len(remote_lm_output) == 0:
            print(f'Still waiting for remote lm final output from ts {remote_lm_output_final_lastEntrySeen}...')
    # print('Received remote lm final output.')

    for entry_id, entry_data in remote_lm_output[0][1]:
        remote_lm_output_final_lastEntrySeen = entry_id

        candidate_sentences = [str(c) for c in entry_data[b'scoring'].decode().split(';')[::5]]
        candidate_acoustic_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[1::5]]
        candidate_ngram_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[2::5]]
        candidate_llm_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[3::5]]
        candidate_total_scores = [float(c) for c in entry_data[b'scoring'].decode().split(';')[4::5]]


    # account for a weird edge case where there are no candidate sentences
    if len(candidate_sentences) == 0 or len(candidate_total_scores) == 0:
        print('No candidate sentences were received from the language model.')
        candidate_sentences = ['']
        candidate_acoustic_scores = [0]
        candidate_ngram_scores = [0]
        candidate_llm_scores = [0]
        candidate_total_scores = [0]

    else:
        # sort candidate sentences by total score (higher is better)
        sort_order = np.argsort(candidate_total_scores)[::-1]

        candidate_sentences = [candidate_sentences[i] for i in sort_order]
        candidate_acoustic_scores = [candidate_acoustic_scores[i] for i in sort_order]
        candidate_ngram_scores = [candidate_ngram_scores[i] for i in sort_order]
        candidate_llm_scores = [candidate_llm_scores[i] for i in sort_order]
        candidate_total_scores = [candidate_total_scores[i] for i in sort_order]

    # loop through candidates backwards and remove any duplicates
    for i in range(len(candidate_sentences)-1, 0, -1):
        if candidate_sentences[i] in candidate_sentences[:i]:
            candidate_sentences.pop(i)
            candidate_acoustic_scores.pop(i)
            candidate_ngram_scores.pop(i)
            candidate_llm_scores.pop(i)
            candidate_total_scores.pop(i)

    lm_out = {
        'candidate_sentences': candidate_sentences,
        'candidate_acoustic_scores': candidate_acoustic_scores,
        'candidate_ngram_scores': candidate_ngram_scores,
        'candidate_llm_scores': candidate_llm_scores,
        'candidate_total_scores': candidate_total_scores,
    }

    return remote_lm_output_final_lastEntrySeen, lm_out