import numpy as np
import re
from g2p_en import G2p



LOGIT_PHONE_DEF = [
    'BLANK', 'SIL', # blank and silence
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]
SIL_DEF = ['SIL']


# remove puntuation from text
def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join(sentence.split())

    return sentence


# Convert RNN logits to argmax phonemes
def logits_to_phonemes(logits):
    seq = np.argmax(logits, axis=1)
    seq2 = np.array([seq[0]] + [seq[i] for i in range(1, len(seq)) if seq[i] != seq[i-1]])

    phones = []
    for i in range(len(seq2)):
        phones.append(LOGIT_PHONE_DEF[seq2[i]])

    # Remove blank and repeated phonemes
    phones = [p for p in phones if  p!='BLANK']
    phones = [phones[0]] + [phones[i] for i in range(1, len(phones)) if phones[i] != phones[i-1]]

    return phones


# Convert text to phonemes
def sentence_to_phonemes(thisTranscription, g2p_instance=None):
    if not g2p_instance:
        g2p_instance = G2p()

    # Remove punctuation
    thisTranscription = remove_punctuation(thisTranscription)

    # Convert to phonemes
    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = SIL_DEF
    else:
        for p in g2p_instance(thisTranscription):
            if p==' ':
                phonemes.append('SIL')

            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        #add one SIL symbol at the end so there's one at the end of each word
        phonemes.append('SIL')
    
    return phonemes, thisTranscription


# Calculate WER or PER
def calculate_error_rate(r, h):
    """
    Calculation of WER or PER with Levenshtein distance.
    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.
    ----------
    Parameters:
    r : list of true words or phonemes
    h : list of predicted words or phonemes
    ----------
    Returns:
    Word error rate (WER) or phoneme error rate (PER) [int]
    ----------
    Examples:
    >>> calculate_wer("who is there".split(), "is there".split())
    1
    >>> calculate_wer("who is there".split(), "".split())
    3
    >>> calculate_wer("".split(), "who is there".split())
    3
    """
    # initialization
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


# calculate aggregate WER or PER
def calculate_aggregate_error_rate(r, h):

    # list setup
    err_count = []
    item_count = []
    error_rate_ind = []

    # calculate individual error rates
    for x in range(len(h)):
        r_x = r[x]
        h_x = h[x]

        n_err = calculate_error_rate(r_x, h_x)

        item_count.append(len(r_x))
        err_count.append(n_err)
        error_rate_ind.append(n_err / len(r_x))

    # Calculate aggregate error rate
    error_rate_agg = np.sum(err_count) / np.sum(item_count)

    # calculate 95% CI
    item_count = np.array(item_count)
    err_count = np.array(err_count)
    nResamples = 10000
    resampled_error_rate = np.zeros([nResamples,])
    for n in range(nResamples):
        resampleIdx = np.random.randint(0, item_count.shape[0], [item_count.shape[0]])
        resampled_error_rate[n] = np.sum(err_count[resampleIdx]) / np.sum(item_count[resampleIdx])
    error_rate_agg_CI = np.percentile(resampled_error_rate, [2.5, 97.5])

    # return everything as a tuple
    return (error_rate_agg, error_rate_agg_CI[0], error_rate_agg_CI[1], error_rate_ind)