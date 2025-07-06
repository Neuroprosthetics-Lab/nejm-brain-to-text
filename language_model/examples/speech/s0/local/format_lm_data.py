import argparse
import json
import re
import nltk

def join_abbreviation(text):
    # Join abbreviations like "A.B.C." into "ABC"
    # Replace 'A.B' or 'a.b' with 'AB'
    text = re.sub(r'\b([a-zA-Z]\.){2,}', lambda x: ''.join(x.group(0).split('.')).upper(), text)
    return text

def handle_links(text):
    # Remove punctuation from links and hashtags, duplicating the characters so that it can not be recognized as a word
    text = re.sub(r'(http\S+|www\S+|https\S+|\S+@\S+|#\S+|@\S+|\b\w+://\S+|\b(\w+\.)+\w{2,})', lambda x: re.sub(r'[^\w]', '', x.group(0)) * 2, text)
    return text

def formalize_punctuation(text):
    conversion_dict = {
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',
        '–': '-',
        '—': '-',
        ' +': ' ',
        ' .': '.',
        ' ,': ',',
        ', ': ',',
    }
    for key, value in conversion_dict.items():
        text = text.replace(key, value)
    return text

def clean_text_pseudo(text):
    for sent in nltk.sent_tokenize(text):
        sent = sent.strip().lower()

        if not sent:
            continue

        yield sent

def clean_text(text):
    for sent in nltk.sent_tokenize(text):
        sent = sent.strip()
        if not sent:
            continue

        sent = handle_links(sent)
        sent = join_abbreviation(sent)
        sent = formalize_punctuation(sent)

        sent = re.sub(r'\:', ' ', sent)  # Replace ':' with space

        # Remove all punctuation using re.sub
        sent = re.sub(r"[^a-zA-Z\s']", " ", sent)

        # Remove all numbers (including float: 1.1 and numbers: 1,000 with comma) using re.sub
        sent = re.sub(r"\d+([\.,]\d+)?", " ", sent)

        # Remove apostrophes from the beginning and end of words (that are incorrectly used as quotes)
        sent = re.sub(r'(?:(?:^|\s)\')|\'$|\s\'\s|([^s])\'\s', r'\1 ', sent.strip()).strip()

        # Remove extra spaces
        sent = re.sub(r'\s+', ' ', sent) 

        sent = sent.strip().lower()

        # check for sigle character words that are not 'i' or 'a'
        if any([len(word) == 1 and word not in ['i', 'a'] for word in sent.split()]):
            continue

        if not sent:
            continue

        yield sent

parser = argparse.ArgumentParser(description='Format LM data')
parser.add_argument('--input_text', type=str, required=True)
parser.add_argument('--output_text', type=str, required=True)
parser.add_argument('--dict', type=str, required=True)
parser.add_argument('--with_punctuation', action='store_true')
parser.add_argument('--with_space_symbol', action='store_true')
parser.add_argument('--unk', action='store_true')
args = parser.parse_args()

# Read the dictionary
lexicons = set()
with open(args.dict, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f.readlines():
        if line.startswith(';;;'):
            continue
        tokens = line.strip().split(' ')
        lexicons.add(tokens[0].lower())

# Preprocess texts and write to output
output = open(args.output_text, 'w')
input = open(args.input_text, 'r')
count = 0
while True:
    line = input.readline()
    if not line:
        break
    
    if not line.strip():
        continue

    count += 1
    if count % 10000 == 0:
        print(count)

    for sub_line in re.split(r'\n+', line):
        sub_line = sub_line.strip()
        if not sub_line:
            continue
        sub_line_clean = clean_text(sub_line)
        for sub_sub_line in sub_line_clean:
            hasAllWords = True
            if not args.unk:
                words = sub_sub_line.lower().split(' ')
                for w in words:
                    if not w in lexicons:
                        hasAllWords = False
                        break
            if hasAllWords:
                output.write(sub_sub_line.upper()+ '\n')

output.close()
input.close()