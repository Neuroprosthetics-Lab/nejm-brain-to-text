import argparse
import re

parser = argparse.ArgumentParser(description='Make corpus from text file')
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)

args = parser.parse_args()

output = open(args.output, 'w')
with open(args.input, 'r') as f:
    lines = f.readlines()
    for line in lines:
        word = line.strip().split(' ')[0].strip()
        if re.match(r'^[a-zA-Z]+$', word) is None:
            continue

        output.write(' '.join([l for l in word]) + '\n')
output.close()

