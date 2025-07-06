import argparse
import os

def make_position_dependent_phones(dict_path, output_path):
    output_file = open(output_path, 'w')
    with open(dict_path, 'r') as f:
        for line in f.readlines():
            word, phones = line.strip().split('\t', 2)
            phones = phones.strip().split(' ')
            new_phones = []
            for i, p in enumerate(phones):
                if i == 0:
                    new_phones.append(p + '_B')
                elif i == len(phones) - 1:
                    new_phones.append(p + '_E')
                else:
                    new_phones.append(p + '_I')
            output_file.write(word + '\t' + ' '.join(new_phones) + '\n')
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', type=str, help='Dictionary file')
    parser.add_argument('--output', type=str, help='Output dictionary file')
    args = parser.parse_args()

    make_position_dependent_phones(args.dict, args.output)