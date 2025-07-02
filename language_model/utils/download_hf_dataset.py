import os
import argparse
import datasets

from tqdm import tqdm


def convert_dataset_to_corpus(dataset, output_file):
    assert output_file is not None and os.path.exists(output_file), "Please provide an valid output file path"

    with open(os.path.join('corpus', 'financial-reports-sec.txt'), 'w') as fw:
        for split in tqdm(dataset.keys(), desc="Writing dataset splits"):
            for sample in tqdm(dataset[split], desc=f"Writing {split} split"):
                fw.write(sample['sentence'] + '\n')

def main(args):
    # Load the lite configuration of the dataset
    raw_dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name)
    convert_dataset_to_corpus(raw_dataset, args.output_file)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_file", type=str, default=None)
    argparser.add_argument("--dataset_name", type=str, default="JanosAudran/financial-reports-sec")
    argparser.add_argument("--dataset_config_name", type=str, default="all")
    args = argparser.parse_args()
    # Load the lite configuration of the dataset
    main(args)