#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --job-name=lm
#SBATCH --mail-type=ALL
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=owners
#SBATCH --signal=USR1@120
#SBATCH --time=10000

export PATH=$PATH:/oak/stanford/groups/henderj/stfan/code/nptlrig2/LanguageModelDecoder/srilm-1.7.3/bin/i686-m64/
ml gcc/10.1.0

. ./path.sh

