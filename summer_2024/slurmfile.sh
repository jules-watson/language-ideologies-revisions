#!/bin/bash
#SBATCH --job-name=lang_ideo_experiment
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --qos=long
#SBATCH --time=30:00:00
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# prepare your environment here
source /h/jwatson/some_env2/bin/activate

# put your command here
python step_2_huggingface_query.py
