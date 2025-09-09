#!/bin/bash
#SBATCH --job-name=lang_ideo_experiment
#SBATCH --gres=gpu:a40:1
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH -c 4
#SBATCH --mem=30GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# prepare your environment here
module load python/3.10.12
source /h/jwatson/some_env2/bin/activate

# put your command here
python step_2_huggingface_query.py
