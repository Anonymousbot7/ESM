#!/bin/bash
#SBATCH --job-name=RFpoi
#SBATCH --output=logdir/%j-%x.log
#SBATCH --error=logdir/error_%j-%x.log
#SBATCH --time=80:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --mem=80GB
#SBATCH --partition=standard
#SBATCH --account=yili0
#SBATCH --mail-type=ALL


module --ignore_cache load "Python/3.12.1"


export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH


python code/part9/Rpoisson.py --n $1 --index $2