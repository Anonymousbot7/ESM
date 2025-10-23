#!/bin/bash
#SBATCH --job-name=part10logist
#SBATCH --output=logdir/%j-%x.log
#SBATCH --error=logdir/error_%j-%x.log
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50GB
#SBATCH --partition=standard
#SBATCH --account=yili1
#SBATCH --mail-type=ALL


module --ignore_cache load "Python/3.12.1"


export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH


python code/part10/alogistic.py --n $1 --index $2