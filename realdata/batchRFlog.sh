#!/bin/bash
#SBATCH --job-name=RFLogreal
#SBATCH --output=logdir/%j-%x.log
#SBATCH --error=logdir/error_%j-%x.log
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15GB
#SBATCH --partition=standard
#SBATCH --account=youraccount
#SBATCH --mail-type=ALL


module --ignore_cache load "Python/3.12.1"


export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH


python parallelRFLog.py 