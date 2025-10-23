# Simulation Description

The simulation codes are written in **Python** and stored in the `code/` folder, where `part1` contains the baseline setup and `part2–part10` include simple modifications of its settings.

## File Descriptions

### part1
- **`alogistic.py`** **`apoisson.py`** **`abinomial.py`**  give the ESM simulation codes under the basic settings.
- **`Bayesianlogisticcorrectloss.m`** **`Bayesianpoissoncorrectloss.m`**   give the code for BNN in comparison. **`Bayesianreport.m`** reports the results.
- **`Kernellogistic.m`** **`Kernelpoisson.m`** give the code for kernel regression.
- **`ensemblelogistic.m`** **`ensemblepoisson.m`** give the code for ensembling method in comparison.


## Usage

Run the files in the following order:

This will generate synthetic data, run the regressions, compute debiased estimators, and produce the final simulation results. The corresponding .sh file helps us to submit the job to the High Performance Computing (HPC) platform.

