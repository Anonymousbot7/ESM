# Simulation Description

The simulation codes are written in **Python** and stored in the `code/` folder, where `part1` contains the baseline setup and `part2–part10` include simple modifications of its settings.

## File Descriptions

### part1
- **`alogistic.py`** **`apoisson.py`** **`abinomial.py`**  give the ESM simulation codes under the basic settings.
- - **`hulclogistic.py`** **`hulcpoisson.py`**   give the code for HulC in comparison. **`hulcreport.py`** reports the results.
- **`Bayesianlogisticcorrectloss.py`** **`Bayesianpoissoncorrectloss.py`**   give the code for BNN in comparison. **`Bayesianreport.py`** reports the results.
- **`ensemblelogistic.py`** **`ensemblepoisson.py`** give the code for ensembling method in comparison. **`ensemblereport.py`** reports the results.
- **`naivebplogistic.py`** **`naivebppoisson.py`** give the code for naive boostrap in comparison. **`naivebpreport.py`** reports the results.
- **`Kernellogistic.py`** **`Kernelpoisson.py`** give the code for kernel regression.
- **`Rlogistic.py`** **`Rpoisson.py`** give the code for random forest.
- **`Blogistic.py`** **`Bpoisson.py`** give the code for different values of $B$ with $r=n^{0.9}$.

### part2-10
`part2` changes $f_0$; 
  
## Usage

Run the files in the following order:

This will generate synthetic data, run the regressions, compute debiased estimators, and produce the final simulation results. The corresponding .sh file helps us to submit the job to the High Performance Computing (HPC) platform.

