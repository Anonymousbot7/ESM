# Simulation Description

The simulation codes are written in **Python** and stored in the `code/` folder, where `part1` contains the baseline setup and `part2–part10` include simple modifications of its settings.

## File Descriptions

### part1
- **`alogistic.py`** **`apoisson.py`** **`abinomial.py`**  give the ESM simulation codes under the basic settings.
- **`hulclogistic.py`** **`hulcpoisson.py`**   give the code for HulC in comparison. **`hulcreport.py`** reports the results.
- **`Bayesianlogisticcorrectloss.py`** **`Bayesianpoissoncorrectloss.py`**   give the code for BNN in comparison. **`Bayesianreport.py`** reports the results.
- **`ensemblelogistic.py`** **`ensemblepoisson.py`** give the code for ensembling method in comparison. **`ensemblereport.py`** reports the results.
- **`naivebplogistic.py`** **`naivebppoisson.py`** give the code for naive boostrap in comparison. **`naivebpreport.py`** reports the results.
- **`Kernellogistic.py`** **`Kernelpoisson.py`** give the code for kernel regression.
- **`Rlogistic.py`** **`Rpoisson.py`** give the code for random forest.
- **`Blogistic.py`** **`Bpoisson.py`** give the code for different values of $B$ with $r=n^{0.9}$.

### part2-10
`part2` changes $f_0$; `part3` sets $B=3{,}000$; `part4` changes data dimension $p$; `part5` changes training epochs; `part6` considers a different NN; `part7` changes weight decay; `part8` changes NN depth; `part9` changes data signal-to-noise ratio; `part10` changes dropout rate.   
  
## Usage

Generate all test data using **`generatetest.ipynb`**, then run the corresponding `.sh` files (Slurm scripts) for each setting to produce results. The exampled submission code is written in the **`submit.txt`**.

