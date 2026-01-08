# Real Data Experiments

The real data used in this study are publicly available from the eICU Collaborative Research Database (https://eicu-crd.mit.edu/) upon completion of a data use agreement and approval through the PhysioNet credentialed access process.


## Data Processing
The downloaded data are stored in **`.parquet`** format.  Use **`data188.ipynb`** and **`data458.ipynb`** to process data from hospital sites 188 and 458, respectively. The processed results are saved in **`.csv`** format.


## Data Analysis
- **`dataprocessinglogistic.ipynb`** and **`dataprocessingpoisson.ipynb`** further process the **`.csv`** files into a custom `sampleset` class.  This class separates the $B$ subsamples and stores them for convenient use in subsequent ESM analysis. The data are equally split into five folders; in each run, four folders are used for training and the remaining one for testing, enabling full prediction coverage across all samples. Features with 46 has 29 continuous features and 17 categorical features. 

- **`parallelNNLog.py`**, **`parallelNNPoi.py`**, **`parallelRFLog.py`**, and **`parallelRFPoi.py`** implement the ESM algorithm under logistic and Poisson regression using neural networks and random forests, respectively. To run the code, use the corresponding **`.sh`** files; example commands are provided in **`submit.txt`**.

- **`parallelNNLog458.py`** and **`parallelNNPoi458.py`** run experiments trained on hospital 188 and directly transferred, without retraining, to hospital 458. To execute the code, use the corresponding **`.sh`** files; example commands are provided in **`submit.txt`**.

- **`readlogisticNN.ipynb`**, **`readpoiNN.ipynb`**, **`readlogisticRF.ipynb`**, and **`readpoiRF.ipynb`** show the results of the ESM algorithm under logistic and Poisson regression using neural networks and random forests for hospital site 188. Runing the last section in **`readlogisticNN.ipynb`** will obtain Figure S.5 in Section B.1. Running these four files with p=10 or 46 will obtain Figures 4, 5, S.6 and S.7. 

- **`readlogisticNN458.ipynb`** and **`readpoiNN458.ipynb`** show the transfer results for hospital site 458. Running these files will obtain Figure S.9.

- **`plotdensitydiff.ipynb`** will show Figure S.8. 

## Note 
In the **`.sh`** files, replace `youraccount` with your specific HPC account.

