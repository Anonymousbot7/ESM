import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sampleset import SampleSet
import os

import shutil
from collections import Counter
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier



class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)


def train_and_predict_RF(sample_set, X_test, mode="Bernoulli"):
    train_samples = sample_set.subtrain
    ntest = X_test.shape[0]
    n = sample_set.n
    r = sample_set.r
    counts = sample_set.counts
    B = len(train_samples)
    # Compute inclusion proportions
    J_dot = {i: counts[i] / B for i in range(n)}
    
    # Collect raw predictions
    all_out = torch.zeros(ntest, B)
    for j, (X_sub, Y_sub, _) in enumerate(train_samples):
        if mode == "Bernoulli":
            rf = RandomForestClassifier(n_estimators=200,
                                        criterion="entropy",
                                        random_state=42)
            rf.fit(X_sub, Y_sub)
            hatp = rf.predict_proba(X_test)[:, 1]
            hatp=np.clip(hatp, 0.02, 1 - 0.02)
            all_out[:, j] = torch.from_numpy(hatp)
        else:  # Poisson
            rf = RandomForestRegressor(n_estimators=200,
                                       criterion="poisson",
                                       random_state=42)
            rf.fit(X_sub, Y_sub)
            lam = rf.predict(X_test)
            all_out[:, j] = torch.from_numpy(lam)
    
    # Transform to logit/log
    if mode == "Bernoulli":
        all_out = torch.log(all_out / (1 - all_out))
    else:
        all_out = torch.log(all_out)

    # Ensemble mean prediction
    hatf_B = all_out.mean(dim=1)
    
    # Variance components
    sum_V2 = torch.zeros(ntest)
    sum_diff2 = torch.zeros(ntest)

    # Loop over original samples
    for i in range(n):
        Zs = torch.zeros(B, ntest)
        for j, (_, _, counts_sub) in enumerate(train_samples):
            in_subset = 1.0 if i in counts_sub else 0.0
            Zs[j] = (in_subset - J_dot[i]) * (all_out[:, j] - hatf_B)
        hat_V_i = Zs.mean(dim=0)
        sum_V2 += hat_V_i.pow(2)
        sum_diff2 += (Zs - hat_V_i.unsqueeze(0)).pow(2).sum(dim=0)

    # Bias correction factor
    factor = (n - 1) / n * (n / (n - r))**2
    term1 = factor * sum_V2
    term2 = factor * sum_diff2 / (B * (B - 1))
    var_corr = term1 - term2

    sd_raw = torch.sqrt(term1)
    sd_corr = torch.sqrt(var_corr)

    return hatf_B, sd_raw, sd_corr



def process_folder(i):
    sample_set_folder = 'sampleset/188/'  # Folder where sample sets are stored
    results_folder = 'results188/'  # Folder to save results
    os.makedirs(os.path.dirname(results_folder), exist_ok=True)
    B=3000
    p=10
    train_path = os.path.join(sample_set_folder, f'sampleset{p}bernoulli{B}_trainfolder{i}.pth')
    test_path  = os.path.join(sample_set_folder, f'sampleset{p}bernoulli{B}_testfolder{i}.pth')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return f"[{i}] folder does not exist"

    sample_train = SampleSet.load(train_path)
    sample_test  = SampleSet.load(test_path)
    X_test, y_test = sample_test.X, sample_test.Y
    print(f"Processing folder {i} with {sample_train.n} samples and {sample_train.p} features")
    hatf_B, sd_f, sd_f_c = train_and_predict_RF(sample_train, X_test, mode="Bernoulli")

    out_path = os.path.join(results_folder, f'resultsber{p}RF{B}_folder{i}.pth')
    torch.save({
        'y_test': y_test,
        'hatf_B': hatf_B,
        'sd_f'  : sd_f,
        'sd_f_c': sd_f_c
    }, out_path)
    return f"[{i}] finish"

if __name__ == '__main__':
    
    with parallel_backend('multiprocessing'):
        results = Parallel(n_jobs=5)(delayed(process_folder)(i) for i in range(5))
    print(results)