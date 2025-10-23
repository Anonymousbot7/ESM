import os

import shutil

import torch
from collections import Counter
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd

from joblib import Parallel, delayed
import argparse
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import LogisticRegression
# folder_path = 'model1/'

# Create the folder if it doesn't exist
# os.makedirs(folder_path, exist_ok=True)

import numpy as np
import torch


def f_1(x):
    return x[:, 0] + 0.25 * x[:, 1] ** 2+0.1*torch.tanh(0.5*x[:,2]-0.3)

# def f_2(x):
#     return x[:,0]**2/2-abs(x[:,4]*x[:,9])+torch.exp(0.1*x[:,14])-torch.sin(3.141592*x[:,19])
# def f_2(x):
#     return 2*(x[:,0]>0)-2*(x[:,0]<0)

def poisson_loss(logits,y_true):
    """
    Compute the Poisson negative log-likelihood loss.
    
    Args:
        y_true (torch.Tensor): True labels (0, 1, 2, ...), shape (batch_size,).
        logits (torch.Tensor): Output of the DNN (before exponentiation), shape (batch_size,).
    
    Returns:
        torch.Tensor: Mean negative log-likelihood loss over the batch.
    """
    # Convert logits to λ(x) = e^logits
    lambda_pred = torch.exp(logits)
    
    # Compute the negative log-likelihood
    loss = lambda_pred - y_true * logits  # Equivalent to λ(x) - Y * log(λ(x))
    return loss.mean()

class SampleSet:
    def __init__(self, n, p,f_X,module='Bernoulli', mean=0, std=1,trials=None):
        """
        Initializes the SampleSet with n samples and p features for X.
        Y is generated based on the conditional probability P(Y=1|X).
        """
        self.n = n
        self.p = p
        self.mean = mean
        self.std = std
        self.r=None
        self.B=None
        self.module=module
        # Generate X with dimension (n, p)
        X_main = torch.normal(0.0, 1.0, size=(n, 2))
        X_noise = torch.normal(0.0, 1, size=(n, p - 2))
        self.X = torch.cat([X_main, X_noise], dim=1)
        # self.X=2*torch.rand((n, p)) -2
        self.subtrain=None
        self.subval=None
        self.counts=None
        self.trials=trials
        self.f_X=f_X
        # Compute z = f(X) and use it to generate P(Y=1|X) and Y
        self.z = self._compute_z(self.X)  # Save z values (f(X))
        self.Y = self._generate_Y(self.z)
    
    def _compute_z(self, X):
        return   self.f_X(X)
    
    def _generate_Y(self, z):
        if self.module=='Bernoulli':
            # Generate Y as a Bernoulli random variable with probability P(Y=1|X)
            P_Y_given_X = 1 / (1 + torch.exp(-z))
            Y = torch.bernoulli(P_Y_given_X)
        elif self.module=='Gaussian':
            Y = torch.normal(mean=z, std=1.0)
        elif self.module=='Binomial':
            if self.trials is None:
                self.trials = torch.randint(low=5, high=6, size=(self.n,))  

            P_Y_given_X = 1 / (1 + torch.exp(-z)) # Ensure the rate parameter is positive
            Y = torch.binomial(self.trials.float(), P_Y_given_X)

        elif self.module == 'Poisson':
        # Generate Y as a Poisson random variable with rate parameter (lambda) equal to exp(z)
            rate_param = torch.log(1+torch.exp(z))   # Ensure the rate parameter is positive
            Y = torch.poisson(rate_param)
        else:
        # Raise an error for unsupported modules
            raise ValueError(f"Unsupported module type: {self.module}. Expected one of: 'Bernoulli', 'Gaussian', 'Exponential', 'Poisson'.")
        return Y
    
    def get_z(self):
        """Returns the computed z values, which represent f(X)."""
        return self.z
    
    def get_sample_set(self):
        """Returns the main sample set (X, Y)."""
        return self.X, self.Y
    
    def get_sub_samples_with_validation(self, B, r):
        """
        Generates B sub-sample sets, each containing r samples randomly selected 
        from the main sample set, along with corresponding validation sets.
        
        Also counts the number of times each index is selected across all B sub-samples.
        
        Returns:
            train_samples: List of tuples, each containing (train_X, train_Y, train_indices)
            validation_samples: List of tuples, each containing (val_X, val_Y, val_indices)
            selection_counts: Dictionary with counts of each index's appearance in the B sub-samples.
        """
        train_samples = []
        validation_samples = []
        selection_counts = Counter({i: 0 for i in range(self.n)})  # To track appearances of each index
        indices = torch.arange(self.n)
        self.B=B
        self.r=r
        for _ in range(B):
            # Randomly select r unique indices for the sub-sample
            selected_indices = indices[torch.randperm(self.n)[:r]]
            
            # Update selection count for each index
            selection_counts.update(selected_indices.tolist())
            
            # Get validation indices (those not in selected_indices)
            val_indices = torch.tensor([i for i in indices if i not in selected_indices])

            # Separate sub-sample and validation sets, including original indices
            X_sub = self.X[selected_indices]
            Y_sub = self.Y[selected_indices]
            X_val = self.X[val_indices]
            Y_val = self.Y[val_indices]
            
            # Append to train_samples and validation_samples lists
            train_samples.append((X_sub, Y_sub, selected_indices))
            validation_samples.append((X_val, Y_val, val_indices))
        self.subtrain=train_samples
        self.subval=validation_samples
        self.counts=dict(selection_counts)
        return train_samples, validation_samples, dict(selection_counts)
    
    def save(self, file_path):
        """Saves the SampleSet instance to a file."""
        torch.save(self, file_path)
    
    @staticmethod
    def load(file_path):
        """Loads a SampleSet instance from a file."""
        return torch.load(file_path)

def kernel_poisson_regression(X_train, y_train, X_test, gamma=0.1, lam=1e-3, max_iter=10):
    y_train = y_train.ravel()
    K = rbf_kernel(X_train, X_train, gamma=gamma)
    K_s = rbf_kernel(X_test, X_train, gamma=gamma)
    f = np.log1p(y_train)  # 初始化

    for _ in range(max_iter):
        mu = np.exp(f)
        W = np.diag(mu)
        z = f + (y_train - mu) / mu
        A = K.T @ W @ K + lam * np.eye(len(y_train))
        b = K.T @ W @ z
        alpha = np.linalg.solve(A, b)
        f = K @ alpha

    return np.exp(K_s @ alpha).ravel()

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"folder {folder_path} does not exist")
        return


    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            os.remove(item_path)

        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)




def train_and_predict_Kernel(sample_set, X_test, mode="Bernoulli"):
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
        X_sub_np = X_sub.detach().cpu().numpy() if hasattr(X_sub, "detach") else X_sub
        Y_sub_np = Y_sub.detach().cpu().numpy().reshape(-1, 1) if hasattr(Y_sub, "detach") else Y_sub.reshape(-1, 1)
        Y_sub_np = Y_sub_np.ravel()
        X_test_np = X_test.detach().cpu().numpy() if hasattr(X_test, "detach") else X_test

        # --- Define RBF kernel ---

        if mode == "Bernoulli":
        # ---- Kernel Logistic Regression (GP classification) ----
            feature_map = Nystroem(kernel="rbf", gamma=1.0/X_sub_np.shape[1], n_components=300)

            # --- Step 2: logistic regression on mapped features ---
            model = make_pipeline(feature_map,
                                LogisticRegression(
                                    penalty='l2',
                                    solver='lbfgs',
                                    max_iter=300,
                                    C=10.0))    

            model.fit(X_sub_np, Y_sub_np)
            hatp = model.predict_proba(X_test_np)[:, 1]   
            all_out[:, j] = torch.from_numpy(hatp)
        else:  # Poisson
            # Predict mean λ = E[Y|X]
            # lam_pred = kernel_poisson_regression(X_sub_np,Y_sub_np,X_test_np)
            # all_out[:, j] = torch.from_numpy(lam_pred)


            feature_map = Nystroem(kernel="rbf", gamma=1.0/X_sub_np.shape[1], n_components=300)

            # Step 2: linear Poisson on the mapped features
            model = make_pipeline(feature_map, PoissonRegressor(alpha=1e-3, max_iter=300))
            model.fit(X_sub_np, Y_sub_np)

            lam_pred = model.predict(X_test_np)
            all_out[:, j] = torch.from_numpy(lam_pred)

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

# Usage:
# hatf_B, sd_raw, sd_corr = train_and_predict_Kernel(sample_set, X_test, mode="Bernoulli")


def run_one_repeat(rep_id, n, r, B, p, GLM_name, f_1, xtest):
    # 每个 repeat 训练 B 个网络，最后做一次 ensemble 预测
    ss = SampleSet(n, p, f_1, module=GLM_name)
    ss.get_sub_samples_with_validation(B, r)

    
    Bf=train_and_predict_Kernel(ss,xtest, mode=GLM_name)
    return Bf[0], Bf[1],Bf[2]






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   required=True)
    parser.add_argument("--index", type=float, required=True)
    args = parser.parse_args()
    n        = args.n
    index_e  = args.index

######Constant Area#######

    r        = int(n ** index_e)
    B        = 1400          # or your B
    p        = 10           # or your p
    GLM_name = "Bernoulli"  # or your mode  
    folder = 'resultspart1'      
    xtest    = torch.load(f"{folder}/xtest10.pt")  # 预先生成并保存

    repeats = 300

    # Parallel 
    results = Parallel(n_jobs=23)(
        delayed(run_one_repeat)(i, n, r, B, p, GLM_name, f_1, xtest)
        for i in range(repeats)
    )

    # 解包并 stack
    Bf0_tensor = torch.stack([res[0] for res in results])  # [100, ntest]
    Bf1_tensor = torch.stack([res[1] for res in results])
    Bf2_tensor = torch.stack([res[2] for res in results])

    # 保存到 CSV
    df_bf0 = pd.DataFrame(Bf0_tensor.numpy())
    df_bf1 = pd.DataFrame(Bf1_tensor.numpy())
    df_bf2 = pd.DataFrame(Bf2_tensor.numpy())

    fn0 = f"{folder}/{GLM_name}fBf1Kerneln{n}p{p}B{B}r{r}.csv"
    fn1 = f"{folder}/{GLM_name}sdf1Kernel{n}p{p}B{B}r{r}.csv"
    fn2 = f"{folder}/{GLM_name}sdcrtf1Kernel{n}p{p}B{B}r{r}.csv"
    df_bf0.to_csv(fn0, index=False, header=False)
    df_bf1.to_csv(fn1, index=False, header=False)
    df_bf2.to_csv(fn2, index=False, header=False)


    print(f"Done n={n}, index={index_e}, saved {fn0}, {fn1},{fn2}")

