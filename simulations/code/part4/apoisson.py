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



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate=0.0):
        """
        Defines a neural network with three layers: two hidden layers with dropout and one output layer.
        The output is a single scalar, representing the approximation of f(X).
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after the first hidden layer
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after the second hidden layer
        self.output = nn.Linear(hidden_size2, 1)  # Single output for f(X)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)  # Apply dropout after first hidden layer
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)  # Apply dropout after second hidden layer
        x = self.output(x)  # Output is a single logit (for binary classification)
        return x

def train_network(network, train_loader,mode="Bernoulli", num_epochs=100, learning_rate=0.1,weight_decay=0.05,tol=0.001,patience=3):
    if mode == "Bernoulli":
        criterion = nn.BCEWithLogitsLoss() 
    elif mode=="Poisson":
        criterion=poisson_loss
    else:
    # Raise an error for unsupported modules
        raise ValueError(f"Unsupported module type {mode}. Expected one of: 'Bernoulli', 'Gaussian', 'Exponential', 'Poisson'.")
    
    # BCELossWithLogits combines sigmoid and binary cross-entropy in one function
    length=len(train_loader.dataset)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,weight_decay=weight_decay)
    prev_epoch_loss = None  
    stable_count=0 
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0  
        for batch_X, batch_Y in train_loader:
            # Forward pass
            logits = network(batch_X).squeeze() # Get scalar logits, shape (batch_size)
            # print(torch.max(logits))  
            loss = criterion(logits, batch_Y.float())  # Y needs to be float for BCELossWithLogits
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
    
    # Compute average loss for the epoch
        epoch_loss = running_loss / length
        # print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Early‐stop check
        if prev_epoch_loss is not None:
            delta = abs(prev_epoch_loss - epoch_loss)
            if delta < tol:
                stable_count += 1
            else:
                stable_count = 0  # 重置计数

            if stable_count >= patience:
                print(f"Early stop at epoch {epoch} after {patience} stable epochs with loss {epoch_loss}.")
                break
        prev_epoch_loss = epoch_loss
    print(f"With loss {epoch_loss}.")

# Instantiate and train B neural networks, one for each sub-sample
def train_multiple_networks(sample_set, input_size, hidden_size1, hidden_size2,mode="Bernoulli",batchsize=64, dropout_rate=0.1, num_epochs=100, learning_rate=0.01,weight_decay=0.05,tol=0.001,patience=3):
    train_samples, validation_samples, selection_counts = sample_set.subtrain, sample_set.subval, sample_set.counts
    
    networks = []  # List to hold trained neural networks
    for i, (train_data, val_data) in enumerate(zip(train_samples, validation_samples)):
        X_sub, Y_sub, _ = train_data
        X_val, Y_val, _ = val_data

        
        # Prepare data loader for this sub-sample
        train_dataset = TensorDataset(X_sub, Y_sub)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        
        # Instantiate a new network for this sub-sample
        network = NeuralNetwork(input_size, hidden_size1, hidden_size2, dropout_rate=dropout_rate)
        # print(f"\nTraining network {i+1}/{B} on sub-sample {i+1}")
        
        # Train the network on the current sub-sample
        train_network(network, train_loader,mode=mode, num_epochs=num_epochs, learning_rate=learning_rate,weight_decay=weight_decay,tol=tol,patience=patience)
        
        # Append the trained network to the list of networks
        networks.append(network)

        # Validation performance (optional)
        with torch.no_grad():
            if mode=="Bernoulli":
                logits = network(X_val).squeeze()
                true_logits=f_1(X_val)
                accuracy = (torch.sign(logits) == torch.sign(true_logits)).float().mean() * 100
                print(f"Validation Accuracy for network {i+1}: {accuracy.item():.2f}%")
                print(torch.max(logits),torch.min(logits))
                print(torch.max(true_logits),torch.min(true_logits))
            elif mode == "Poisson":
                logits = network(X_val).squeeze()

                true_lambda=torch.log(1+torch.exp(f_1(X_val)))
                estimated_lambda=torch.exp(logits)
                print(f" network {i+1}: {torch.max(true_lambda),torch.min(true_lambda)}%")

                print(torch.mean(abs(true_lambda-estimated_lambda)),torch.std(abs(true_lambda-estimated_lambda)))

          
            else:
                raise ValueError(f"Unsupported module type {mode}. Expected one of: 'Bernoulli', 'Gaussian', 'Exponential', 'Poisson'.")


    
    return networks



# def ensemble_predict_batch_f(Xtest, networks,sample_set):
#     ntest = Xtest.shape[0]
#     n = sample_set.n  # Total number of original samples
#     r = sample_set.r  # Size of each sub-sample
#     B = len(networks)  # Number of sub-samples (number of neural networks)
#     # Prepare a tensor to store all network outputs (shape: [ntest, B])
#     all_outputs = torch.zeros(ntest, B)
#     # Step 1: Compute the average prediction (log-odds) for each xtest sample across all networks
#     with torch.no_grad():  # Disable gradient computation for inference
#         for i, network in enumerate(networks):
#             logits = network(Xtest).squeeze(1)  # Get logits for all test samples (shape: [ntest])
#             all_outputs[:, i] = logits  # Store the logits from each network in the i-th column
#     # Step 2: Compute J_bji and J_.i for each index i
#     J_bji = sample_set.counts  # This is J_bji as a dictionary from SampleSet

#     J_dot_i = {i: J_bji[i]/B for i in range(n)}  # J_.i = mean of J_bji across B
#     # Average the logits across the B networks to get ensemble logits

#     # Step 3: Compute Cov_i,* for each index i and each test sample in xtest
#     hatf_B=all_outputs.mean(dim=1)
#     sigma_squared_star_f = torch.zeros(ntest)  # Initialize covariance estimate for each test sample
#     for i in range(n):
#         cov_i_star = torch.zeros(ntest)
#         correction_i_star=torch.zeros(ntest)
#         for j in range(B):
#             _,_,Jbjicount=sample_set.subtrain[j]

#             J_bji_value = 1 if i in Jbjicount else 0  # Indicator if index i is in sub-sample j
#             deviations = all_outputs[:, j] - hatf_B  # Deviation of each network's prediction from the mean, shape: (ntest,)
#             cov_i_star += (J_bji_value - J_dot_i[i]) * deviations / B 
        
#         # Sum cov_i_values over all B sub-samples for index i, then square and sum for all i
#         sigma_squared_star_f += cov_i_star.pow(2) 
#     # print(sigma_squared_star_f)
#     print(sum(deviations**2))
#     # factor=(n-1)/n*(n/(n-r))**2
#     factor=(n-1)/n*(n/(n-r))**2
#     var_f=factor*sigma_squared_star_f
#     sd_f=torch.sqrt(var_f)
#     upper_f=(hatf_B+1.96*sd_f)
#     lower_f=(hatf_B-1.96*sd_f)



#     return all_outputs, [hatf_B,sd_f,upper_f,lower_f]#,[probabilities_h,sd_h,upper_h,lower_h]    # based on average over f or h



def ensemble_predict_batch_f(Xtest, networks, sample_set):
    ntest = Xtest.shape[0]
    n = sample_set.n       # Total number of original samples
    r = sample_set.r       # Size of each sub-sample
    B = len(networks)      # Number of sub-samples (number of neural networks)

    # Collect logits from all networks for the test set (shape: [ntest, B])
    all_outputs = torch.zeros(ntest, B)
    with torch.no_grad():
        for j, net in enumerate(networks):
            logits = net(Xtest).squeeze(1)
            all_outputs[:, j] = logits

    # Compute inclusion counts J_bji and mean inclusion J_dot_i for each training index i
    J_bji = sample_set.counts  # Dict mapping i -> count of i in each sub-sample
    J_dot_i = {i: J_bji[i] / B for i in range(n)}

    # Ensemble mean prediction for each test sample
    hatf_B = all_outputs.mean(dim=1)  # Shape: [ntest]

    # Initialize accumulators for variance correction terms
    sum_V2 = torch.zeros(ntest)      # Accumulate sum of hat_V_i^2 over i
    sum_Zdiff2 = torch.zeros(ntest)  # Accumulate sum of (Z_ji - hat_V_i)^2 over i and j

    # Loop over each original data index i
    for i in range(n):
        # Gather Z_{b_j i}(x*) for all sub-samples j (shape: [B, ntest])
        Zs = torch.zeros(B, ntest)
        for j in range(B):
            _, _, Jbjicount = sample_set.subtrain[j]
            in_subset = 1.0 if (i in Jbjicount) else 0.0
            deviations = all_outputs[:, j] - hatf_B  # Shape: [ntest]
            Zs[j] = (in_subset - J_dot_i[i]) * deviations

        # Compute hat_V_i(x*) and accumulate
        hat_V_i = Zs.mean(dim=0)  # Shape: [ntest]
        sum_V2 += hat_V_i.pow(2)
        sum_Zdiff2 += (Zs - hat_V_i.unsqueeze(0)).pow(2).sum(dim=0)

    # Correction factor: n(n-1)/(n-r)^2
    factor = (n - 1) / n * (n / (n - r))**2

    # Compute corrected variance terms
    term1 = factor * sum_V2
    term2 = factor * sum_Zdiff2 / (B * (B - 1))
    var_f = term1 - term2          # Bias-corrected variance estimate

    # Standard deviations
    sd_f_raw = torch.sqrt(term1)       # Without bias correction
    sd_f_correct = torch.sqrt(var_f)   # With bias correction


    return all_outputs, [hatf_B, sd_f_raw, sd_f_correct]



def run_one_repeat(rep_id, n, r, B, p, GLM_name, f_1, xtest):
    # 每个 repeat 训练 B 个网络，最后做一次 ensemble 预测
    ss = SampleSet(n, p, f_1, module=GLM_name)
    ss.get_sub_samples_with_validation(B, r)

    networks = train_multiple_networks(
        ss,
        input_size=p,
        hidden_size1=128,
        hidden_size2=64,
        mode=GLM_name,
        batchsize=256,
        num_epochs=500,
        learning_rate=0.1,
        weight_decay=0.02,
        dropout_rate=0.1,
        tol=0.0001,
        patience=7
    )

    _, Bf = ensemble_predict_batch_f(xtest, networks, ss)
    return Bf[0], Bf[1],Bf[2]







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   required=True)
    parser.add_argument("--index", type=float, required=True)
    args = parser.parse_args()


######Constant Area#######
    n        = args.n
    index_e  = args.index
    r        = int(n ** index_e)
    B        = 1400            # or your B
    p        = 7           # or your p
    GLM_name = "Poisson"  # or your mode      
    folder = 'resultspart4'  
    xtest    = torch.load(f"{folder}/xtest{p}.pt")  # 预先生成并保存

    repeats = 300

    # Parallel 
    results = Parallel(n_jobs=20)(
        delayed(run_one_repeat)(i, n, r, B, p, GLM_name, f_1, xtest)
        for i in range(repeats)
    )
    os.makedirs(folder, exist_ok=True)
    # 解包并 stack
    Bf0_tensor = torch.stack([res[0] for res in results])  # [100, ntest]
    Bf1_tensor = torch.stack([res[1] for res in results])
    Bf2_tensor = torch.stack([res[2] for res in results])

    # 保存到 CSV
    df_bf0 = pd.DataFrame(Bf0_tensor.numpy())
    df_bf1 = pd.DataFrame(Bf1_tensor.numpy())
    df_bf2 = pd.DataFrame(Bf2_tensor.numpy())

    fn0 = f"{folder}/{GLM_name}fBf1n{n}p{p}B{B}r{r}.csv"
    fn1 = f"{folder}/{GLM_name}sdf1nn{n}p{p}B{B}r{r}.csv"
    fn2 = f"{folder}/{GLM_name}sdcrtf1nn{n}p{p}B{B}r{r}.csv"
    df_bf0.to_csv(fn0, index=False, header=False)
    df_bf1.to_csv(fn1, index=False, header=False)
    df_bf2.to_csv(fn2, index=False, header=False)

    print(f"Done n={n}, index={index_e}, saved {fn0}, {fn1},{fn2}")




