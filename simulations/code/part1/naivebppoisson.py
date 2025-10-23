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


def f_1(x):
    return x[:, 0] + 0.25 * x[:, 1] ** 2+0.1*torch.tanh(0.5*x[:,2]-0.3)



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
    
    def get_bootstrap_sample(self):
        n = self.X.shape[0]


        indices = torch.randint(low=0, high=n, size=(n,), dtype=torch.long)

    
        X_boot = self.X[indices]
        Y_boot = self.Y[indices]

        return X_boot, Y_boot
    
    
    def save(self, file_path):
        """Saves the SampleSet instance to a file."""
        torch.save(self, file_path)
    
    @staticmethod
    def load(file_path):
        """Loads a SampleSet instance from a file."""
        return torch.load(file_path)



def clear_folder(folder_path):
    """
    删除文件夹中的所有内容，但保留文件夹本身
    :param folder_path: 文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
 
        if os.path.isfile(item_path):
            os.remove(item_path)

        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"clear {folder_path}")


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
    train_samples  = sample_set.get_sample_set()
    
    
    networks = []  # List to hold trained neural networks
    for i in range(B):
        X_sub, Y_sub= sample_set.get_bootstrap_sample()
        
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

    
    return networks


def ensemble_predict_batch_f(Xtest, networks):
    ntest = Xtest.shape[0]

    B = len(networks)      # Number of sub-samples (number of neural networks)

    
    all_outputs = torch.zeros(ntest, B)
    with torch.no_grad():
        for j, net in enumerate(networks):
            logits = net(Xtest).squeeze(1)
            all_outputs[:, j] = logits

   
    return all_outputs



def run_one_repeat(rep_id, n, B, p, GLM_name, f_1, xtest):
    # 每个 repeat 训练 B 个网络，最后做一次 ensemble 预测
    ss = SampleSet(n, p, f_1, module=GLM_name)
    train_samples,Y= ss.X, ss.Y

    networks = train_multiple_networks(
        ss,
        input_size=p,
        hidden_size1=128,
        hidden_size2=64,
        mode=GLM_name,
        batchsize=100,
        num_epochs=700,
        learning_rate=0.1,
        weight_decay=0.02,
        dropout_rate=0.1,
        tol=0.0001,
        patience=7
    )
    Af = ensemble_predict_batch_f(xtest, networks)
    mean_pred = Af.mean(dim=1)
    std_pred = Af.std(dim=1, unbiased=False)
    q_lower = torch.quantile(Af, 0.025, dim=1)
    q_upper = torch.quantile(Af, 0.975, dim=1)
    return mean_pred,std_pred,q_lower,q_upper







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   required=True)

    args = parser.parse_args()
    n        = args.n

    # n=700
######Constant Area#######


    B        = 400      # or your B
    p        = 10           # or your p
    GLM_name = "Poisson"  # or your mode  
    folder = 'resultspart1' 

    xtest    = torch.load(f"{folder}/xtest10.pt")  # 预先生成并保存

    repeats = 300


    # Parallel 
    results = Parallel(n_jobs=23)(
        delayed(run_one_repeat)(i, n, B, p, GLM_name, f_1, xtest)
        for i in range(repeats)
    )

    # 解包并 stack
    Bf0_tensor = torch.stack([res[0] for res in results])  
    Bf1_tensor = torch.stack([res[1] for res in results])
    Bf2_tensor = torch.stack([res[2] for res in results])  
    Bf3_tensor = torch.stack([res[3] for res in results])

    df_bf0 = pd.DataFrame(Bf0_tensor.numpy())
    df_bf1 = pd.DataFrame(Bf1_tensor.numpy())
    df_bf2 = pd.DataFrame(Bf2_tensor.numpy())
    df_bf3 = pd.DataFrame(Bf3_tensor.numpy())
 

    fn0 = f"{folder}/naivebp/{GLM_name}En{n}p{p}.csv"
    fn1 = f"{folder}/naivebp/{GLM_name}stdn{n}p{p}.csv"
    fn2 = f"{folder}/naivebp/{GLM_name}lowern{n}p{p}.csv"
    fn3 = f"{folder}/naivebp/{GLM_name}uppern{n}p{p}.csv"
    df_bf0.to_csv(fn0, index=False, header=False)
    df_bf1.to_csv(fn1, index=False, header=False)
    df_bf2.to_csv(fn2, index=False, header=False)
    df_bf3.to_csv(fn3, index=False, header=False)
   