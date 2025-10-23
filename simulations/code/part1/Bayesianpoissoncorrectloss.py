import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import Counter

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from joblib import Parallel, delayed
import pandas as pd
import argparse
import errno

import pickle

def f_1(x):
    return x[:, 0] + 0.25 * x[:, 1] ** 2+0.1*torch.tanh(0.5*x[:,2]-0.3)


class my_Net_relu(torch.nn.Module):
    def __init__(self):
        super(my_Net_relu, self).__init__()
        self.gamma = []
        self.fc1 = nn.Linear(10, 128)
        self.gamma.append(torch.ones(self.fc1.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc1.bias.shape, dtype=torch.float32))
        self.fc2 = nn.Linear(128, 100)
        self.gamma.append(torch.ones(self.fc2.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc2.bias.shape, dtype=torch.float32))
        self.fc3 = nn.Linear(100, 10)
        self.gamma.append(torch.ones(self.fc3.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc3.bias.shape, dtype=torch.float32))
        self.fc4 = nn.Linear(10, 1)
        self.gamma.append(torch.ones(self.fc4.weight.shape, dtype=torch.float32))
        self.gamma.append(torch.ones(self.fc4.bias.shape, dtype=torch.float32))

    def to(self, *args, **kwargs):
        super(my_Net_relu, self).to(*args, **kwargs)
        parsed = torch._C._nn._parse_to(*args, **kwargs)
        device, dtype, non_blocking = parsed[:3]
        for index in range(self.gamma.__len__()):
            self.gamma[index] = self.gamma[index].to(device)

    def forward(self, x):
        for i, para in enumerate(self.parameters()):
            para.data.mul_(self.gamma[i])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def mask(self, user_gamma, device):
        for i, para in enumerate(self.parameters()):
            if self.gamma[i].shape != user_gamma[i].shape:
                print('size doesn\'t match')
                return 0
        for i, para in enumerate(self.parameters()):
            self.gamma[i].data = torch.tensor(user_gamma[i], dtype=torch.float32).to(device)


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGHMC(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, temperature = 1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, temperature = temperature)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGHMC, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            temperature = group['temperature']
            lr = group['lr']

            alpha = 1 - momentum
            scale = np.sqrt(2.0*alpha*temperature/lr)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.add_(torch.ones_like(buf).normal_().mul(scale))
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                else:
                    d_p = d_p.add(torch.ones_like(d_p).normal_().mul(scale))

                p.data.add_(-group['lr'], d_p)

        return loss






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









def main():
    r=int(n ** 0.9)
    subn = r
    



    TotalP = 10
    p=TotalP
    
    folder = 'resultspart1'  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    NTrain= r
    Nval = n-r
    NTest = 80


    prior_sigma_0_init = 0.00005

    prior_sigma_0 = prior_sigma_0_init


    prior_sigma_0_anneal = 0.000001

    prior_sigma_1 = 0.01

    lambda_n = 0.0000001


   
    repeats = 300
    Ey_all, SD_all, lower_all, upper_all = None, None, None, None

    for repeat0 in range(repeats):
        ss = SampleSet(n, p, f_1, module=GLM_name)

        ss.get_sub_samples_with_validation(1, r)




        x_train = ss.subtrain[0][0]
        y_train = ss.subtrain[0][1].unsqueeze(1)


        x_val = ss.subval[0][0]
        y_val = ss.subval[0][1].unsqueeze(1)
        print(x_train.shape)

        x_test    = torch.load(f"{folder}/xtest10.pt")
        z = f_1(x_test)
        rate_param = torch.log(1+torch.exp(z))   # Ensure the rate parameter is positive
        y_test = torch.poisson(rate_param).unsqueeze(1)

        x_test_confidence = x_test
        y_test_confidence = y_test

        num_seed = 1


        num_selection_list = np.zeros([num_seed])
        num_selection_true_list = np.zeros([num_seed])
        train_loss_list = np.zeros([num_seed])
        val_loss_list = np.zeros([num_seed])
        test_loss_list = np.zeros([num_seed])

        for my_seed in range(num_seed):
            
            net = my_Net_relu()

            net.to(device)
            loss_func = nn.PoissonNLLLoss(log_input=True)


            sigma = torch.FloatTensor([1]).to(device)

            c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
            c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
            threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                    0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

            PATH = folder + '/Bayesian/'

            if not os.path.isdir(PATH):
                try:
                    os.makedirs(PATH)
                except OSError as exc:  # Python >2.5
                    if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                        pass
                    else:
                        raise

            show_information = 500


            step_lr = 0.005
            step_momentum = 0.9

            temperature = 0

            optimization = SGHMC(net.parameters(), lr=step_lr, momentum=step_momentum, weight_decay=0, temperature=temperature)

            max_loop = 2001


            anneal_start = 50

            anneal_end = 200

            prior_anneal_end = 600



            para_path = []
            para_gamma_path = []
            for para in net.parameters():
                para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
                para_gamma_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

            train_loss_path = np.zeros([max_loop])
            val_loss_path = np.zeros([max_loop])
            test_loss_path = np.zeros([max_loop])

            confidence_interval = 100

            train_output_path = np.zeros([max_loop // confidence_interval + 1, NTrain])
            test_output_path = np.zeros([max_loop // confidence_interval + 1, NTest])

            for iter_index in range(max_loop):
                if subn == NTrain:
                    subsample = range(NTrain)
                else:
                    subsample = np.random.choice(range(NTrain), size=subn, replace=False)

                if iter_index < anneal_start:
                    anneal_lambda = 0
                    temperature = 0
                    for para in optimization.param_groups:
                        para['temperature'] = 1.0 * temperature / NTrain

                elif iter_index < anneal_end:
                    anneal_lambda = iter_index * 1.0 / anneal_end
                    temperature = 0.1
                    for para in optimization.param_groups:
                        para['temperature'] = 1.0 * temperature / NTrain

                else:
                    anneal_lambda = 1
                    if iter_index <= prior_anneal_end:
                        temperature = 0.1
                    else:
                        temperature = 0.1 * 1.0 / (iter_index - prior_anneal_end)
                    for para in optimization.param_groups:
                        para['temperature'] = 1.0 * temperature / NTrain


                if iter_index < anneal_end:
                    prior_sigma_0 = prior_sigma_0_init
                if iter_index >= anneal_end and iter_index < prior_anneal_end:
                    prior_sigma_0 = (iter_index - anneal_end)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_anneal + (prior_anneal_end - iter_index)*1.0/(prior_anneal_end - anneal_end) * prior_sigma_0_init
                if iter_index >= prior_anneal_end:
                    prior_sigma_0 = prior_sigma_0_anneal

                c1 = np.log(lambda_n) - np.log(1 - lambda_n) + 0.5 * np.log(prior_sigma_0) - 0.5 * np.log(prior_sigma_1)
                c2 = 0.5 / prior_sigma_0 - 0.5 / prior_sigma_1
                threshold = np.sqrt(np.log((1 - lambda_n) / lambda_n * np.sqrt(prior_sigma_1 / prior_sigma_0)) / (
                        0.5 / prior_sigma_0 - 0.5 / prior_sigma_1))

                net.zero_grad()
                output = net(x_train[subsample,])
                loss = loss_func(output, y_train[subsample,])

                train_loss_path[iter_index] = loss.cpu().data.numpy()

                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))

                loss.backward()

                # prior gradient
                with torch.no_grad():
                    for para in net.parameters():
                        temp = para.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = para.div(-prior_sigma_0).mul(temp) + para.div(-prior_sigma_1).mul(1 - temp)
                        prior_grad = temp.div(NTrain)
                        para.grad.data -= anneal_lambda * prior_grad


                optimization.step()

                with torch.no_grad():
                    output = net(x_val)
                    loss = loss_func(output, y_val)
                    val_loss_path[iter_index] = loss.cpu().data.numpy()
                    output = net(x_test)
                    loss = loss_func(output, y_test)
                    test_loss_path[iter_index] = loss.cpu().data.numpy()


                if iter_index % confidence_interval == 0:
                    with torch.no_grad():
                        output = net(x_train)
                        train_output_path[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()

                        output = net(x_test_confidence)
                        test_output_path[iter_index//confidence_interval, :] = output.view(-1).cpu().data.numpy()

                if iter_index % show_information == 0:
                    print('iteration:', iter_index)
                    with torch.no_grad():

                        print("train loss:", train_loss_path[iter_index])

                        print("val loss:", val_loss_path[iter_index])

                        print("test loss:", test_loss_path[iter_index])

                        print('sigma:', sigma)

                        for i, para in enumerate(net.parameters()):
                            para_path[i][iter_index // show_information,] = para.cpu().data.numpy()
                            para_gamma_path[i][iter_index // show_information,] = (para.abs() > threshold).cpu().data.numpy()

                        print('number of 1:', np.sum(np.max(para_gamma_path[0][iter_index // show_information,], 0) > 0))
                        print('number of true:',
                            np.sum((np.max(para_gamma_path[0][iter_index // show_information,], 0) > 0)[0:5]))


            # filename = PATH + 'data' + "_simu_"    + str(subn) + '_' + str(
            #     lambda_n) + '_' + str(prior_sigma_0) + '_' + str(prior_sigma_1) + '.txt'
            # f = open(filename, 'wb')
            # pickle.dump([para_path, para_gamma_path, train_loss_path, val_loss_path, test_loss_path, train_output_path, test_output_path], f, protocol=4)
            # f.close()

            num_selection_list[my_seed] = np.sum(np.max(para_gamma_path[0][-1,], 0) > 0)
            num_selection_true_list[my_seed] = np.sum((np.max(para_gamma_path[0][-1,], 0) > 0)[0:5])

            user_gamma = []
            for index in range(para_gamma_path.__len__()):
                user_gamma.append(para_gamma_path[index][-1,])

            with torch.no_grad():
                for i, para in enumerate(net.parameters()):
                    para.data = torch.FloatTensor(para_path[i][-1,]).to(device)

            net.mask(user_gamma, device)

            fine_tune_loop = 401


            para_path_fine_tune = []
            para_gamma_path_fine_tune = []

            for para in net.parameters():
                para_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))
                para_gamma_path_fine_tune.append(np.zeros([fine_tune_loop // show_information + 1] + list(para.shape)))


            train_loss_path_fine_tune = np.zeros([fine_tune_loop ])
            val_loss_path_fine_tune = np.zeros([fine_tune_loop ])
            test_loss_path_fine_tune = np.zeros([fine_tune_loop ])


            train_output_path_fine_tune = np.zeros([fine_tune_loop // confidence_interval + 1, NTrain])
            test_output_path_fine_tune = np.zeros([fine_tune_loop // confidence_interval + 1, NTest])


            step_lr = 0.005
            step_momentum = 0.9
            optimization = torch.optim.SGD(net.parameters(), lr=step_lr, momentum=step_momentum, weight_decay=0)


            Ey_given_X = np.mean(test_output_path, axis=0) 
            SDy_given_X = np.std(test_output_path, axis=0) 
            lower_q = np.quantile(test_output_path, 0.025, axis=0)
            upper_q = np.quantile(test_output_path, 0.975, axis=0)
        if Ey_all is None:
            Ey_all = Ey_given_X[None, :]      # [1, NTest]
            SD_all = SDy_given_X[None, :]
            lower_all = lower_q[None, :]
            upper_all = upper_q[None, :]
        else:
            Ey_all = np.vstack([Ey_all, Ey_given_X])
            SD_all = np.vstack([SD_all, SDy_given_X])
            lower_all = np.vstack([lower_all, lower_q])
            upper_all = np.vstack([upper_all, upper_q])
        df_Ey_all= pd.DataFrame(Ey_all)
        df_SD_all = pd.DataFrame(SD_all)
        df_lower_all = pd.DataFrame(lower_all)
        df_upper_all = pd.DataFrame(upper_all)

        df_Ey_all_file = f"{PATH}/{GLM_name}Eyxn{n}.csv"
        df_SD_all_file = f"{PATH}/{GLM_name}SDyxn{n}.csv"
        df_lower_all_file = f"{PATH}/{GLM_name}lowern{n}.csv"
        df_upper_all_file = f"{PATH}/{GLM_name}uppern{n}.csv"
  
        df_Ey_all.to_csv(df_Ey_all_file, index=False, header=False)
        df_SD_all.to_csv(df_SD_all_file, index=False, header=False)
        df_lower_all.to_csv(df_lower_all_file, index=False, header=False)
        df_upper_all.to_csv(df_upper_all_file, index=False, header=False)


            
            


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n",     type=int,   required=True)
    # args = parser.parse_args()


    n= 400
    GLM_name = 'Poisson'
    main() 