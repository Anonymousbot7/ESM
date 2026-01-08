B=1400
GLM_name='Poisson'  # or your model
n = 700  # Main sample size
p = 10   # Number of features
r = int(n**0.9)  # Size of each sub-sample

folder='resultspart1'

 

import pandas as pd
import  numpy as np
import torch

def sigmoid(x):
    return 1/(1+np.exp(-x))

def f_1(x):
    return x[:, 0] + 0.25 * x[:, 1] ** 2+0.1*torch.tanh(0.5*x[:,2]-0.3)

PATH=folder+'/Bayesian/'
df_Ey_all_file = f"{PATH}/{GLM_name}Eyxn{n}.csv"
df_SD_all_file = f"{PATH}/{GLM_name}SDyxn{n}.csv"
df_lower_all_file = f"{PATH}/{GLM_name}lowern{n}.csv"
df_upper_all_file = f"{PATH}/{GLM_name}uppern{n}.csv"

df_bf1 = pd.read_csv(df_Ey_all_file, header=None)
df_sd  = pd.read_csv(df_SD_all_file, header=None)
df_lower  = pd.read_csv(df_lower_all_file, header=None)
df_upper  = pd.read_csv(df_upper_all_file, header=None)


df_bf1=df_bf1.to_numpy()
df_sd=df_sd.to_numpy()
df_lower=df_lower.to_numpy()
df_upper=df_upper.to_numpy()



xtest = torch.load(f"{folder}/xtest{p}.pt")


truef = torch.tensor(f_1(xtest), dtype=torch.float32)
true_f=truef.numpy()

if GLM_name=="Bernoulli":
    def activation_inv(y):
        return np.log(y / (1 - y))
    def activate(x):
        return 1/(1+np.exp(-x)) 
    
    
    estimated_f = df_bf1
    estimated_f0=estimated_f.mean(axis=0)
    std0=np.std(estimated_f,axis=0)

    true_p=true_f

    std=df_sd 
    

   
    cover=((true_f >= df_lower) & (true_f <= df_upper))
    
    AIL=np.nanmean(activate(df_upper)-activate(df_lower))
    AILsd=np.nanstd(activate(df_upper)-activate(df_lower))
    # print(df_upper-df_lower)
    
    # within_confidence_interval = ((true_f >= estimated_f-1.96*std) & (true_f <= estimated_f+1.96*std))
elif GLM_name=="Poisson":
    def activate(x):
        return np.exp(x)
    def activation_inv(y):
        return np.log(y)
    true_f=np.log(np.log(1+np.exp(true_f)))
    estimated_f = (df_bf1)
    estimated_f0 = np.nanmean(estimated_f, axis=0)
    std0=np.nanstd(estimated_f,axis=0)

    true_p=activate(true_f)

    std=df_sd 
    # print(estimated_f0)
    # print(true_f)
    std0=np.nanstd(estimated_f,axis=0)

   
    cover=((true_f>= df_lower) & (true_f <= df_upper))
    
    AIL=np.nanmean(activate(df_upper)-activate(df_lower))
    AILsd=np.nanstd(activate(df_upper)-activate(df_lower))
    # print(df_upper-df_lower)

else: 
    def activate(x):
        return 1/(1+np.exp(-x))
    true_f=truef.numpy()

    estimated_f = tensor_bf1.numpy()
    estimated_f0=estimated_f.mean(axis=0)
    
    quantile=np.quantile(np.abs(estimated_f0-true_f),maintain)
    index=(np.abs(estimated_f0-true_f)<=quantile)
    estimated_f = estimated_f[:,index]
    estimated_f0=estimated_f.mean(axis=0)
    std=tensor_sd.numpy()[:,index]
    stdcrt=tensor_sdcrt.numpy()[:,index]
    std0=np.std(estimated_f,axis=0)
    std1=np.mean(stdcrt,axis=0)
    true_f=true_f[index]
   
    cover=((true_f >= estimated_f-1.96*stdcrt) & (true_f <= estimated_f+1.96*stdcrt))
    cover0=((estimated_f0 >= estimated_f-1.96*stdcrt) & (estimated_f0 <= estimated_f+1.96*stdcrt))
    cover1=((true_f >= estimated_f-1.96*std1) & (true_f <= estimated_f+1.96*std1))
    AIL=np.mean(activate(estimated_f+1.96*stdcrt)-activate(estimated_f-1.96*stdcrt))
    AILsd=np.std(activate(estimated_f+1.96*stdcrt)-activate(estimated_f-1.96*stdcrt))


print("biasf:", np.nanmean(estimated_f0 - true_f),
      "(sd):", np.nanstd(estimated_f - true_f))

print("MAEf:", np.nanmean(np.abs(estimated_f - true_f)),
      "(sd):", np.nanstd(np.abs(estimated_f - true_f)))

print("biaspsi:", np.nanmean(activate(estimated_f) - activate(true_f)),
      "sd:", np.nanstd(activate(estimated_f) - activate(true_f)))

print("MAEpsi:", np.nanmean(np.abs(activate(estimated_f) - activate(true_f))),
      "SD:", np.nanstd(np.abs(activate(estimated_f) - activate(true_f))))

print('The empirical std:',np.mean(std0))
# print('SE:',np.mean(std))
print("coverprob", np.nanmean(np.nanmean(cover, axis=0)))
print("AIL:", np.nanmean(AIL))
print("AILsd:",AILsd)




