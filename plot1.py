import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import cosine_similarity
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd


## load data
train1 = torch.load("./train_tensor_1.pt", map_location=torch.device('cpu')).float()
test1 = torch.load("./test_tensor_1.pt", map_location=torch.device('cpu')).float()

pred_ds_1 = np.load('./pred_ds_1.npy')
real_ds_1 = np.load('./real_ds_1.npy')
pred_te_1 = np.load('./pred_te_1.npy')
real_te_1 = np.load('./real_te_1.npy')

fig,ax = plt.subplots(1, 2)
a = ax[0].imshow(train1.detach().cpu()/14000,vmin=-1,vmax=1,origin='lower')
b = ax[1].imshow(test1.detach().cpu()/7000,vmin=-1,vmax=1,origin='lower')

plt.title("Effective Connectivity")
plt.colorbar(a, ax = [ax[0], ax[1]])
plt.show()

t_losses = np.load('./trainloss1.npy')
v_losses = np.load('./tesloss1.npy')
x = np.asarray(t_losses)
y = np.asarray(v_losses)
plt1 = plt.plot(x, label='trainloss')
plt2 = plt.plot(y, label='testloss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("epoch loss")
plt.show()

# correlation value
weight_tensor = torch.load("D:/research/sust_ncc/att_ec/models/data_use/weighted_values_10w.pt", map_location=torch.device('cpu')).float()
tra_m = train1 - torch.mean(train1)
te_m = test1 - torch.mean(test1)
w_m = weight_tensor - torch.mean(weight_tensor)
corr = torch.sum(tra_m * w_m) / (torch.sqrt(torch.sum(tra_m ** 2)) * torch.sqrt(torch.sum(w_m ** 2)))
print(corr) #-0.0899