import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SpatialSelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, H):
        N, T = H.shape[0],H.shape[1]
        Q = self.query_layer(H)
        K = self.key_layer(H).transpose(-1,-2)
        A = torch.matmul(Q, K)
        A /= torch.sqrt(torch.tensor(K.size(-1)).float())
        #A = self.softmax_layer(A)
        H_ = torch.matmul(A, H)
        return H_ , A


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.attention_layer = SpatialSelfAttention(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x, a = self.attention_layer(x)
        #x = self.norm(x)
        return x, a


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attention_layer = SpatialSelfAttention(input_dim, hidden_dim)

    def forward(self, x):
        #x = self.attention_layer(x)
        #x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim,hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_dim, hidden_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(input_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        encoder_out = x
        for layer in self.encoder_layers:
            encoder_out, enc_a = layer(encoder_out)
        for layer in self.decoder_layers:
            decoder_out = layer(encoder_out)
        out = self.fc_layer(encoder_out)  ##############################
        return out, [enc_a]

# load data
data_ori = np.loadtxt('/100000data.txt')
# input_dim = data_ori.shape[1]
# seq_len = data_ori.shape[0]
input_dim = data_ori.shape[0]
seq_len = data_ori.shape[1]
split_ratio = 0.7
hidden_dim = 65
num_layers = 2
num_heads = 4
batch_size = 1

train_len = int(split_ratio * seq_len)
test_len = seq_len - train_len
#train_data, test_data = data_ori[:train_len,:], data_ori[test_len:,:]
train_data, test_data = data_ori[:,:train_len], data_ori[:,train_len:]
step =  100

## split and assign data into train and test
s = torch.tensor([])
p = torch.tensor([])
for i in range(train_len-step):
    seq = torch.tensor(train_data[:,i:i+step].reshape(batch_size, input_dim, step), dtype = torch.float)
    pre = torch.tensor(train_data[:,i+step:i+step+1].reshape(batch_size, input_dim, 1), dtype = torch.float)
    s=torch.concat([s,seq])
    p=torch.concat([p,pre])

ss = torch.tensor([])
pp = torch.tensor([])
for i in range(test_len-step-1):
    sequ = torch.tensor(test_data[:,i:i+step].reshape(batch_size, input_dim, step), dtype = torch.float)
    pred = torch.tensor(test_data[:,i+step:i+step+1].reshape(batch_size, input_dim, 1), dtype = torch.float)
    ss=torch.concat([ss,sequ])
    pp=torch.concat([pp,pred])

train_ds, test_ds = TensorDataset(s, p), TensorDataset(ss, pp)
train_loader, test_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = Transformer(step, hidden_dim, num_layers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
learning_rate = 4e-4
n_epochs = 20
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

## train data
## train data
t_losses, v_losses = [], []
pred_ds = []
real_ds = []
pred_te = []
real_te = []

for epoch in range(n_epochs):
    train_loss, valid_loss = 0.0, 0.0
    model.train()
    a_map_train = torch.zeros((input_dim, 20), device = device)
    a_map_train2 = torch.zeros((input_dim, 20),  device = device)
    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.squeeze().to(device)
        preds, a = model(x)
        preds = preds.squeeze()
        loss = criterion(preds, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        a_map_train += a[0].squeeze().to(device)
        #a_map_train2 += a[1].squeeze().to(device)
    pred_ds.append(preds)
    real_ds.append(y)
    
    epoch_loss = train_loss / len(train_loader)
    t_losses.append(epoch_loss)
    model.eval()
    a_map_test = torch.zeros((input_dim, 20), device = device)
    a_map_test2 = torch.zeros((input_dim, 20), device = device)
    for x, y in test_loader:
        with torch.no_grad():
            x, y = x.to(device), y.squeeze().to(device)
            preds, aa = model(x)
            preds = preds.squeeze()
            error = criterion(preds, y)
        valid_loss += error.item()
        a_map_test += aa[0].squeeze().to(device)
        #a_map_test2 += aa[1].squeeze().to(device)
    valid_loss = valid_loss / len(test_loader)
    v_losses.append(valid_loss)
    pred_te.append(preds)
    real_te.append(y)
    print(f'{epoch}-train: {epoch_loss}, valid:{valid_loss}')

torch.save(a_map_train, "train_tensor_3.pt")
torch.save(a_map_test, "test_tensor_3.pt")

pred_ds = np.array([p.detach().cpu().numpy().flatten() for p in pred_ds])
real_ds = np.array([r.detach().cpu().numpy().flatten() for r in real_ds])
pred_te = np.array([pt.detach().cpu().numpy().flatten() for pt in pred_te])
real_te = np.array([rt.detach().cpu().numpy().flatten() for rt in real_te])
np.save('pred_ds_3.npy', pred_ds)
np.save('real_ds_3.npy', real_ds)
np.save('pred_te_3.npy', pred_te)
np.save('real_te_3.npy', real_te)

trainloss3 = np.array(t_losses)
testloss3 = np.array(v_losses)
np.save('trainloss3.npy', trainloss3)
np.save('testloss3.npy', testloss3)


## plot
for i in range(input_dim):
    seq_pred = pred_ds[i][:]
    seq_real = real_ds[i][:]
    te_pred = pred_te[i][:]
    te_real = real_te[i][:]

    plt.plot(seq_pred, label ='Pred train 3')
    plt.plot(seq_real, label ='Real train 3')
    plt.plot(te_pred, label ='Pred test 3')
    plt.plot(te_real, label ='Real test 3')
    plt.legend()
    plt.show()
    plt.savefig('legend_3'+ i + '.png')
