import numpy as np
import matplotlib.pyplot as plt
import torch
#from nilearn import plotting
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score
import scipy.io as scio
import matplotlib.colors as colors
#from nilearn.connectome import ConnectivityMeasure

# FI curve
def h(x):
    return np.tanh(x)

def integrate(W, X, length):
    ''' Euler integrate eq (1)
    
    Parameters
    ----------
    W : (n,n) numpy array (connectivity)
    X : (n) numpy array Initial condition 
    '''
    T = length    # total integration time
    dt = 1   # time-step
    t = np.arange(0,T,dt)
    np.random.seed(29)
    
    # allocate array for network states
    x = np.zeros((len(X),len(t)))
    print(x.shape)
    # set initial condition
    x[:,0] = X
    # Euler integration
    for i in range(1,len(t)):
        x[:,i] = x[:,i-1] + dt*(-x[:,i-1]+W.dot(h(x[:,i-1]))) + np.random.normal(0,0.1,(1,len(X)))
    return t, x

def integrate_no_noise(W, X, length):
    ''' Euler integrate eq (1)
    
    Parameters
    ----------
    W : (n,n) numpy array (connectivity)
    X : (n) numpy array Initial condition 
    '''
    T = length    # total integration time
    dt = 0.01   # time-step
    t = np.arange(0,T,dt)
    np.random.seed(29)
    
    # allocate array for network states
    x = np.zeros((len(X),len(t)))
    # set initial condition
    x[:,0] = X
    # Euler integration
    for i in range(1,len(t)):
        x[:,i] = x[:,i-1] + dt*(-x[:,i-1]+W.dot(h(x[:,i-1]))) 
    return t, x

def plot_spectrum(W,ax):    
    ''' Make a scatter plot of the eigenvalues of W
    in the complex plane
    
    Parameters
    ----------
    W :(n,n) numpy array
    ax : pyplot ax
    '''
        
    # get eigen-values
    L = np.linalg.eig(W)[0]
    
    # scatter real part vs imaginary part
    ax.scatter(np.real(L),np.imag(L))
    
    # add some lines for reference
    ax.axvline(0,c='k',lw=0.5)
    ax.axvline(1,c='k',lw=0.5)
    ax.axhline(0,c='k',lw=0.5)
    # and some circles
    theta = np.arange(0,7,0.01)
    ax.plot(np.cos(theta), np.sin(theta),c='k',lw=0.5)
    # and labels
    ax.set_xlabel(r'Re($\lambda$)')
    ax.set_ylabel(r'Im($\lambda$)')
    ax.axis('equal')
    
def get_corr(true_ec, inferred_ec):
    n = true_ec.shape[0]
    true_ec_flatten = []
    inferred_ec_flatten = []
    for i in range(n):
        for j in range(n):
            if i!=j:
                true_ec_flatten.append(true_ec[i,j])
                inferred_ec_flatten.append(inferred_ec[i,j])
    true_ec_flatten = np.array(true_ec_flatten)
    inferred_ec_flatten = np.array(inferred_ec_flatten)
    true_ec_flatten /= np.max(np.abs(true_ec_flatten))
    inferred_ec_flatten /= np.max(np.abs(inferred_ec_flatten))
    corr = np.corrcoef(true_ec_flatten, inferred_ec_flatten)[0,1]
    return corr, true_ec_flatten, inferred_ec_flatten


n = 20
s = 1
sigma = s/np.sqrt(n)
length = 100000
W = np.random.RandomState(0).normal(0,sigma,(n,n)) # generate connectivity
X = np.random.RandomState(0).normal(0,sigma,(n)) # generate random intial condition
print(W.shape, X.shape)
t, x = integrate(W,X,length)
W = torch.tensor(W)

ts_mean = X.mean()
ts_std = X.std()
ts_max = X.max()
ts_min = X.min()
print(ts_mean, ts_std, ts_max, ts_min)
#np.savetxt('1000data.txt', x)
# fig, ax = plt.subplots(figsize=(9,3))  
# ax.plot(t[:10000], x.T[1000:11000,:10])
# plt.show()

# subsampled_x = []
# for i in range(1000, length*100):
#     if i%100 == 0:
#         subsampled_x.append(x.T[i,:])
# subsampled_x = np.array(subsampled_x)

# fig, ax = plt.subplots(figsize=(9,3))  
# ax.plot(subsampled_x[:100,:10])
# plt.show()
#####################################################
# torch.save(W, 'weighted_values_10w.pt')
plt.imshow(W*3, vmin=-1, vmax=1, origin='lower')
plt.title('weighted value')
plt.xlabel('')
plt.ylabel('')
plt.colorbar()
plt.show()