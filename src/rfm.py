import numpy as np
import torch
from numpy.linalg import solve, svd, norm
from scipy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import classic_kernel
import time
from tqdm import tqdm

from copy import deepcopy

from sklearn.metrics import r2_score

class Kernel:
    def __init__(self): 
        self.alphas = None
        self.M = None
        self.L = None
        
    def predict(self, X_train, X_test):
        K_test = laplace_kernel_M(X_train, X_test, self.L, self.M).numpy()
        preds = (self.alphas @ K_test).T
        return preds

def get_mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))


def kernel(pair1, pair2, nngp=False):
    out = pair1 @ pair2.transpose(1, 0)
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1)
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1)

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    if nngp:
        out = first
    else:
        sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
        out = first + sec

    return out

def laplace_kernel(pair1, pair2, bandwidth):
    return classic_kernel.laplacian(pair1,pair2, bandwidth)

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1,pair2, bandwidth, M)


def original_ntk(X_train, y_train, X_test, y_test, use_nngp=False):
    K_train = kernel(X_train, X_train, nngp=use_nngp).numpy()
    sol = solve(K_train, y_train).T
    K_test = kernel(X_train, X_test, nngp=use_nngp).numpy()
    y_pred = (sol @ K_test).T

    mse = get_mse(y_pred, y_test.numpy())
    if use_nngp:
        print("Original NNGP MSE: ", mse)
        return mse
    else:
        print("Original NTK MSE: ", mse)
        return mse


def get_grads(X, sol, L, P):
    M = 0.

    start = time.time()
    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)
    
    #"""
    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    #n, d = X.shape
    #x = np.random.normal(size=(1000, d))
    #x = torch.from_numpy(x)

    K = laplace_kernel_M(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L

    M = 0.
    
    bs = 20
    batches = torch.split(G, bs)
    for i in tqdm(range(len(batches))):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        #gradT = torch.swapaxes(grad, 1, 2)#.cuda()
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)

    M = M.numpy()

    end = time.time()
    
    print("Time: ", end - start)
    return M


def train(X_train, y_train, X_val, y_val, L, reg, iters=5, classification=True, use_lstsq=False):
    
        
    if len(y_train.shape)==1:
        y_train = y_train[:,None]
        y_val = y_val[:,None]
        
    n, d = X_train.shape
    
    M = np.eye(d).astype('float32')
    
    best_val_metric = -float("inf")
    best_M = None
    best_sol = None
    
    
    for i in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, M).numpy()
        
        if use_lstsq:
            sol = lstsq(K_train + reg * np.eye(len(K_train)), y_train, lapack_driver = 'gelsd')[0].T
        else:
            sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        K_val = laplace_kernel_M(X_train, X_val, L, M).numpy()
        preds = (sol @ K_val).T
        val_mse = np.mean(np.square(preds - y_val))
        print("Round " + str(i) + " MSE: ", val_mse)
        
        print("preds",preds.shape)
        print("val",y_val.shape)
        val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
        print("Round " + str(i) + " R2: ", val_r2)
        
        
        if classification:
            y_pred = preds
            preds = np.argmax(y_pred, axis=-1)
            labels = np.argmax(y_val, axis=-1)
            count = np.sum(labels == preds)
            val_acc = count/len(labels)
            print("Round " + str(i) + " Acc: ", val_acc*100,"%")
            val_metric = val_acc
        else:
            val_metric = val_r2
            
        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            best_sol = deepcopy(sol)
            best_M = deepcopy(M)
        
        M  = get_grads(X_train, sol, L, M)
        
    K_train = laplace_kernel_M(X_train, X_train, L, M).numpy()
    if use_lstsq:
        sol = lstsq(K_train + reg * np.eye(len(K_train)), y_train, lapack_driver = 'gelsd')[0].T
    else:
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
    K_val = laplace_kernel_M(X_train, X_val, L, M).numpy()
    preds = (sol @ K_val).T
    
    val_mse = np.mean(np.square(preds - y_val))
    print("Final MSE: ", val_mse)
    print("preds",preds.shape)
    print("val",y_val.shape)
    val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
    print("Final R2: ", val_r2)
        
        
    if classification:
        y_pred = preds
        preds = np.argmax(y_pred, axis=-1)
        labels = np.argmax(y_val, axis=-1)
        count = np.sum(labels == preds)
        val_acc = count / len(labels)
        print("Final Acc: ", val_acc*100,"%")
        val_metric = val_acc
    else:
        val_metric = val_r2
        
    if val_metric >= best_val_metric:
        best_val_metric = val_metric
        best_sol = deepcopy(sol)
        best_M = deepcopy(M)     
            
    model = Kernel()
    model.L = L
    model.M = best_M
    model.alphas = best_sol
    
    return model

