import numpy as np
import torch
from numpy.linalg import solve, svd, norm
from scipy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import classic_kernel
from classic_kernel import convert_to_tensor
import time
from tqdm import tqdm
from sklearn.base import BaseEstimator
from copy import deepcopy

from sklearn.metrics import r2_score
from sklearn.svm import SVC

def matrix_sqrt(a):
    evalues, evectors = np.linalg.eig(a)
    sqrt_matrix = evectors @ np.diag(np.sqrt(np.abs(evalues))) @ evectors.T
    return np.real(sqrt_matrix)

class SVMKernel(BaseEstimator):

    def __init__(self): 
        self.svm_solver = None
        self.sqrt_M = None
        
    def fit(self, X_train, y_train, L, M, kernel=None):

        sqrt_M = matrix_sqrt(M)
        self.sqrt_M = sqrt_M

        X_train = X_train@sqrt_M

        if kernel=="laplace":
            self.svm_solver = SVC(kernel=classic_kernel.laplacian)
        else:
            self.svm_solver = SVC(gamma="auto")

        self.svm_solver.fit(X_train,np.argmax(y_train,axis=-1))
        return 

    def predict(self, X_test):
        trans_X_test = X_test@self.sqrt_M
        return self.svm_solver.predict(trans_X_test) 

class Kernel(BaseEstimator):
    def __init__(self, kernel="laplace"): 
        self.kernel=kernel
        self.alphas = None
        self.M = None
        self.L = None
        self.reg = None
        self.threshold = None
        self.best_t = -1
        self.top_k = None
        
    def fit(self, X_train, y_train, L, reg, M, use_lstsq=False, t=0.5, threshold=False, use_M_k=False, M_k=None):
        self.M = M
        self.L = L
        self.reg = reg
        self.use_M_k = use_M_k

        self.best_t = t   
        self.threshold = threshold
        if threshold:
            print("yes thresh")
            print("thresh is",t)
            k = int(t*len(M))

            # compute top k eigenvectors
            #evalues, evectors = np.linalg.eig(M)
            #top_evecs = evectors[:,-k:]

            #X_train = X_train@top_evecs.astype(np.float32)
            
            top_k = np.argsort(np.abs(np.diag(M)))[-k:]
            self.top_k = top_k
            
            X_train = X_train[:,top_k].astype(np.float32)
            
            if use_M_k:
                self.M_k = M_k
                M = M_k
            else:
                M = np.eye(k).astype(np.float32)
            
        if self.kernel=="laplace":
            K_train = laplace_kernel_M(X_train, X_train, L, M).numpy()
        elif self.kernel=="gaussian":
            sqrt_M = matrix_sqrt(M)
            self.sqrt_M = sqrt_M
            K_train = classic_kernel.gaussian(X_train@sqrt_M, X_train@sqrt_M, L).numpy()
        elif self.kernel=="ntk":
            sqrt_M = matrix_sqrt(M)
            self.sqrt_M = sqrt_M
            K_train = kernel(X_train@sqrt_M, X_train@sqrt_M).numpy()
        
        if use_lstsq:
            sol = lstsq(K_train + reg * np.eye(len(K_train)), y_train, lapack_driver = 'gelsd')[0].T
        else:
            sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        self.alphas = sol
        return 

    def predict(self, X_test, X_train, t=-1):
        L = self.L
        M = self.M
        
      
        if self.threshold:
            if t < 0:
                t = self.best_t
            else:
                print("no best t found")


            print("yes thresh")
            print("thresh is",t)
            k = int(t*len(M))

            #evalues, evectors = np.linalg.eig(M)
            #top_evecs = evectors[:,-k:]
            #X_train = X_train@top_evecs.astype(np.float32)
            #X_test = X_test@top_evecs.astype(np.float32)

            top_k = np.argsort(np.abs(np.diag(M)))[-k:]
            X_test = X_test[:,top_k].astype(np.float32)
            X_train = X_train[:,top_k].astype(np.float32)
            
            if self.use_M_k:
                M = self.M_k
            else:
                M = np.eye(k).astype(np.float32)
                

        if self.kernel=="laplace":
            K_test = laplace_kernel_M(X_train, X_test, L, M).numpy()
        elif self.kernel=="gaussian":
            K_test = classic_kernel.gaussian(X_train@self.sqrt_M, X_test@self.sqrt_M, L).numpy()
        elif self.kernel=="ntk":
            K_test = kernel(X_train@self.sqrt_M, X_test@self.sqrt_M).numpy()
        preds = (self.alphas @ K_test).T

        return preds 


def get_mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))


def kernel(pair1, pair2, nngp=False):
    pair1 = convert_to_tensor(pair1)
    pair2 = convert_to_tensor(pair2)

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


def get_grads(X, sol, L, P, center=False):
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
    
    if center:
        print("center")
        G -= torch.mean(G,dim=0)

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



def train(model, X_train, y_train, X_val, y_val, L, reg, iters=5, classification=True, 
            use_lstsq=False, use_svm=False, kernel=None, threshold=False, center=False, use_diagonal=False):
        
    if len(y_train.shape)==1:
        y_train = y_train[:,None]
        y_val = y_val[:,None]
        
    n, d = X_train.shape
    
    M = np.eye(d).astype('float32')
    


    if use_svm:
        model.fit(X_train,y_train,L=L,M=M, kernel=kernel)
        preds = model.predict(X_val)
    else:
        model.fit(X_train, y_train, L=L, reg=reg, M=M, use_lstsq=use_lstsq)
        preds = model.predict(X_val, X_train=X_train)

    if classification:
        if use_svm:
            y_pred = preds
        else:
            y_pred = np.argmax(preds, axis=-1)
        labels = np.argmax(y_val, axis=-1)
        count = np.sum(labels == y_pred)
        val_acc = count / len(labels)
        print("Init Acc: ", val_acc*100,"%")
        val_metric = val_acc
    else:
        val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
        val_metric = val_r2
        print("Init R2: ", val_r2)
        
    if not use_svm:
        val_mse = np.mean(np.square(preds - y_val))
        print("Init MSE: ", val_mse)
        
    best_val_metric = val_metric
    best_model = deepcopy(model)
    best_M = deepcopy(M)
    
    grad_model = Kernel()
    
    for i in range(iters):
        
        if isinstance(model,Kernel):
            print("Is kernel.")
            alphas = model.alphas
        elif use_svm:
            alphas = grad_model.alphas
        else:
            print("Is NOT kernel.")
            alphas = model.regressor_.alphas
        if len(alphas.shape)==1:
            alphas = alphas[None,:]
            
        M  = get_grads(X_train, alphas, L, M, center)
        if use_diagonal:
            M = np.diag(np.diag(M))
        
        if use_svm:
            model.fit(X_train,y_train,L=L,M=M, kernel=kernel)
            grad_model.fit(X_train, y_train, L=L, reg=reg, M=M)
            preds = model.predict(X_val)
        else:
            if kernel != "laplace":
                grad_model.fit(X_train, y_train, L=L, reg=reg, M=M)

            model.fit(X_train, y_train, L=L, reg=reg, M=M, use_lstsq=use_lstsq) 
            preds = model.predict(X_val, X_train=X_train)

        if classification:
            if use_svm:
                y_pred = preds
            else:
                y_pred = np.argmax(preds, axis=-1)
            labels = np.argmax(y_val, axis=-1)
            count = np.sum(labels == y_pred)
            val_acc = count / len(labels)
            print("Round " + str(i) + " Acc: ", val_acc*100,"%")
            val_metric = val_acc
        else:
            val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
            val_metric = val_r2
            print("Round " + str(i) + " R2: ", val_r2)
        
        if not use_svm:
            val_mse = np.mean(np.square(preds - y_val))
            print("Round " + str(i) + " MSE: ", val_mse)

        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            best_model = deepcopy(model)
            best_M = deepcopy(M)

   
    # treshold
    if threshold:
        print("Yes, threshold")
        ts = [0.05,0.1,0.2,0.4,0.6,0.8,0.95]
        best_val = -float("inf")
        best_t = None
        for t in ts:
            best_model.fit(X_train, y_train, L=L, reg=reg, M=best_M, t=t, threshold=True)
            preds = best_model.predict(X_val, X_train=X_train, t=t)

            if classification:
                y_pred = np.argmax(preds, axis=-1)
                labels = np.argmax(y_val, axis=-1)
                count = np.sum(labels == y_pred)
                val_acc = count / len(labels)
                val_metric = val_acc
            else:
                val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
                val_metric = val_r2
            if val_metric > best_val:
                best_val = val_metric
                best_t = t
            print("threshold: %.3f, Val metric: %.3f" %(t,val_metric))
            
        # Run RFM
        
        best_model.fit(X_train, y_train, L=L, reg=reg, M=best_M, t=best_t, threshold=True)
        
        if isinstance(model,Kernel):
            print("Is kernel.")
            alphas = best_model.alphas
            top_k = best_model.top_k
        else:
            print("Is NOT kernel.")
            alphas = best_model.regressor_.alphas
            top_k = best_model.regressor_.top_k
            
        if len(alphas.shape)==1:
            alphas = alphas[None,:]
            
        M_k  = get_grads(X_train[:,top_k].astype(np.float32), alphas, L, np.eye(len(top_k)).astype(np.float32) )
        
        best_model.fit(X_train, y_train, L=L, reg=reg, M=best_M, t=best_t, threshold=True, use_M_k=True, M_k=M_k)
        preds = model.predict(X_val, X_train=X_train)
        
        if classification:
            y_pred = np.argmax(preds, axis=-1)
            labels = np.argmax(y_val, axis=-1)
            count = np.sum(labels == y_pred)
            val_acc = count / len(labels)
            val_metric = val_acc
        else:
            val_r2 = r2_score(y_val.reshape(-1),preds.reshape(-1))
            val_metric = val_r2
        print("Tresh + RFM Score: ", val_metric)
    
    return best_model

