import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, solve, norm
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.svm import SVC
import classic_kernel
import eigenpro
    
def laplace_kernel(pair1, pair2, bandwidth=10):
    pair1.cuda()
    pair2.cuda()
    return classic_kernel.laplacian(pair1, pair2, bandwidth).cpu()

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

class Kernel:
    def __init__(self): 
        self.alphas = None
        self.M = None
        self.L = None
        
    def predict(self, X_train, X_test):
        K_test = laplace_kernel_M(X_train, X_test, self.L, self.M).numpy()
        preds = (self.alphas @ K_test).T
        return preds
    
def get_grads(X, sol, L, P):
    M = 0.

    start = time.time()
    num_samples = 10000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel(X, x, L)

    dist = classic_kernel.euclidean_distances(X, x, squared=False)
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

#     print("STEP 1: ", step1.shape)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

#     print("STEP 2: ", step2.shape)

    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ K).T

#     print("STEP 3: ", step3.shape)
    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

#     print("FINAL: ", step3.shape)

    G = (step2 - step3) * -1/L

    M = 0.

    bs = 5
    batches = torch.split(G, bs)
    for i in tqdm(range(len(batches))):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        T = gradT @ grad
        if T.shape[0] > 1:
            T = torch.sum(T, dim=0)
        else:
            # Pytorch couldn't handle summing tensors with 1 entry
            T = T[0]
        M += T.cpu()
        del grad, gradT, T
        torch.cuda.empty_cache()
    M /= len(G)

    M = M.numpy()

    end = time.time()
    return M


def convert_one_hot(y, c):
    o = np.zeros((y.size, c))
    o[np.arange(y.size), y] = 1
    return o

def eigenpro_solve(X_train, y_train, X_val, y_val, c, L, steps,
                   M, classification=True, kernel_fn=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")
    M = M.cuda()
    #kernel_fn = laplace_kernel
    if kernel_fn is None:
        kernel_fn = lambda x,y: laplace_kernel_M(x, y, bandwidth=L, M=M)
    else:
        kernel_fn = lambda x,y: ntk_relu_M(x, y, M)
        
    if len(y_train.shape)==1:
        y_train = y_train[:,None]
        y_val = y_val[:,None]
        
    model = eigenpro.FKR_EigenPro(kernel_fn, X_train, y_val.shape[1], device=device)
    res = model.fit(X_train, y_train, X_val, y_val,
                    epochs=list(range(steps)), mem_gb=12, classification=classification)
    best_val_metric = -float("inf")
    for r in res:
        if classification:
            val_metric = res[r][1]['multiclass-acc']
        else:
            val_metric = -1*res[r][1]['mse']
        if val_metric >= best_val_metric:
            best_val_metric = val_metric

    weight = model.weight.cpu().numpy()
    del model
    del M
    torch.cuda.empty_cache()
    
    if best_val_metric < 0:
        best_val_metric = -1*best_val_metric
        
    return weight, best_val_metric


def train(X_train, y_train, X_val, y_val, c=2, iters=3, ep_iter=40, L=10, classification=True):

    n, d = X_train.shape
    
    #y_train = convert_one_hot(y_train, c)
    #y_test = convert_one_hot(y_test, c)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')

    best_val_metric = 0
    best_test_metric = 0
    best_sol = None
    best_M = None
    
    M = np.eye(d, dtype='float32')
    M = torch.from_numpy(M)
    for i in range(iters-1):
        sol, val_metric = eigenpro_solve(X_train, y_train,
                                                X_val, y_val,
                                                c=c, L=L, steps=ep_iter, M=M,
                                                classification=classification)
        sol = sol.T

        if val_metric >= best_val_metric:
            best_val_metric = val_metric
            best_sol = sol
            best_M = M

        M  = get_grads(torch.from_numpy(X_train).float(), sol, L, M)
        M = torch.from_numpy(M)


    sol, val_metric = eigenpro_solve(X_train, y_train,
                                        X_val, y_val,
                                        c=c, L=L, steps=ep_iter, M=M, 
                                        classification=classification)
    if val_metric >= best_val_metric:
        best_val_metric = val_metric
        best_sol = sol
        best_M = M
        
    model = Kernel()
    model.alphas = sol.T
    model.M = best_M
    model.L = L
    
    return model

def ntk_relu(X, Z, depth=2, bias=0.):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.

    depth  (int): number of layers of the network
    bias (float): (default=0.)
    """
    from torch import acos
    pi = np.pi
    
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    eps = 0
    norm_x = X.norm(dim=-1)[:, None].clip(eps)
    norm_z = Z.norm(dim=-1)[None, :].clip(eps)
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(-1+eps,1-eps)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2

    return N



def ntk_relu_M(X, Z, M, depth=2, bias=0.):    
    from torch import acos
    pi = np.pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    eps = 0
    norm_x = (X @ M) * X
    norm_x = torch.sqrt(torch.sum(norm_x, dim=1, keepdim=True))
    norm_z = (Z @ M) * Z
    norm_z = torch.sqrt(torch.sum(norm_z, dim=1, keepdim=True).T)
    S = (X @ M) @  Z.T

    N = S + bias**2
    
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(-1+eps,1-eps)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N


def train_using_net(train_loader, val_loader, test_loader, c, Ms, ep_iter=100, L=10):
    X_train, y_train = get_data(train_loader)
    X_val, y_val = get_data(val_loader)
    X_test, y_test = get_data(test_loader)

    #y_train = convert_one_hot(y_train, c)
    #y_test = convert_one_hot(y_test, c)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    M = torch.from_numpy(Ms[0]).cuda()
    
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)


    best_val_acc = 0
    best_test_acc = 0

    sol, val_acc, test_acc = eigenpro_solve(X_train, y_train,
                                            X_val, y_val,
                                            X_test, y_test,
                                            c, L, ep_iter, M, kernel_fn=ntk_relu)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    print("BEST Val Acc: ", best_val_acc, "Best Test Acc: ", best_test_acc)
    return best_test_acc


def train_using_net_direct(train_loader, val_loader, test_loader, c, Ms, ep_iter=100, L=10):
    X_train, y_train = get_data(train_loader)
    X_val, y_val = get_data(val_loader)
    X_test, y_test = get_data(test_loader)

    #y_train = convert_one_hot(y_train, c)
    #y_test = convert_one_hot(y_test, c)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    """
    for M in Ms:
        start = time.time()
        U, s, Vt = svd(M, full_matrices=False)
        end = time.time()
        #print("SVD time: ", end - start, U.shape, s.shape, Vt.shape,
        #      X_train.shape, X_val.shape, X_test.shape, s, s.sum())
        s = np.sqrt(np.abs(s))
        sqrt_M = U @ np.diag(s) @ Vt

        #print("Mult 1")
        X_train = X_train @ sqrt_M
        #print("Mult 2")
        X_val = X_val @ sqrt_M
        #print("Mult 3")
        X_test = X_test @ sqrt_M
    #"""
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    M = torch.from_numpy(Ms[0])
    reg = 1e-1
    L = 10
    #K_train = laplace_kernel_M(X_train, X_train, M, L).numpy()

    K_train = ntk_relu_M(X_train, X_train, M).numpy()
    sol = solve(K_train + reg*np.eye(len(K_train)), y_train).T
    K_test = ntk_relu_M(X_train, X_test, M).numpy()
    #K_test = laplace_kernel_M(X_train, X_test, M, L).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()
    print("2 Layer NTK Acc: ", count / len(labels))


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0).numpy(), torch.cat(y, dim=0).numpy()
