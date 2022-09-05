import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.linear_model import LinearRegression
from baselines import *
#grad_steps = 500 # old?

from numba import jit

# import torch

# # regularization parameter, to be tuned
# lam = 1.
# # number of data points
# n = 4
# # dimension
# d = 3
# # number of components in a corner
# m = 2
# # learning rate
# lr = 2.

# decomposes weight vector a into convex combination of "m-corners"
# def corners(a,m):
#     # print max(a), 1./m
#     n = len(a)
#     corners_list = [0]*n
#     coeffs = [0]*n
#     acopy = a.copy()
#     total = 0
#     while np.sum(acopy) > 1e-5:
#         bdys = np.array([np.abs(acopy[i] - np.sum(acopy)/float(m)) < 1e-8 for i in range(n)])
#         num_comps = sum(bdys)
#         if num_comps < m:
#             for i in range(n):
#                 if bdys[i] != True and acopy[i] > 1e-8 and num_comps < m:
#                     bdys[i] = True
#                     num_comps += 1
#         r = np.array([1./m if bdys[i] else 0. for i in range(n)])
#         s = min(acopy[bdys])
#         l = max(acopy[~bdys])
#         p = min(m*s,np.sum(acopy) - m*l)
#         # print "info:", np.sum(acopy), m, l, s, p, bdys
#         acopy -= p*r
#         corners_list[total] = list(bdys)
#         coeffs[total] = p
#         total += 1
#         # print acopy, r, np.sum(acopy)
#     return corners_list[:total], coeffs[:total]


### Could check Herbster and Warmuth paper for linear time algo
@jit(nopython=True)
def cap(a,m):
    if np.max(a) <= 1./m:
        return a
    #sorted_a = np.sort(a)
    ## BUG FIX: use -a to get descending order
    sort_ids = np.argsort(-a)

    # Faster code
    Z = np.sum(a)
    for i in range(1,m + 1):
        Z -= a[sort_ids[i - 1]]
        aprime_next = (m - i) * a[sort_ids[i]]/float(m * Z)
        if aprime_next <= 1.0/m:
            aprime = np.copy(a)
            aprime[sort_ids[:i]] = 1./m
            aprime[sort_ids[i:]] *= (m - i)/float(m * Z)
            return aprime



    ## Old, slow code
    # Can start from 1 due to initial check, we know at least one number is too big
    #for i in range(1,m):
    #    #aprime[:] = a
    #    np.copyto(aprime,a)
    #    aprime[sort_ids[:i]] = 1./m
    #    #for j in range(i):
    #        #aprime[sort_ids[j]] = 1./m
    #    Z = np.sum(aprime[sort_ids[i:]])
    #    aprime[sort_ids[i:]] *= (m - i)/float(m * Z)

    #    #if np.max(aprime) <= 1.0/m: 
    #    # We know if any element is bigger than 1.0/m, this one is
    #    if aprime[sort_ids[i]] <= 1.0/m:
    #        #print("CAPPED", i)
    #        return aprime


# jit seems to make this slower
#@jit(nopython=True) 
def outerprod(a,X):
    """ Compute X^T diag(a) X """
    left = np.multiply(X.T,a)
    return np.dot(left,X)

## v_step for get_weights
# tolerance default in eigsh is 0, which is machine precision.
# shrinking didn't seem to help?
def v_step(a,X,Sig,eta,tol=0):
    """ Compute top eigenvector of X^T diag(1/n-a) X"""
    # Bug fix: this was formerly 1 and not 1/n
    n = a.shape[0]
    #m = int(n*(1 - eta)) 
    M = outerprod(1./n-a,X)

    # this method is slower sometimes and faster sometimes vs other one, depending on d?
    d = M.shape[0]

    # Want top eigenvalue algebraically
    eigenvalue, v = eigh(M, eigvals=(d-1,d-1)) 
    #eigenvalue, v = largest_eigsh(M, 1, which='LA',tol=tol)
    #print(eigenvalue)
    #print(np.sum(v ** 2))
    
    ## Don't regularize if constraint is satisfied
    if eigenvalue > 0:
        return v[:,0]
    else:
        return np.zeros(shape=v[:,0].shape)
    
    
# @jit(nopython=True)
def a_step(a,X,resids_sq,v,lr,lam,eta,n):
    """ Step to minimize \sum_i a_i resids_sq[i]^2 + \lambda sigma_{max}(X^T diag(1/n - a) X)"""
    m = min(int(n*(1 - eta))+1,n) 
    
    Xv_squared = np.dot(X,v)**2
    penalties = resids_sq - lam * Xv_squared

    #print("OBJ: ",np.dot(a,resids_sq), np.dot(a,-lam * Xv_squared) + lam * (1/n) * np.sum(Xv_squared))

    #print("BEFORE: ",np.dot(a,penalties) + lam * (1/n) * np.sum(Xv_squared))

    # multiplicative update
    a *= np.exp(-lr * penalties)
    a /= np.sum(a)
    #print("MIDDLE:: ",np.dot(a,penalties) + lam * (1/n) * np.sum(Xv_squared))
    # project back to sliced simplex
    a = cap(a,m)
    #print("AFTER: ",np.dot(a,penalties) + lam * (1/n) * np.sum(Xv_squared))
    #assert(a is not None)
    return a

def get_weights(X,y,w,params):
    n,d = X.shape
    lr,lam,steps,eta = params

    # a = np.array([1./m]*m + [0.]*(n-m))
    a = np.ones(n)/float(n)
    #Sig = np.cov(X.T)
    Sig = np.matmul(X.T,X)
    resids_sq = (y - np.matmul(X,w))**2

    #print("begin MW")
    for j in range(steps):
        if j % 10 == 0:
            print(j)
        if lam > 0:
            v = v_step(a,X,Sig,eta)
        else:
            v = np.zeros(d)
        #print(np.linalg.norm(v,ord=1))
        #assert(a is not None)
        #print('a step')
        a = a_step(a,X,resids_sq,v,lr=lr,lam=lam,eta=eta,n=n)
        #print('end a step')
        # print a
        # print a, np.sum(a)
    return a

# Directly solve problem usingn multiplicative weights 
# Note input params = (lr,lam,steps,eta)
def MW_no_alt_min(X,y,params):
    n,d = X.shape
    lr,lam,steps,eta = params

    w = np.zeros(d)
    a = np.ones(n)/float(n)
    Sig = np.cov(X.T)

    for j in range(steps):
        w = ols(X,y,a)
        resids_sq = (y - np.matmul(X,w))**2
        v = v_step(a,X,Sig,eta)
        a = a_step(a,X,resids_sq,v,lr=lr,lam=lam,eta=eta,n=n)
        # if j < steps/5.:
        #     w = ols(X,y,a)
        # else:
        #     for l in range(50):
        #         w += .01/n*np.sum(np.matmul((y - np.matmul(X,w))*a,X),axis=0)
        #   print clean_loss(Xs,pre_Ys,w,ids)
        # return w
    return w

# Xs: nxd matrix
# Ys: d-dimensional array
# alpha is ridge parameter as in sklearn
def altmin_step(Xs,Ys,a,params,init=False,alpha=0):
    n,d = Xs.shape
    if len(Ys) != n:
        raise("Xs and Ys diff dims")
    if init:
        w = huber(Xs,Ys)
    elif alpha > 0:
        # scale weights back up by n so alpha parameter behaves like normal
        w = ridge(Xs,Ys,alpha=alpha,a=a*n,intercept=False,standardize=False)
    else:
        #print('ols')
        w = ols(Xs,Ys,a)
        #print('done ols')
    # print "OUR ITERATE", w
    a = get_weights(Xs,Ys,w,params)
    return w, a

def altmin_loop(Xs,Ys,params,am_steps,alpha,a=None):
    n,d = Xs.shape
    if a is None:
        a = np.array([1./n]*n)
    for _ in range(am_steps):
        w,a = altmin_step(Xs,Ys,a,params,alpha=alpha)
    return w,a
   

# a = np.array([1.,1.,0.,0.,]) + np.array([1.,0.,1.,0.])
# a /= 4.

# resids = [0.1,0.2,0.4,1.]
# Xs = np.random.random((4,5))
# v = np.random.random(5)
# print a_step(a,Xs,resids,v)

# M = np.random.random((d,d))
# M = M + M.T
# print np.linalg.eig(M)
# _, v = eigh(M,eigvals=(d-1,d-1))
# print v[:,0]


# a = np.random.random(4)
