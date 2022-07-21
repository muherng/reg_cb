import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.linear_model import LinearRegression
from baselines import *
#grad_steps = 500 # old?

# from numba import jit

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

# @jit(nopython=True)
def cap(a,m):
    if np.max(a) <= 1./m:
        return a
    #sorted_a = np.sort(a)
    sort_ids = np.argsort(a)
    for i in range(m):
        aprime = np.copy(a)
        for j in range(i):
            aprime[sort_ids[j]] = 1./m
        Z = np.sum(aprime[sort_ids[i:]])
        aprime[sort_ids[i:]] *= (m - i)/float(m * Z)
        if np.max(aprime) <= 1./m:
            return aprime


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
    """ Compute top eigenvector of X diag(a) X^T - (1 - 1.1 \eta) Sigma"""
    M = outerprod(a,X) - (1. - eta)*Sig

    # this method is slower sometimes and faster sometimes vs other one, depending on d?
    d = M.shape[0]
    ## CHANGED: pick bottom instead of top eigenvector?
    _, v = eigh(M, eigvals=(0,0)) 
    #_, v = largest_eigsh(M, 1, which='LM',tol=tol)
    return v[:,0]

# @jit(nopython=True)
def a_step(a,X,resids_sq,v,lr,lam,eta,n):
    """ Step to minimize \sum_i a_i resids_sq[i]^2, with langrage multiplier for X diag(a) X^T - (1 - 1.1 \eta) \Sigma < 0"""
    m = int(n*(1 - 1.1*eta))
    Xv_squared = np.dot(X,v)**2
    # was this:
    #penalties = a * resids_sq - lam * (a * Xv_squared - (1 - 2*eta) * Xv_squared/float(n))
    # fixed version (gradient instead of function, also sign fix?). penalizes points with big squared loss or corr with eigendirection
    penalties = resids_sq + lam * Xv_squared
    # multiplicative update
    a *= np.exp(-lr * penalties)
    a /= np.sum(a)
    # project back to sliced simplex
    a = cap(a,m)
    return a


def get_weights(X,y,w,params):
    n,d = X.shape
    lr,lam,steps,eta = params

    # a = np.array([1./m]*m + [0.]*(n-m))
    a = np.ones(n)/float(n)
    Sig = np.cov(X.T)
    resids_sq = (y - np.matmul(X,w))**2

    #print("begin MW")
    for j in range(steps):
        # if j % 10 == 0:
        #     print(j)
        v = v_step(a,X,Sig,eta)
        a = a_step(a,X,resids_sq,v,lr=lr,lam=lam,eta=eta,n=n)
        # print a
        # print a, np.sum(a)
    return a

# can probably analyze this too? if grad for a keeps being big,
# then we keep shrinking objective value?
def MW_no_alt_min(X,y,params):
    n,d = X.shape
    lr,lam,steps,eta = params

    w = np.zeros(d)
    a = np.ones(n)/float(n)
    Sig = np.cov(X.T)

    for j in range(steps):
        resids_sq = (y - np.matmul(X,w))**2
        v = v_step(a,X,Sig,eta)
        a = a_step(a,X,resids_sq,v,lr=lr,lam=lam,eta=eta,n=n)
        w = ols(X,y,a)
        # if j < steps/5.:
        #     w = ols(X,y,a)
        # else:
        #     for l in range(50):
        #         w += .01/n*np.sum(np.matmul((y - np.matmul(X,w))*a,X),axis=0)
        #   print clean_loss(Xs,pre_Ys,w,ids)
        # return w
    return w


# def get_weights(X,y,w):


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
