import numpy as np
#import mosek
from sklearn.linear_model import HuberRegressor, LinearRegression, TheilSenRegressor, RANSACRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh

SEVER_STEPS = 5
TORRENT_STEPS = 100

def clean_loss(Xs,pre_Ys,w,ids):
    n,d = Xs.shape
    if len(pre_Ys) != n:
        raise("Xs and Ys diff dims")
    out = 0
    for i in range(n):
        if i not in ids:
            out += (pre_Ys[i] - np.inner(Xs[i,:],w))**2
    return out/float(n)

def ols(X,y,a,fast=True):
    n,d = X.shape    
    if fast:
        # w = np.array([0.]*d)
        # for l in range(grad_steps):
        #   w += 0.0001/n*np.sum(np.matmul((Ys - np.matmul(Xs,w))*a,Xs),axis=0)
        #   print clean_loss(Xs,pre_Ys,w,ids)
        # return w
        if not(a is None):
            a = n * np.array(a)
        reg = LinearRegression(fit_intercept=False).fit(X,y,a)
        w = reg.coef_
        return reg.coef_
    else:
        if a is None:
            a = np.ones(n)
        aX = np.matmul(np.diag(a),X)
        return np.matmul(np.linalg.pinv(np.matmul(aX.T,X)),np.matmul(aX.T,y))

def huber(X,y,a=None):
    huber = HuberRegressor(fit_intercept=False,max_iter=100,tol=1e-2).fit(X,y,a)
    return huber.coef_

def torrent(X,y,eta,eps):
    n,d = X.shape
    beta = 1.1*eta
    resids = np.array([np.inf]*n)
    counter = 0
    m = int((1. - beta)*n)
    if m == n:
        return ols(X,y,None)
    ids = range(n)
    while np.linalg.norm(resids[ids])**2/float(n) > eps and counter < TORRENT_STEPS:
        print(counter)
        w = ols(X[ids,:],y[ids],None)
        resids = (y - np.matmul(X,w))**2
        ids = np.argpartition(resids, m)[:m]
        counter += 1
        # print counter, np.linalg.norm(w), np.linalg.norm(resids[ids])**2
    print("torrent resids: ", np.linalg.norm(resids[ids])**2/float(n))
    return w

def theilsen(X,y):
    ts = TheilSenRegressor(fit_intercept=False).fit(X,y)
    #print(ts.coef_)
    # counteract special behavior in 1d where it returns scalar
    return np.atleast_1d(ts.coef_)

def ransac(X,y):
    rs = RANSACRegressor(base_estimator=HuberRegressor(fit_intercept=False)).fit(X,y)
    return rs.estimator_.coef_

# output has intercept
# WARNING: standardize and intercept may not work correctly
def ridge(X,y,alpha,a=None,ker=None,intercept=True,standardize=True):
    mean = np.mean(X,axis=0)
    if standardize:
        std = np.std(X,axis=0)
        std[std < 1e-12] = 1.
        newX = X - mean
        newX /= std
        newy = y - np.mean(y)
    else:
        newX = X
        newy = y
    kr = Ridge(alpha=alpha,fit_intercept=False).fit(newX,newy,a)
    w = kr.coef_
    if standardize:
        w /= std
    if intercept:
        b = np.mean(y)-np.inner(mean,w)
        w = np.append(w,b)
    return w
    
# output has intercept
def sever(X,y,eta,r=5650,intercept=True):
    n,d = X.shape
    def ids_to_a(ids):
        out = np.zeros(n)
        for i in ids:
            out[i] = 1
        return out
    # active_ids = np.array(range(n))
    a = np.ones(n)
    for t in range(SEVER_STEPS):
        # X = X[active_ids,:]
        n,d = X.shape
        # print d
        # m = int((1. - eta/20.)*n)
        # y = y[active_ids]
        w = ridge(X,y,r,a,intercept=intercept)
        if intercept:
            resids = y - np.matmul(X,w[:-1]) - w[-1]
        else:
            resids = y - np.matmul(X,w)
        grads = np.matmul(np.diag(resids), X)
        # center gradients
        grads -= np.mean(grads,axis=0)
        M = np.matmul(grads.T,grads)
        _, v = largest_eigsh(M, 1, which='LM')
        v = v[:,0]
        scores = np.matmul(grads,v)**2
        max_active_score = max(scores[a > 1e-9])
        a[a>1e-9] *= (1 - scores[a >1e-9]/max_active_score)
        # print min(a)
        # print "SEVER ITERATE", w
    return w




# eta = 0.2
# n = 100
# d = 10
# X = np.random.normal(size=(n,d))
# w = np.random.normal(0,1,size=10)
# w /= np.linalg.norm(w)
# pre_Y = np.matmul(X,w)
# bound = max(pre_Y)
# ids = []
# y = pre_Y.copy()
# for i in range(n):
#     if np.random.rand() < eta:
#         y[i] += np.random.random()*10*bound - 5*bound + np.random.normal(0,0.1)
#         ids.append(i)

# what = theilsen(X,y)
# out = 0
# for i in range(n):
#     if i not in ids:
#         out += (pre_Y[i] - np.inner(X[i,:],what))**2
# print out
