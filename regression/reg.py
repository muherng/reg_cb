import cvxpy as cp
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io
# mosek only needed if we don't use MW
#import mosek
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import cProfile
from baselines import *
from scipy.linalg import sqrtm

np.random.seed(0)

# Alternating minimization steps
AM_steps = 5

# use capped multiplicative weights if True, or use mosek if False
USE_MW = True
new_MW = True

# error on uncorrupted points
# ids: indices of corrupted points
def clean_loss(Xs,pre_Ys,w,ids):
    n,d = Xs.shape
    if len(w) == d:
        out = 0
        for i in range(n):
            if i not in ids:
                out += (pre_Ys[i] - np.inner(Xs[i,:],w))**2
    elif len(w) == d + 1:
        out = 0
        for i in range(n):
            if i not in ids:
                out += (pre_Ys[i] - np.inner(Xs[i,:],w[:-1]) - w[-1])**2
    else:
        raise Exception("incompatible dims")
    return out/float(n)

if USE_MW:
    if new_MW:
        from mw2 import MW_no_alt_min, get_weights, altmin_step
    else:
        from mw import MW_no_alt_min, get_weights
else:
    def get_weights(Xs,Ys,w,params):
        n,d = Xs.shape
        lr,lam,steps,eta = params
        if len(Ys) != n:
            raise("Xs and Ys diff dims")
        # unnormalized covariance of samples
        Sig = np.matmul(Xs.T,Xs)
        resids = (Ys - np.matmul(Xs,w))**2
        a = cp.Variable(n)
        constraints = []
        constraints.append(0<=a)
        constraints.append(1>=a)
        constraints.append(cp.sum(a) >= (1 - eta)*n)
        constraints.append(sum([np.outer(Xs[i,:],Xs[i,:])*(1-a[i]) for i in range(n)]) << eta*Sig)
        obj = cp.Minimize(a*resids/float(n))
        prob = cp.Problem(obj,constraints)
        # print("solving SDP")
        result = prob.solve(solver = cp.MOSEK, 
                        # mosek_params = {mosek.dparam.optimizer_max_time:  100.0
                                        # mosek.iparam.intpnt_solve_form:   mosek.solveform.dual
                                        # },
                        verbose = False)
        # result = prob.solve(solver=cp.SCS,verbose=True)
        # for i in range(n):
        #   if a.value[i] < 0.99:
        #       print i, a.value[i]
        return a.value.clip(0)


## Maybe delete isotropic part
# altmin wrapped around MW
def test_offline(Xs_train,Xs_test,Ys_train,pre_Ys_test,ids_test,params):
    n,d = Xs_train.shape
    w = [0.]*d
    a = [1.]*n
    iso_Xs_train, Sig_sqrt = isotropic(Xs_train)
    # print "not putting in isotropic"
    for i in range(AM_steps):
        if i == 0:
            init = False
        else:
            init = False
        w,a = altmin_step(iso_Xs_train,Ys_train,a,params,init=init)
        print(clean_loss(Xs_test,pre_Ys_test,np.matmul(Sig_sqrt,w),ids_test))
    final_w = np.matmul(Sig_sqrt,w)
    return final_w, clean_loss(Xs_test,pre_Ys_test,final_w,ids_test)


# just alternate between MW and a full ols solve
def test_offline2(Xs_train,Xs_test,Ys_train,pre_Ys_test,ids_test,params):
    n,d = Xs_train.shape
    w = [0]*d
    w = MW_no_alt_min(Xs_train,Ys_train,params)
    return w, clean_loss(Xs_test,pre_Ys_test,w,ids_test)


# Xs = np.random.rand(n,d)
# Xs /= np.linalg.norm(Xs,axis=1,ord=2)[:,None]
# w = np.random.rand(d)
# print "wnorm", np.linalg.norm(w)
# pre_Ys = np.matmul(Xs,w)
# # pre_Ys += np.random.multivariate_normal([0]*n,np.eye(n))
# # Ys = pre_Ys.copy()
# Ys = pre_Ys + np.random.multivariate_normal([0]*n,np.eye(n))*0.1



# subsampled = np.zeros((d,d))
# uncorrupted_count = 0
# # ids of the corrupted points
# ids = []
# for i in range(n):
#     if np.random.rand() < eta:
#         Ys[i] = np.random.normal(0,1)
#         ids.append(i)
#     else:
#         uncorrupted_count += 1
#         subsampled += np.outer(Xs[i,:],Xs[i,:])
# Sig = np.matmul(Xs.T,Xs)

# print len(ids)

# true_a = np.ones(n)
# for i in ids:
#     true_a[i] = 0.

# what = ols(Xs,pre_Ys,true_a)
# print "ground truth MSE:", clean_loss(Xs,pre_Ys,what,ids)

# test_offline(Xs,Ys,pre_Ys,ids)

# returns X in isotropic position and change of basis (Sig^{-1/2})
def isotropic(Xs,fake=False):
    if fake:
        return Xs, np.eye(Xs.shape[1])
    Sig = np.matmul(Xs.T,Xs)
    Sig_sqrt = np.linalg.inv(sqrtm(Sig))
    new_Xs = np.matmul(Xs,Sig_sqrt)
    return new_Xs, Sig_sqrt

def load_dataset(n,d,synth=True):
    if synth:
        Xs = np.random.rand(n,d)
        Xs /= max(np.linalg.norm(Xs,axis=1,ord=2))
        w = np.random.rand(d)/4.
        pre_Ys = np.matmul(Xs,w)
        Ys = pre_Ys.copy()
        for i in range(n):
            Ys[i] += np.random.normal(0,0.1)
    else:
        drug_data = scipy.io.loadmat('qsar.mat')
        Xs = drug_data['X_train']
        Xs = np.array(Xs,dtype=float)
        Xs /= max(np.linalg.norm(Xs,axis=1,ord=2))

        pre_Ys = np.array(drug_data['y_train'])[:,0]
        Ys = pre_Ys.copy()
    return Xs, pre_Ys, Ys

def defeat_torrent(n, eta):
    w = np.array([1])
    X = np.zeros((n,1))
    R = 100
    sig = 10
    flags = np.zeros(n)
    for i in range(int(0.1*n)):
        X[i,:] = [R + np.random.normal(0,3)]
    for i in range(int(0.1*n),int(.2*n)):
        X[i,:] = [-R + np.random.normal(0,3)]
    for i in range(int(0.2*n),n):
        X[i,:] = [R/5 + np.random.normal(0,3)]
    # X, _ = isotropic(X)
    # X /= R
    pre_y = X[:,0]*w[0]
    y = X[:,0]*w[0] + np.random.multivariate_normal([0]*n,sig**2 * np.eye(n))
    for i in range(n):
        if np.random.rand() < eta:
            flags[i] = 1
            y[i] = 0.
    return X/R, pre_y/R, y/R, flags
    # return X, pre_y, y, flags

def random_unit_vec(d):
    out = np.random.normal(0,1,size=d)
    return out/np.linalg.norm(out)

def defeat_torrent2(n,d,eta):
    # w = np.random.normal(0,1,size=(d))
    w = np.array([1] + [0.]*(d-1))
    X = np.zeros((n,d))
    R = 100
    sig = 10
    flags = np.zeros(n)
    shift1 = np.array([R] + [0.]*(d-1))
    shift2 = np.array([0.,R] + [0.]*(d-2))
    # for i in range(int(0.1*n)):
    #     X[i,:] = shift1 
    #     # + np.random.normal(0,1,size=(d))
    # for i in range(int(0.1*n),int(.2*n)):
    #     X[i,:] = -shift1
    #      # + np.random.normal(0,1,size=(d))
    for i in range(int(0.1*n)):
        # X[i,:] = R * random_unit_vec(d)
        X[i,:] = shift1
    for i in range(int(0.1*n),int(0.2*n)):
        # X[i,:] = R * random_unit_vec(d)
        X[i,:] = -shift1*2
    for i in range(int(0.2*n),n):
        X[i,:] = np.random.normal(0,1,size=(d))
    # X, _ = isotropic(X)
    # X /= R
    pre_y = np.matmul(X,w)
    y = pre_y + np.random.normal(0,sig,size=n)
    for i in range(n):
        if np.random.rand() < eta:
            flags[i] = 1
            y[i] = 0.
    return X/R, pre_y/R, y/R, flags


def basic_contaminate(Ys,eta):
    n = len(Ys)
    uncorrupted_count = 0
    # ids of the corrupted points
    flags = np.zeros(n)
    for i in range(n):
        if np.random.rand() < eta:
            # Ys[i] = np.random.normal(0,1)
            Ys[i] = 0
            flags[i] = 1
    return Ys, flags
    

def gen_params_grid(eta,profile=False,tune=False):
    #lrs = np.linspace(0.5,2.,10) * (1.0/250) # trying to shrink
    lrs = [0.5]
    # lams = np.linspace(.5,1.7,6)
    lams = [0.2,0.4,0.8,1.6,3.2,6,12,24,48]
    if not tune:
        lrs = [lrs[0]]
        lams = [lams[0]]
    # MW_stepss = [120,160,200]
    MW_stepss = [500] # was 500
    out = [0]*(len(lams)*len(lrs)*len(MW_stepss))
    i = 0
    for lam in lams:
        for lr in lrs:
            for MW_steps in MW_stepss:
                params = (lr,lam,MW_steps,eta)
                out[i] = params
                i += 1
    if USE_MW and not profile:
        return out
    else:
        return out #[out[12]]


# pick between no altmin or altmin
subroutine = test_offline

def run_baselines(Xs_train,Xs_test,Ys_train,pre_Ys_test,ids_test,eta):
    def eval(w):
        if w is None:
            return None
        else:
            return clean_loss(Xs_test,pre_Ys_test,w,ids_test)
    # run OLS
    print("Running OLS")
    w_ols = ols(Xs_train,Ys_train,None)
    # run Huber
    print("Running Huber")
    w_hub = huber(Xs_train,Ys_train)
    # run torrent
    print("Running torrent")
    w_tor = torrent(Xs_train,Ys_train,eta,0.00000000008)
    # run theil sen
    #print("Running theil sen")
    if Xs_train.shape[1] < 10:
        w_ts = theilsen(Xs_train,Ys_train)
    else:
        w_ts = None
    # run ransac
    try:
        w_rs = ransac(Xs_train,Ys_train)
    except:
        w_rs = None
    # run sever
    print("Running SEVER")
    rs = [0.]
    best_sever_perf = np.inf
    best_r = None
    w_sev = None
    for r in rs:
        w = sever(Xs_train,Ys_train,eta,r=r,intercept=False)
        perf = eval(w)
        if perf < best_sever_perf:
            best_sever_perf = perf
            best_r = r
            w_sev = w
    # return [w_sev]
    return [np.zeros(len(w_ols)),w_ols,w_hub,w_tor,w_ts,w_rs,w_sev]

def synth_test(eta,profile=False):
    #Xs, pre_Ys, Ys = load_dataset(2000,10)
    #Xs, pre_Ys, Ys, flags = defeat_torrent(300,eta)
    Xs, pre_Ys, Ys, flags = defeat_torrent2(n=1000,d=500,eta=eta) # was 10000, 500
    # Xs, pre_Ys, Ys, flags = defeat_torrent2(300,5,eta)
    # Ys, flags = basic_contaminate(Ys,eta)
    n,d = Xs.shape

    best_params = None
    best_perf = np.inf

    params_grid = gen_params_grid(eta,profile=profile)

    Xs_train, Xs_test, pre_Ys_train, pre_Ys_test, Ys_train, Ys_test, flags_train, flags_test = train_test_split(Xs,pre_Ys,Ys,flags, test_size = .2)
    Xs_train, Xs_val, pre_Ys_train, pre_Ys_val, Ys_train, Ys_val, flags_train, flags_val = train_test_split(Xs_train,pre_Ys_train,Ys_train,flags_train, test_size = .2)

    def eval(w):
        if w is None:
            return None
        else:
            return clean_loss(Xs_val,pre_Ys_val,w,ids_val)

    # Xs_train2, Xs_test2, pre_Ys_train2, pre_Ys_test2, Ys_train2, Ys_test2, flags_train2, flags_test2 = train_test_split(Xs,pre_Ys,Ys,flags, test_size = .25)    
    ids_test = [i for i in range(len(flags_test)) if flags_test[i] == 1]
    ids_val = [i for i in range(len(flags_val)) if flags_val[i] == 1]

    #print("BASELINES DISABLED")
    # baseline_ws = []
    # print("BASELINES: ", run_baselines(Xs_train2,Xs_test2,Ys_train2,pre_Ys_test2,ids_test2))

    #baseline_ws = run_baselines(Xs_train,Xs_val,Ys_train,pre_Ys_val,ids_val,eta)
    #print([eval(w) for w in baseline_ws])
    for params in params_grid:
        print("Running SCRAM", params)
        # Xs_train, Xs_test, pre_Ys_train, pre_Ys_test, Ys_train, Ys_test, flags_train, flags_test = train_test_split(Xs,pre_Ys,Ys,flags, test_size = .25)
        w, perf = subroutine(Xs_train,Xs_test,Ys_train,pre_Ys_test,ids_test,params)
        if perf < best_perf:
            best_params = params
            best_perf = perf
            best_w = w
            # print("NEW", best_perf, best_params,subroutine(Xs_train2,Xs_test2,Ys_train2,pre_Ys_test2,ids_test2,best_params))
            print("NEW", best_perf, best_params, eval(w))
    ws = baseline_ws + [best_w]
    return [eval(w) for w in ws]


if USE_MW:
    if new_MW:
        print("Using new MW's")
    else:
        print("Using old MW's")
else:
    print("Using MOSEK")

etas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45]
NUM_TRIALS = 1
NUM_ETAS = len(etas)
out = np.zeros((NUM_TRIALS,NUM_ETAS,8))
for i in range(NUM_TRIALS):
    for j in range(NUM_ETAS):
        print("Synthetic test: ", etas[j])
        out[i,j,:] = synth_test(etas[j],profile=False)

with open('synth.npy','wb') as f:
    np.save(f,out)

with open('synth.npy', 'rb') as f:
    result = np.load(f)


print(result)

