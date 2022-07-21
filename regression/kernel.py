import cvxpy as cp
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io
import mosek
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import cProfile
from baselines import *

np.random.seed(0)

def clean_loss(Xs,pre_Ys,w,ids):
    n,d = Xs.shape
    if len(pre_Ys) != n:
        raise("Xs and Ys diff dims")
    out = 0
    for i in range(n):
        if i not in ids:
            out += (pre_Ys[i] - np.inner(Xs[i,:],w))**2
    return out/float(n)

drug_data = scipy.io.loadmat('qsar.mat')
X_train = drug_data['X_train']
X_test = drug_data['X_test']
X_train = np.array(X_train,dtype=float)
X_test = np.array(X_test,dtype=float)
# X_train /= max(np.linalg.norm(X_train,axis=1,ord=2))
# X_test /= max(np.linalg.norm(X_test,axis=1,ord=2))
pre_y_train = np.array(drug_data['y_train'])[:,0]
pre_y_test = np.array(drug_data['y_test'])[:,0]
y_train = pre_y_train.copy()
y_test = pre_y_test.copy()


eta = 0.1
n,d = X_train.shape
nbad = int(eta * n)

def corrupt(lam):
	X_corrupt = np.zeros((nbad,d))
	y_corrupt = np.zeros(nbad)
	# print y_train.shape, y_corrupt.shape
	for i in range(nbad):
		bad_direction = np.matmul(y_train.T,X_train)/float(nbad)
		X_corrupt[i,:] = (bad_direction + np.linalg.norm(bad_direction) * np.random.normal(0,1)/(10. * np.sqrt(d)))/lam
		y_corrupt[i] = -lam
	return [np.vstack((X_train,X_corrupt)),np.hstack((y_train,y_corrupt))]

lam = 3.

X_train2, y_train2 = corrupt(lam)

true_eta = eta/(1. + eta)

w = ols(X_train,y_train,None)
# alphas = np.linspace(4000,5000,100.)
alphas = [150.]
test_n,test_d = X_test.shape
X_test = np.hstack((X_test,np.ones((test_n,1))))
for a in alphas:
	w = sever(X_train2,y_train2,eta,r=5650,intercept=True)
	print a, clean_loss(X_test,pre_y_test,w,[])

# w = sever(X_train2,y_train2,true_eta,r=5650)
# print clean_loss(X_test,pre_y_test,w,[])

# print np.linalg.norm(pre_y_test)
