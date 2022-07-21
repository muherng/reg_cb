import numpy as np
import cvxpy as cp
from scipy.linalg import expm
from scipy.linalg import eigh

# np.random.seed(0)

# ch. 4 of kale
# alpha: current guess for optimum
def MMW(As,b,C,R,delta,alpha,rho,steps):
	m = len(As)
	n,_ = np.shape(C)
	def oracle(X):
		coeffs = [(As[j]*X.T).sum() for j in range(m)]
		constant = (C*X.T).sum()
		constraints = []
		y = cp.Variable(n,nonneg=True)
		constraints.append(b*y <= alpha)
		constraints.append(sum([coeffs[j]*y[j] - constant])>=0)
		obj = cp.Minimize(0)
		prob = cp.Problem(obj,constraints)
		prob.solve(solver = cp.MOSEK)
		if prob.value == np.inf:
			return False, X
		else:
			return True, y.value

	X = R/float(n)*np.eye(n)
	Ms = []
	eps = delta * alpha/(2*rho*R)
	epsprime = np.log(1 - eps)
	for _ in range(steps):
		oracle_success, y = oracle(X)
		if oracle_success:
			M = sum([As[j]*y[j] - C + rho*np.eye(n) for j in range(m)])/(2*rho)
			Ms.append(M)
			W = expm(-epsprime * sum([Ms[t] for t in range(len(Ms))]))
			X = R * W/np.trace(W)
		else:
			return X

def topvec(M):
	n,_ = np.shape(M)
	val, v = eigh(M, eigvals=(n-1,n-1))
	return val, v

# ch. 4 of kale
# feasibility
def MW(As,b,alpha,R,rho,eps,steps):
	delta = eps/(3*R)
	m = len(As)
	n,_ = np.shape(As[0])
	def oracle(p):
		# print "b vector:", b
		# print "p:", p
		mat = sum([p[j] *(As[j] - b[j]/float(R)*np.eye(n)) for j in range(m)])
		# print mat
		val, v = topvec(mat)
		if val[0] <= -delta:
			return "neg def"
		else:
			return R * np.outer(v,v)
	# run scalar MW
	w = np.ones(m)
	mw_out = np.zeros((n,n))
	for t in range(steps):
		p = w/float(np.sum(w))
		x = oracle(p)
		if x == "neg def":
			return "infeasible"
		else:
			costs = [((As[j]*x.T).sum()-b[j])/rho for j in range(m)]
			for i in range(m):
				if costs[i] >= 0:
					w[i] *= (1-eps)**costs[i]
				else:
					w[i] *= (1+eps)**(-costs[i])
			mw_out = (mw_out*t + x)/(t+1.)
	return mw_out

def maxqp(A):
	n,_ = A.shape
	m = n
	As = [0]*(m+1)
	b = [-1]*(m+1)
	b[-1] = 1
	for j in range(m):
		As[j] = np.zeros((m,m))
		As[j][j,j] = -1
	left = 1./n
	right = 1.
	found = False
	counter = 0
	while (not found) or (counter <= 10):
		alpha = (left + right)/2.
		As[-1] = A/alpha
		print alpha
		new_mw_out = MW(As,b,alpha,float(n),float(n/alpha),0.01,200)
		if new_mw_out == "infeasible":
			right = alpha
		else:
			found = True
			left = alpha
			mw_out = new_mw_out
			print left, right
		counter += 1
	return mw_out

n = 100
M = np.random.random((n,n))*2-1
A = M + M.T
# for i in range(n):
# 	A[i,i] = np.abs(A[i,i])
A /= np.sum(abs(A))


mw_out = maxqp(A)
print "MW OUT:", (A*mw_out.T).sum()


constraints = []
X = cp.Variable((n,n),PSD=True)
for i in range(n):
	constraints.append(X[i,i] <= 1)
obj = cp.Maximize(cp.trace(A.T*X))
prob = cp.Problem(obj,constraints)
prob.solve(solver=cp.MOSEK)
print "MOSEK:", prob.value