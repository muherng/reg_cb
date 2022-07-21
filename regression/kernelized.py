# compute gaussian kernel
import numpy as np
import mw2
import baselines
import matplotlib.pyplot as plt
from PIL import Image

def compute_K(norms_left,norms_right,dot_prods,gamma=500):
    #norms = torch.diag(dot_products) # only for square case!!
    pre_exp = (-norms_left.reshape(-1,1) + 2 * dot_prods - norms_right.reshape(1,-1))/gamma
    #expnorms_left = torch.exp(-norms_left/gamma)
    #expnorms_right = torch.exp(-norms_right/gamma)
    #print(expnorms_left.shape)
    #print(expnorms_right.shape)
    #print(dot_prods.shape)
    #K = expnorms_left * torch.exp(2 * dot_prods/gamma) * expnorms_right
    return np.exp(pre_exp)

def compute_gaussian_kernel(dot_prods,gamma=500):
    norms = np.diag(dot_products)
    K = compute_K(norms,norms,dot_products,gamma=gamma)
    return K


def featurize(K):
    """Return A so that A A^T = K, i.e. rows are examples represented in finite dimension"""
    u,s,vh = np.linalg.svd(K,hermitian=True)
    return u @ np.diag(np.sqrt(s)) 
    # seems to complain a lot about 'not pd'
    #return np.linalg.cholesky(K)

def unfeaturize(A,w):
    """Given A so that AA^T = K, and vector A w, represent it in A-independent way by solving Aw = Ku,
       i.e. A w = AA^T u, i.e. w = A^Tu"""
    return np.linalg.solve(A.T, w)

def gen_example1(n,test_n,d = 2):
    X = 3 * (np.random.rand(n,d) - 0.5)
    test_X = 3 * (np.random.rand(test_n,d) - 0.5)
    #Y = np.linalg.norm(X,axis=1) ** 2 + 0.1 * np.random.rand(n)
    f = lambda X : np.linalg.norm(X,axis=1) * np.sin(5 * np.linalg.norm(X,axis=1))
    Y = f(X) + 0.1 * (np.random.rand(n) - 0.5) #np.linalg.norm(X,axis=1) ** 2 + 0.1 * np.random.rand(n)
    test_Y = f(test_X) #np.linalg.norm(test_X,axis=1) ** 2
    return (X,test_X,Y,test_Y)

def gen_example2(n,test_n):
    name = 'earthmoon.jpeg'
    img = Image.open(name).convert('L')
    a = np.asarray(img)
    print(a.shape)
    X = np.random.rand(n,2)
    f = lambda x : a[(a.shape[0] * x[:,0]).astype(int),(a.shape[1] * x[:,1]).astype(int)]
    Y = f(X)

    dimt = int(np.sqrt(test_n))
    test_X = np.zeros((dimt * dimt,2))
    c,d = np.meshgrid(np.linspace(0,1,dimt,endpoint=False),np.linspace(0,1,dimt,endpoint=False))
    test_X[:,0],test_X[:,1] = c.flatten(),d.flatten()
    #test_X = np.random.rand(test_n,2)# + np.random.rand(n) 
    test_Y = f(test_X)#a[test_X[:,0],test_X[:,1]]
    return X,test_X,Y,test_Y

def run_example():
    #n = 100
    #n = 2000
    n = 1500
    test_n = 32000
    #gamma = 0.001
    #gamma = 0.001
    gamma = 0.002
    d = 2
    #X,test_X,Y,test_Y = gen_example1(n,test_n,d)
    X,test_X,Y,test_Y = gen_example2(n,test_n)

    if True:
        Y[0:int(n/10)] = 100 #corruption
        #Y[0:int(n/10)] = 150 #corruption


    dot_products = X @ X.T
    norms = np.diag(dot_products)
    K = compute_K(norms,norms,dot_products,gamma=gamma)
    A = featurize(K)

    # example: ridge regression done using features
    #lbda = 0.001
    lbda = 0.01
    if True:
        # vanilla ridge
        #lbda = 0.1
        a = [1./n] * n
        wv = np.linalg.inv(A.T @ A + lbda * np.eye(n)) @ A.T @ Y
    if True:
        # actual altmin
        #eta = 0.15
        eta = 0.2
        #eta = 1./n
        # no spectral regularization
        #params = 0.0008,0,100,eta
        #params = 0.0008,0,200,eta
        #params = 0.0001,0,800,eta
        # spectral regularization (check is it working???)
        #params = 0.0001,0.01,800,eta
        #params = 0.0002,50000,40,eta
        params = 0.0001,50000,40,eta

        am_steps=30
        alpha = lbda

        params_unspectral = (params[0],0,params[2],params[3])
        #a_init = None
        a_init = mw2.get_weights(A,Y,np.zeros(A.shape[0]),params_unspectral)
        #params_unspectral[1] = 0
        wu,au = mw2.altmin_loop(A,Y,params_unspectral,am_steps=am_steps,alpha=alpha,a=a_init)

        # spectrally regularized
        # disabled bd init right now
        ws,a = mw2.altmin_loop(A,Y,params,am_steps=am_steps,alpha=alpha,a=a_init)
        print("Support: ", a)

        
    fig, axs = plt.subplots(2,2)
    axs = axs.flat
    plotteroo = lambda ax,y : ax.scatter(test_X[:,1],1 - test_X[:,0],s=20,c=y,cmap='gray',vmin=0,vmax=255,marker=",")

    def evaluate(w,ax,gt=False):
        u = unfeaturize(A,w)
        #print("Predictor: ", u)

        test_dot_products = X @ test_X.T
        test_norms = np.sum(test_X ** 2,axis=1)
        K_test = compute_K(norms, test_norms, test_dot_products,gamma=gamma)
        prediction_Y = u @ K_test
        #print(test_Y)
        #print(prediction_Y)

        if d == 2:
            if False:
                # 3d scatter plot
                ax = fig.add_subplot(projection='3d')
                ax.scatter(test_X[:,0],test_X[:,1],c=prediction_Y)
                ax.scatter(test_X[:,0],test_X[:,1],c=test_Y)
            else:
                plotteroo(ax,prediction_Y)
                #plt.show()
                #if gt:
                    #plotteroo(test_Y)
                #plt.scatter(test_X[:,0],test_X[:,1],c=prediction_Y)
                #plt.scatter(test_X[:,1],1 - test_X[:,0],c=test_Y,cmap='gray',vmin=0,vmax=255)
        #plt.show()
    evaluate(wv,axs[0])
    if True:
        evaluate(wu,axs[1])
        evaluate(ws,axs[2])
        plotteroo(axs[3],test_Y)
    plt.show()


if __name__ == "__main__":
    run_example()
