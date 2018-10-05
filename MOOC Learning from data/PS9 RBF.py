# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:43:27 2018

@author: HOME1
"""

import numpy as np
from svmutil import svm_train, svm_predict, svm_problem, svm_parameter


def f(X):
   return  np.sign(X[:, 1] - X[:, 0] + 0.25 * np.sin(np.pi * X[:, 0]))


def SVM(X, Y, gamma):
    X = X.tolist()
    Y = Y.tolist()
    prob = svm_problem(Y, X)
    param = svm_parameter('-g ' + str(gamma) + ' -t 2 -c 1e6') # hard margin
    m = svm_train(prob, param)
    
    # return model
    def g(X):
        N = X.shape[0]
        Y = np.zeros(N)
        return np.array(svm_predict(Y.tolist(), X.tolist(), m, options = '-q')[0])
    
    return g

    
def lloyds(X, K):
    N = X.shape[0]
    
    def findcluster(mu):
        S = np.zeros((K, N), dtype = bool) # Occupation matrix for sets S, S[m, n] = True if X[n] belongs to Mu[m]
        dis = np.zeros((K, N)) # Distance matrix
        for i in range(K):
            dis[i] = np.linalg.norm(X - mu[i], axis = 1)
        ind = np.argmin(dis, axis = 0)
        S[ind, np.arange(N)] = True
        
        return S
    
    def findcentre(S):
        mu = np.zeros((K, 2))
        for i in range(K):
            mu[i] = np.mean(X[S[i]], axis = 0)
        
        return mu
    
    # Initialise random cluster centres and assign X to clusters
    mu = np.random.uniform(-1, 1, (K, 2))
    while True:
        S = findcluster(mu)
        # discard and generate new centres when clusters are empty
        while 0 in np.sum(S, axis = 1): 
             mu = np.random.uniform(-1, 1, (K, 2))
             S = findcluster(mu)
        muold = mu
        mu = findcentre(S)
        if np.array_equal(muold, mu):
            break
    
    return mu

    
def RBF(X, Y, K, gamma):
    N = X.shape[0]
    mu = lloyds(X, K)
    Phi = np.exp([[-gamma * np.linalg.norm(X[i] - mu[j]) ** 2 for j in range(K)] for i in range(N)])
    Phi = np.concatenate((np.ones((N, 1)), Phi), axis = 1) # add bias
    Phidag = np.linalg.inv(np.dot(Phi.T, Phi)).dot(Phi.T)
    w = Phidag.dot(Y)
   
    # return model
    def g(X):
        N = X.shape[0]
        Phi = np.exp([[-gamma * np.linalg.norm(X[i] - mu[j]) ** 2 for j in range(K)] for i in range(N)])
        Phi = np.concatenate((np.ones((N, 1)), Phi), axis = 1) # add bias
        return np.sign(Phi.dot(w))
    
    return g


Ntrain = 100
Ntest = 1000
K = 9
gamma = 1.5

"""
EinSVM = []
EoutSVM = []
EinRBF = []
EoutRBF = []

for i in range(1000):
    Xtrain = np.random.uniform(-1, 1, (Ntrain, 2))
    Ytrain = f(Xtrain)
    
    Xtest = np.random.uniform(-1, 1, (Ntest, 2))
    Ytest = f(Xtest)
    
    gSVM = SVM(Xtrain, Ytrain, gamma)
    gRBF = RBF(Xtrain, Ytrain, K, gamma)
    
    EinSVM.append(np.mean(gSVM(Xtrain) != Ytrain))
    EoutSVM.append(np.mean(gSVM(Xtest) != Ytest))
    
    EinRBF.append(np.mean(gRBF(Xtrain) != Ytrain))
    EoutRBF.append(np.mean(gRBF(Xtest) != Ytest))


print('K =', K, ', gamma =', gamma)
print('mean Ein SVM =', np.mean(EinSVM))
print('mean Eout SVM =', np.mean(EoutSVM))

print('mean Ein RBF =', np.mean(EinRBF))
print('mean Eout RBF =', np.mean(EoutRBF))

print('% time SVM beats RBF =', np.mean(np.array(EoutSVM) < np.array(EoutRBF)))
"""

EinRBF = []
EoutRBF = []
numiter = 1000
for i in range(numiter):
    Xtrain = np.random.uniform(-1, 1, (Ntrain, 2))
    Ytrain = f(Xtrain)
    
    Xtest = np.random.uniform(-1, 1, (Ntest, 2))
    Ytest = f(Xtest)
    
    gRBF = RBF(Xtrain, Ytrain, K, gamma)
    
    EinRBF.append(np.mean(gRBF(Xtrain) != Ytrain))
    EoutRBF.append(np.mean(gRBF(Xtest) != Ytest))

EinRBF = np.array(EinRBF)
EoutRBF = np.array(EoutRBF)