# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 17:06:10 2018

@author: HOME1
"""
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

#import matplotlib.pyplot as plt

N = 100
repeat = 1000

# Calculate the probability that P(f(x) != g(x)) for N random points
def Pfneqg(fw,gw,N):
    x = np.random.uniform(-1,1,[2,N])
    x = np.concatenate((np.ones((1,N)),x),0)
    return np.mean(np.sign(np.dot(fw,x)) != np.sign(np.dot(gw,x)))

# Create target function f
def newf():
    fpts = np.random.uniform(-1,1,[2,2]) # ((x1,y1),(x2,y2))
    fw = (fpts[1,1]*(fpts[1,0] - fpts[0,0]) - fpts[1,0]*(fpts[1,1] - fpts[0,1]),
    fpts[1,1] - fpts[0,1],
    fpts[0,0] - fpts[1,0])
    return np.array(fw)

# Create sample data and disregard if all points on one side of f
def newxy(N,fw):
    dot = np.zeros(N)
    while np.sum(dot >= 0) == N or np.sum(dot >= 0) == 0:
        x = np.concatenate((np.ones((N,1)), np.random.uniform(-1,1,[N,2])),1)
        dot = np.dot(fw, x.T)
    y = np.sign(dot)
    return (x,y)

# PLA algorithm
def PLA(x,y):
    PLAw = np.zeros(3)
    while True:
        misclasspts = np.sign(np.dot(PLAw, x.T)) != y
        if np.sum(misclasspts) != 0:
            misclassptsindex = np.nonzero(misclasspts)[0]
            index = np.random.choice(misclassptsindex)
            PLAw = PLAw + y[index]*x[index]
        else:
            break
    return PLAw

# SVM algorithm
def SVM(x,y):
    N = y.size
    xr = x[:,1:]

    P = matrix([[y[m]*y[n]*np.dot(xr[m],xr[n]) for m in range(N)] for n in range(N)])
    q = matrix(-np.ones(N))
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(y).trans()
    b = matrix(0.0)

    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(sol['x'])
    SVindex = alpha > 1e-4

    wr = np.sum(alpha[SVindex] * y[SVindex] * xr[SVindex].T, axis = 1)
    k = np.mean(y[SVindex] - np.dot(wr, xr[SVindex].T))
    
    return np.concatenate((np.array([k]),wr))

#plt.axis([-1,1,-1,1])
#plt.plot(fpts[:,0],fpts[:,1])
#plt.plot(fpts[0,0],fpts[0,1], 'ro')
#plt.plot(fpts[1,0],fpts[1,1], 'bo')
#plt.plot(x[1,0], x[2,0], 'ro')
#plt.plot(x[1,1], x[2,1], 'bo')
count = 0
for i in range(repeat):
    fw = newf()
    x,y = newxy(N, fw)
    PLAw = PLA(x,y)
    SVMw = SVM(x,y)
    EoutPLA = Pfneqg(fw,PLAw,1000*N)
    EoutSVM = Pfneqg(fw,SVMw,1000*N)
    if EoutPLA > EoutSVM:
        count += 1
percent = count/repeat

print('% of gSVM better than gPLA in approximating f =',percent)
#print('Eout for PLA = ', EoutPLA)
#print('Eout for SVM = ', EoutSVM)

