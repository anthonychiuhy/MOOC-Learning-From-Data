# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:45:35 2018

@author: HOME1
"""

import numpy as np
from svmutil import svm_train, svm_predict, svm_problem, svm_parameter
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

X = [[1, 0],
     [0, 1],
     [0, -1],
     [-1, 0],
     [0, 2],
     [0, -2],
     [-2, 0]]

Y = [-1, -1, -1, 1, 1, 1, 1]


# LIBSVM
prob = svm_problem(Y, X)
param = svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 1e6')
m = svm_train(prob, param)

numsvlibsvm = m.get_nr_sv()

print('no. of SV (libsvm) =', numsvlibsvm)


# CVXOPT
N = len(X)

def Kernel(x1,x2):
    return (1 + np.dot(x1,x2))**2

K = np.array([[Kernel(X[i], X[j]) for i in range(N)] for j in range(N)], dtype = float)
YY = np.outer(Y,Y).astype(float)

P = matrix(K * YY)
q = matrix(-np.ones(N))
G = matrix(-np.eye(N))
h = matrix(np.zeros(N))
A = matrix(np.array(Y, dtype = float)).trans()
b = matrix(0.)

sol = solvers.qp(P, q, G, h, A, b)

alpha = np.array(sol['x']).ravel()

numsvcvxopt = np.sum(alpha > 1e-6) # SV is when alpha != 0

print('np. of SV (cvxopt) =', numsvcvxopt)