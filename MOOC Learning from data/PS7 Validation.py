# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:38:24 2018

@author: HOME1
"""

import numpy as np

# load train and test data
with open('in.dta.txt') as fin, open('out.dta.txt') as fout:
    datain = np.array([[float(num) for num in line.split()] for line in fin.readlines()])
    dataout = np.array([[float(num) for num in line.split()] for line in fout.readlines()])

# group train data into matrix
x1 = datain[:,0][:,np.newaxis]
x2 = datain[:,1][:,np.newaxis]
x0 = np.ones([len(datain),1])
# nonlinear transformation
z = np.concatenate([x0, x1, x2, x1**2, x2**2, x1*x2, np.abs(x1 - x2), np.abs(x1 + x2)], 1)
y = datain[:,2]

# group test data into matrix
x1t = dataout[:,0][:,np.newaxis]
x2t = dataout[:,1][:,np.newaxis]
x0t = np.ones([len(dataout),1])
# nonlinear transformation
zt = np.concatenate([x0t, x1t, x2t, x1t**2, x2t**2, x1t*x2t, np.abs(x1t - x2t), np.abs(x1t + x2t)], 1)
yt = dataout[:,2]


numtrain = 25
numval = 10
numtest = len(dataout)
kmax = 7
kmin = 3

Errval = np.zeros(kmax - kmin + 1)
Errout = np.zeros(kmax - kmin + 1)
for k in range(kmin,kmax+1):
    ztemp = z[:numtrain,:k+1]
    # linear regression on training set
    ztempdag = np.linalg.inv(np.dot(ztemp.T,ztemp)).dot(ztemp.T)
    w = ztempdag.dot(y[:numtrain])
    # predict on validation set
    predy = np.sign(np.dot(w,z[numtrain:,:k+1].T))
    Errval[k-3] = np.sum(predy != y[numtrain:])/numval
    
    predy = np.sign(np.dot(w,zt[:,:k+1].T))
    Errout[k-3] =  np.sum(predy != yt)/numtest
    
print('Errval = ', Errval)
print('Errout = ', Errout)