# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:10:29 2018

@author: HOME1
"""
import numpy as np

# load train and test data
with open('in.dta.txt') as fin, open('out.dta.txt') as fout:
    datain = np.array([[float(num) for num in line.split()] for line in fin.readlines()])
    dataout = np.array([[float(num) for num in line.split()] for line in fout.readlines()])

# group train data into matrix
X1 = datain[:,0][:,np.newaxis]
X2 = datain[:,1][:,np.newaxis]
X0 = np.ones([len(datain),1])
# nonlinear transformation
Z = np.concatenate([X0, X1, X2, X1**2, X2**2, X1*X2, np.abs(X1 - X2), np.abs(X1 + X2)], 1)
Y = datain[:,2]

k = -1
lamb = 10**k

print('k = ', k)

# linear regression with regularisation
Zdag = np.linalg.inv(np.dot(Z.T,Z) + lamb*np.eye(np.shape(Z)[1])).dot(Z.T)
w = Zdag.dot(Y)
# predict
predY = np.sign(np.dot(w,Z.T))

# calculate Ein
Ein = np.sum(predY != Y)/len(Y)
print('Ein = ', Ein)

# group test data into marix
X1out = dataout[:,0][:,np.newaxis]
X2out = dataout[:,1][:,np.newaxis]
X0out = np.ones([len(dataout),1])
# nonlinear transformation
Zout = np.concatenate([X0out, X1out, X2out, X1out**2, X2out**2, X1out*X2out, np.abs(X1out - X2out), np.abs(X1out + X2out)], 1)
Yout = dataout[:,2]

# predict
predYout = np.sign(np.dot(w,Zout.T))

# calculate Eout
Eout = np.sum(predYout != Yout)/len(Yout)
print('Eout = ', Eout)