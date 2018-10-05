# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:27:31 2018

@author: HOME1
"""
import numpy as np

# Load data
datatrain = np.loadtxt('features.train')
datatest = np.loadtxt('features.test')



# Calculate Ein and Eout with feature transformation for lambda = 0.01 and 1
Lambda = [0.01, 1]
digitvdigit = [1, 5]
Ein = []
Eout = []

# Create (X, Y) for one versus one
indtr = np.logical_or(datatrain[:,0] == digitvdigit[0], datatrain[:,0] == digitvdigit[1])
indte = np.logical_or(datatest[:,0] == digitvdigit[0], datatest[:,0] == digitvdigit[1])

Xtrain = np.concatenate((np.ones((np.sum(indtr), 1)), datatrain[indtr, 1:]), axis = 1)
Ytrain = np.where(datatrain[indtr, 0] == digitvdigit[0], 1, -1)

Xtest = np.concatenate((np.ones((np.sum(indte), 1)), datatest[indte, 1:]), axis = 1)
Ytest = np.where(datatest[indte, 0] == digitvdigit[0], 1, -1)

# With Feature Transformation
Ztrain = np.concatenate((Xtrain, Xtrain[:, 1, np.newaxis] * Xtrain[:, 2, np.newaxis], Xtrain[:, 1, np.newaxis]**2, Xtrain[:, 2, np.newaxis]**2), axis = 1)
Ztest = np.concatenate((Xtest, Xtest[:, 1, np.newaxis] * Xtest[:, 2, np.newaxis], Xtest[:, 1, np.newaxis]**2, Xtest[:, 2, np.newaxis]**2), axis = 1)

for lmda in Lambda:
    # Regularised Linear Regression for Classification
    Zdag = np.linalg.inv(Ztrain.T.dot(Ztrain) + lmda * np.eye(Ztrain.shape[1])).dot(Ztrain.T)
    w = Zdag.dot(Ytrain)
    
    # Find Ein and Eout
    Ein.append(np.mean(np.sign(w.dot(Ztrain.T)) != Ytrain))
    Eout.append(np.mean(np.sign(w.dot(Ztest.T)) != Ytest))




# Calculate Ein and Eout for with and without feature transformation for one versus all
lmda = 1
digitvall = [i for i in range(10)]
Eintrans = []
Eouttrans = []
Einnotrans = []
Eoutnotrans = []

for digit in digitvall:
    # Create (X, Y) for one versus all
    Ytrain = np.where(datatrain[:, 0] == digit, 1, -1)
    Xtrain = np.concatenate((np.ones((Ytrain.size, 1)), datatrain[:,1:]), axis = 1)
    
    Ytest = np.where(datatest[:, 0] == digit, 1, -1)
    Xtest = np.concatenate((np.ones((Ytest.size, 1)), datatest[:,1:]), axis = 1)
    
    # No Feature Transfoamrion
    Ztrainnotrans = Xtrain
    Ztestnotrans = Xtest
    # With Feature Transformation
    Ztraintrans = np.concatenate((Xtrain, Xtrain[:, 1, np.newaxis] * Xtrain[:, 2, np.newaxis], Xtrain[:, 1, np.newaxis]**2, Xtrain[:, 2, np.newaxis]**2), axis = 1)
    Ztesttrans = np.concatenate((Xtest, Xtest[:, 1, np.newaxis] * Xtest[:, 2, np.newaxis], Xtest[:, 1, np.newaxis]**2, Xtest[:, 2, np.newaxis]**2), axis = 1)
    
    # Regularised Linear Regression for Classification with feature transformation
    Zdagtrans = np.linalg.inv(Ztraintrans.T.dot(Ztraintrans) + lmda * np.eye(Ztraintrans.shape[1])).dot(Ztraintrans.T)
    wtrans = Zdagtrans.dot(Ytrain)
    # Regularised Linear Regression for Classification with no feature transformation
    Zdagnotrans = np.linalg.inv(Ztrainnotrans.T.dot(Ztrainnotrans) + lmda * np.eye(Ztrainnotrans.shape[1])).dot(Ztrainnotrans.T)
    wnotrans = Zdagnotrans.dot(Ytrain)
    
    # Find Ein and Eout
    Eintrans.append(np.mean(np.sign(wtrans.dot(Ztraintrans.T)) != Ytrain))
    Eouttrans.append(np.mean(np.sign(wtrans.dot(Ztesttrans.T)) != Ytest))
    
    Einnotrans.append(np.mean(np.sign(wnotrans.dot(Ztrainnotrans.T)) != Ytrain))
    Eoutnotrans.append(np.mean(np.sign(wnotrans.dot(Ztestnotrans.T)) != Ytest))
