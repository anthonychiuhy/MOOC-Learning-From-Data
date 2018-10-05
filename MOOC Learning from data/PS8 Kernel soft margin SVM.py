# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 22:01:51 2018

@author: HOME1
"""
import numpy as np
import svmutil
#from cvxopt import matrix, solvers
#solvers.options['show_progress'] = True

# SVM trainer with kernel in form of K = (1 + Xm.Xn)^Q

datatrain = np.loadtxt('features.train')
datatest = np.loadtxt('features.test')

# Q2 - Q4
Q = 2
C = 0.01

model1 = []
Ein1 = []
Eout1 = []
numSV1 = []

for digit in range(10):
    
    Xtrain = datatrain[:,1:].tolist()
    ytrain = np.where(datatrain[:,0] == digit, 1, -1).tolist()
    
    Xtest = datatest[:,1:].tolist()
    ytest = np.where(datatest[:,0] == digit, 1, -1).tolist()
    
    prob = svmutil.svm_problem(ytrain, Xtrain)
    param = svmutil.svm_parameter('-t 1 -d ' + str(Q) + ' -g 1 -r 1 -c ' + str(C))
    
    m = svmutil.svm_train(prob, param)
    model1.append(m)
    
    numSV1.append(m.get_nr_sv())
    
    ypred, yacc, yprob = svmutil.svm_predict(ytrain, Xtrain, m)
    Ein1.append((100 - yacc[0])/100)
    
    ypred, yacc, yprob = svmutil.svm_predict(ytest, Xtest, m)
    Eout1.append((100 - yacc[0])/100)


# Q5
Q = 5
C = [10**(-i) for i in range(5)]

model2 = []
Ein2 = []
Eout2 = []
numSV2 = []

for i in range(5):
    indtrain = np.logical_or(datatrain[:,0] == 1, datatrain[:,0] == 5)
    Xtrain = datatrain[indtrain, 1:].tolist()
    ytrain = np.where(datatrain[indtrain, 0] == 1, 1, -1).tolist()
    
    indtest = np.logical_or(datatest[:,0] == 1, datatest[:,0] == 5)
    Xtest = datatest[indtest, 1:].tolist()
    ytest = np.where(datatest[indtest, 0] == 1, 1, -1).tolist()
    
    prob = svmutil.svm_problem(ytrain, Xtrain)
    param = svmutil.svm_parameter('-t 1 -d ' + str(Q) + ' -g 1 -r 1 -c ' + str(C[i]))
    
    m = svmutil.svm_train(prob, param)
    model2.append(m)
    
    numSV2.append(m.get_nr_sv())
    
    ypred, yacc, yprob = svmutil.svm_predict(ytrain, Xtrain, m)
    Ein2.append((100 - yacc[0])/100)
    
    ypred, yacc, yprob = svmutil.svm_predict(ytest, Xtest, m)
    Eout2.append((100 - yacc[0])/100)