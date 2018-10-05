# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:15:36 2018

@author: HOME1
"""

import numpy as np
from svmutil import svm_train, svm_predict, svm_problem, svm_parameter

datatrain = np.loadtxt('features.train')
datatest = np.loadtxt('features.test')

indtr = np.logical_or(datatrain[:, 0] == 1, datatrain[:, 0] == 5)
indte = np.logical_or(datatest[:, 0] == 1, datatest[:,0 ] == 5)
Xtrain = datatrain[indtr, 1:].tolist()
Xtest = datatest[indte, 1:].tolist()

ytrain = np.where(datatrain[indtr, 0] == 1, 1, -1).tolist()
ytest = np.where(datatest[indte, 0] == 1, 1, -1).tolist()

C = [10**(6 - 2*i) for i in range(5)]

Ein = []
Eout = []

prob = svm_problem(ytrain, Xtrain)
for c in C:
    param = svm_parameter('-t 2 -g 1 -c ' + str(c) + ' -q')
    m = svm_train(prob, param)
    
    predin, accin, probin = svm_predict(ytrain, Xtrain, m)
    predout, accout, probout = svm_predict(ytest, Xtest, m)
    
    Ein.append(1 - accin[0]/100)
    Eout.append(1 - accout[0]/100)
    
CminEin = C[np.argmin(Ein)]
CminEout = C[np.argmin(Eout)]

print('C with min Ein =', CminEin)
print('C with min Eout = ', CminEout)