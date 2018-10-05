# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:54:39 2018

@author: HOME1
"""

import numpy as np
from svmutil import svm_problem, svm_parameter, svm_train, svm_predict

datatrain = np.loadtxt('features.train')
datatest = np.loadtxt('features.test')

Q = 2
C = [10**(i - 4) for i in range(5)]
numfoldcv = 10


numCpick = [0,0,0,0,0]

indtrain = np.logical_or(datatrain[:,0] == 1, datatrain[:,0] == 5)
Xtrain = datatrain[indtrain, 1:]
ytrain = np.where(datatrain[indtrain, 0] == 1, 1, -1)

sizecv = ytrain.size//numfoldcv


runs = 100
for i in range(runs):
    
    Ecv = np.zeros(len(C))
    
    indcv = np.random.permutation(ytrain.size)[:sizecv * numfoldcv].reshape((numfoldcv, sizecv))
    
    for j in range(len(C)):
        
        Ev = np.zeros(numfoldcv)
                    
        for k in range(numfoldcv):
            
            indtraincv = np.ones(numfoldcv, dtype = bool)
            indtraincv[k] = False
            indtraincv = indcv[indtraincv].ravel()
        
            Xtraincv = Xtrain[indtraincv].tolist()
            ytraincv = ytrain[indtraincv].tolist()
            
            Xcv = Xtrain[indcv[k]].tolist()
            ycv = ytrain[indcv[k]].tolist()
            
            prob = svm_problem(ytraincv, Xtraincv)
            param = svm_parameter('-t 1 -d ' + str(Q) + ' -g 1 -r 1 -c ' + str(C[j]))
            
            m = svm_train(prob, param)
            ypred, yacc, yprob = svm_predict(ycv, Xcv, m, options = '-q')
            
            Ev[k] = 1 - yacc[0]/100
        
        Ecv[j] = Ev.mean()
    
    numCpick[Ecv.argmin()] += 1




Cwin = C[np.argmax(numCpick)]

Ecvwin = np.zeros(runs)

for i in range(runs):
    
    indcv = np.random.permutation(ytrain.size)[:sizecv * numfoldcv].reshape((numfoldcv, sizecv))
        
    Ev = np.zeros(numfoldcv)
                
    for k in range(numfoldcv):
        
        indtraincv = np.ones(numfoldcv, dtype = bool)
        indtraincv[k] = False
        indtraincv = indcv[indtraincv].ravel()
    
        Xtraincv = Xtrain[indtraincv].tolist()
        ytraincv = ytrain[indtraincv].tolist()
        
        Xcv = Xtrain[indcv[k]].tolist()
        ycv = ytrain[indcv[k]].tolist()
        
        prob = svm_problem(ytraincv, Xtraincv)
        param = svm_parameter('-t 1 -d ' + str(Q) + ' -g 1 -r 1 -c ' + str(Cwin))
        
        m = svm_train(prob, param)
        ypred, yacc, yprob = svm_predict(ycv, Xcv, m, options = '-q')
        
        Ev[k] = 1 - yacc[0]/100
    
    Ecvwin[i] = Ev.mean()
            
Ecvwin = Ecvwin.mean()

print('Most selected C =', Cwin, ', Mean Ecv =', Ecvwin)
            
            