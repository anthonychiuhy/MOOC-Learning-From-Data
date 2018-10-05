# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:34:54 2017

@author: HOME1
"""

import numpy as np

N = 1000
loop = 1000

gws = np.zeros([loop,6])
for i in range(0,loop):
    # generate random points and evaluate target function
    
    pts = np.random.uniform(-1,1,[N,2])
    pts = np.concatenate([np.ones([N,1]),pts],1)
    
    fs = np.sign(pts[:,1]**2 + pts[:,2]**2 - 0.6) # f = x1^2 + x2^2 - 0.6
    
    
    # generate noise by flipping sign to random 10% of samples
    
    fs[np.random.choice(N,int(0.1*N),False)] *= -1
    
    
    # nonlinear transformation
    
    transpts = np.zeros([N,6])
    transpts[:,0:3] = pts
    transpts[:,3] = pts[:,1]*pts[:,2]
    transpts[:,4] = pts[:,1]**2
    transpts[:,5] = pts[:,2]**2
    
    
    # linear regression
    
    pseudoinv = np.dot(np.linalg.inv(np.dot(transpts.T,transpts)),transpts.T)
    gw = np.dot(pseudoinv,fs)
    
    gws[i] = gw

avergw = np.average(gws,0)



testN = 10000

Eouts = np.zeros(loop)
for i in range(0,loop):
    testpts = np.random.uniform(-1,1,[testN,2])
    testpts = np.concatenate([np.ones([testN,1]),testpts],1)
    
    fs = np.sign(testpts[:,1]**2 + testpts[:,2]**2 - 0.6)
    fs[np.random.choice(testN,int(0.1*testN),False)] *= -1
    
    testtranspts = np.zeros([testN,6])
    testtranspts[:,0:3] = testpts
    testtranspts[:,3] = testpts[:,1]*testpts[:,2]
    testtranspts[:,4] = testpts[:,1]**2
    testtranspts[:,5] = testpts[:,2]**2
    
    gs = np.sign(np.dot(testtranspts,gws[i]))
    
    Eouts[i] = np.average(gs != fs)

Eout = np.average(Eouts)

print('Eout = ', Eout)
