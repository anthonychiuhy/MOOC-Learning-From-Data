# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:55:22 2018

@author: HOME1
"""

import numpy as np

runs = 100
numepoch = 0
Ein = 0
Eout = 0
N = 100
N2 = 100000
eta = 0.01
# function that generates random function f
def randf():
    pts = np.random.uniform(-1,1,[2,2])
    delx = pts[1,0] - pts[0,0]
    dely = pts[1,1] - pts[0,1]
    m = dely/delx
    c = pts[0,1] - m*pts[0,0]
    def f(x):
        if m*x[1] - x[2] + c >= 0:
            return 1
        else:
            return 0
    return f

for run in range(runs):
    # generate (x,y)s and initialise w
    f = randf()
    x = np.random.uniform(-1,1,[N,2])
    x = np.concatenate((np.ones([N,1]),x),1)
    
    y = np.zeros(np.array([N,1]))
    for i in range(N):
        y[i] = f(x[i])
        if y[i] == 0:
            y[i] = -1
    
    w = np.zeros(3)
    
    # stocastic gradient decent
    while True:
        winit = w
        for i in np.random.permutation(N):
            gradErr = -y[i]*x[i]/(1 + np.exp(y[i]*np.dot(w,x[i])))
            w = w - eta * gradErr
        if abs(np.sqrt((w - winit).dot(w - winit))) < 0.01:
            break
        numepoch += 1
    
    # calculate Ein
    sum = 0
    for i in range(N):
        sum = sum + np.log(1 + np.exp(-y[i]*w.dot(x[i])))
        
    Ein += (1/N)*sum
    #print('Ein = ', Ein)
    
    # calculate Eout(Cross Entropy) by generating new points
    x2 = np.random.uniform(-1,1,[N2,2])
    x2 = np.concatenate((np.ones([N2,1]),x2),1)
    
    y2 = np.zeros(np.array([N2,1]))
    for i in range(N2):
        y2[i] = f(x2[i])
        if y2[i] == 0:
            y2[i] = -1
    
    sum = 0
    for i in range(N2):
        sum = sum + np.log(1 + np.exp(-y2[i]*w.dot(x2[i])))
        
    Eout += (1/N2)*sum
    
    #print('Eout =', Eout)

numepoch /= runs
Ein /= runs
Eout /= runs

print('Average Epochs = ', numepoch)
print('Average Ein = ', Ein)
print('Average Eout =', Eout)