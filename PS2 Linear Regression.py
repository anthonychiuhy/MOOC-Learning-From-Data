# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:04:23 2017

@author: HOME1
"""

import numpy as np

loop = 1000

N = 10


# create target function (line)

# x1 = fpts[0,0]
# x2 = fpts[1,0]
# y1 = fpts[0,1]
# y2 = fpts[1,1]

fws = np.zeros([loop,3])
gws = np.zeros([loop,3])

PLAiter = np.zeros(loop)

for i in range(0,loop):
    fpts = np.array(np.random.uniform(-1,1,[2,2]))
    
    fw0 = fpts[0,0]*(fpts[1,1]-fpts[0,1]) - fpts[0,1]*(fpts[1,0]-fpts[0,0])
    fw1 = fpts[0,1] - fpts[1,1]
    fw2 = fpts[1,0] - fpts[0,0]
    fw = np.array([fw0,fw1,fw2])
    
    fws[i] = fw
    
    
    # create random N labeled points
    
    pts = np.random.uniform(-1,1,[N,2])
    pts = np.concatenate([np.ones([N,1]),pts],1)
    
    fys = np.sign(np.dot(pts,fw))
    
    
    # linear regression
    
    pseudinv = np.dot(np.linalg.inv(np.dot(pts.T,pts)), pts.T)
    gw = np.dot(pseudinv,fys)
    
    
    # Perceptron Learning Algorithm
    
    while True:
        gys = np.sign(np.dot(pts,gw))
        
        ydisagree = gys != fys
        
        if True in ydisagree:
            disagreepts = pts[ydisagree]
            disagreefys = fys[ydisagree]
            index = np.random.randint(0,len(disagreepts))
            
            gw = gw + disagreefys[index]*disagreepts[index]
            
            PLAiter[i] += 1
            
        else:
            break
    
    gws[i] = gw
    

# estimate out of sample error

N2 = 10000
Eout = np.zeros(loop)

for i in range(0,loop):
    
    pts = np.random.uniform(-1,1,[N2,2])
    pts = np.concatenate([np.ones([N2,1]),pts],1)
    
    fys = np.sign(np.dot(pts,fws[i]))
    gys = np.sign(np.dot(pts,gws[i]))
    
    disagree = fys != gys
    
    Eout[i] = np.average(disagree)
    

# calculate results
    
averPLAiter = np.average(PLAiter)
averEout = np.average(Eout)

print('average number of PLA iterations = ', averPLAiter)
print('Eout = ', averEout)