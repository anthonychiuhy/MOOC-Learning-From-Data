# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 13:13:59 2017

@author: HOME1
"""

import random,time

random.seed(time.time())

def sign(num):
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0


#define the correct line
lowbound = -1
upbound = 1
pt1 = (random.uniform(lowbound,upbound),random.uniform(lowbound,upbound)) #(x1,x2)
pt2 = (random.uniform(lowbound,upbound),random.uniform(lowbound,upbound))
wcorrect = (pt1[0]*(pt2[1] - pt1[1]) - pt1[1]*(pt2[0] - pt1[0]), pt1[1] - pt2[1], pt2[0] - pt1[0])


#create random points
N = 1000 #number of data points
pts = [(random.uniform(lowbound,upbound),random.uniform(lowbound,upbound)) for i in range (0,N)]

#initialise weights of PLA
w = [0,0,0] 


#start PLA algorithm
numofiter = 0
while True:
    #find unclassified points
    unclasspts = []
    for i in pts:
        label = sign(w[0] + i[0]*w[1] + i[1]*w[2])
        correctlabel = sign(wcorrect[0] + i[0]*wcorrect[1] + i[1]*wcorrect[2])
        if label != correctlabel:
            unclasspts += [i]
    
    if len(unclasspts) != 0:
        ranpt = (1,) + random.choice(unclasspts) #pick random point as (x0,x1,x2)
        correctlabel = sign(sum(wcorrect[i]*ranpt[i] for i in range(0,3)))
        
        w = [w[i] + correctlabel*ranpt[i] for i in range(0,3)]
        
        numofiter += 1
    else:
        break

        
#PLA = [sign(w[0] +  w[1]*pts[i][0] + w[2]*pts[i][1]) for i in range(0,N)]
#correct = [sign(wcorrect[0] +  wcorrect[1]*pts[i][0] + wcorrect[2]*pts[i][1]) for i in range(0,N)]

#print(PLA)
#print(correct)
#print([PLA[i] == correct[i] for i in range(0,N)])
print('number of iterations = ', numofiter)