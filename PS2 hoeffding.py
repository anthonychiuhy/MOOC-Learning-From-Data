# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:48:13 2017

@author: HOME1
"""
import random,time

random.seed(time.time())

numexp = 10000
numcoins = 1000
numflips = 10

nu1 = []
nurand = []
numin = []

for exp in range(0,numexp):
    
    hts = [[random.randint(0,1) for j in range(0,numflips)] for i in range(0,numcoins)] # 0 = tail ; 1 = head
    
    minht = sum(hts[0])
    minindex = 0
    for i in range(0,numcoins):
        sumht = sum(hts[i])
        if sumht < minht:
            minht = sumht
            minindex = i
    
    cmin = hts[minindex]
    c1 = hts[0]
    crand = hts[random.randint(0,numcoins-1)]
    
    nu1.append(sum(c1)/numflips)
    nurand.append(sum(crand)/numflips)
    numin.append(sum(cmin)/numflips)

print('numin',sum(numin)/numexp)
print('nu1',sum(nu1)/numexp)
print('nurand',sum(nurand)/numexp)
    
    
