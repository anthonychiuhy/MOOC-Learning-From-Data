# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 00:32:43 2018

@author: HOME1
"""

import numpy as np

eta = 0.1
i = 0

def Err(v):
    return (v[0]*np.exp(v[1]) - 2*v[1]*np.exp(-v[0]))**2

def diffErr(v):
    return 2 * (v[0]*np.exp(v[1]) - 2*v[1] * np.exp(-v[0])) * np.array([np.exp(v[1]) + 2*v[1]*np.exp(-v[0]) , v[0]*np.exp(v[1]) - 2*np.exp(-v[0])])

v = np.array([1,1])
E = Err(v)

while(E > 1e-14):
    v = v - eta*diffErr(v)
    E = Err(v)
    i += 1

print('i = ',i)
print('v = ',v)

v = np.array([1,1])

for i in range(15):
    v = v - eta * np.array([diffErr(v)[0] , 0])
    v = v - eta * np.array([0, diffErr(v)[1]])
    
E = Err(v)

print('Err = ',E)