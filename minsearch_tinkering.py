# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:11:10 2019

@author: rmlocke
"""

import numpy as np
from Steep import Steep as steep

x0 = np.array([5,1]);
f = lambda x: (x[0]-1)**2 + (x[1]+9)**2
gf = lambda x: np.array([2*x[0]-2, 2*x[1]+18])
max_d = 5
eps = 1e-8 # epsilon convergence criteria

x1 = steep(x0,max_d,f,gf,eps)
print('x*=',x1,', f(x*)=',f(x1),', gradf(x*)=',gf(x1))
