#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:22:38 2020

@author: rmlocke84
"""

from matplotlib.pyplot import plot, draw, show
# from bokeh.plotting import figure, output_file, show
import numpy as np
import math
import statistics as stats
import matplotlib.pyplot as plt
from scipy.stats import norm

while True:
    lda = float(input('arrival rate lambda:'))
    mu = float(input('service rate mu:'))
    N = int(input('maximum event:'))
    Nsim = int(input('total number of simulations:'))
    N0 = int(input('initial queue length:'))
    W = np.zeros([N])
    
    for k in range(1,Nsim):
        t = 0  # current time of system
        Output = np.zeros([1,3])
        NA = 0 # num arrivals
        ND = 0 # num departures
        n = N0 # num in system
        if n > 0:                               # if system nonempty
            U1 = np.random.rand(1)
            tD = -math.log(U1) / mu             # generate interdeparture time
        else:
            tD = 1000000000000000;
        tA = -math.log(np.random.rand(1)) / lda # generate interarrival time
        Output = np.zeros([N0,3])
        nvec = np.zeros(0) # temp
        tAs = np.zeros(0) # temp
        tDs = np.zeros(0) # temp
        
        while ND < N + N0:
            nvec = np.append(nvec, n) # temp
            if tA <= tD:    # if next event is arrival
                t = tA      # update time of system
                NA = NA + 1 # index num arrivals
                n = n + 1   # index num in system
                tA = t - math.log(np.random.rand(1)) / lda # generate interarrival time
                if n == 1:  # if system just left empty state
                    Y = -math.log(np.random.rand(1)) / mu # generate service time
                    tD = t + Y                            # time of next departure
                Output = np.row_stack((Output, [NA, t, 0])) # record arrival time
            else:           # if next event is departure
                t = tD      # update time of system
                ND = ND + 1 # index num departures
                n = n - 1   # index num in system
                if n == 0:  # if system becomes empty
                    tD = 1000000000000000
                else:       # if system becomes nonempty
                    Y = -math.log(np.random.rand(1)) / mu # generate service time
                    tD = t + Y        # time of next departure
                Output[ND - 1, 2] = t # record time of recent departure
        w = Output[N0 : N0 + N, 2] - Output[N0 : N0 + N, 1] # record wait time
        W = W + w
        plt.plot(range(len(nvec)), nvec); input('Hit ENTER:') # temp
        plt.show() # temp
        print('Sim number ',k) # temp
        
    tAs = Output[1:,1] - Output[:len(Output)-1,1] # temp; interarrival times
    tDs = Output[1:,2] - Output[:len(Output)-1,2] # temp; interdeparture times
    plt.hist(tAs) # temp
    plt.hist(tDs) # temp
    EW = W / Nsim
    plt.plot(np.linspace(0, len(EW), len(EW)), EW)
    plt.show()
    
    
    
    
    
    
    
    
    