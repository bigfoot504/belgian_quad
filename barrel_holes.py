# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 08:00:27 2021

@author: locker
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

# 26-inch diameter barrel, (x,y) coords will be num inches from lower left
bbl_dia = 26 # inches
CTR = (bbl_dia/2, bbl_dia/2)
num_holes = 5 # number of holes to drill
np.random.rand(1000,2)

# function takes in points and gives direction and distance from center
def dirdis_from_ctr(hole_coords, returns='dir_dist'):
    # assume np array with rows as holes and columns as x,y coords in inches
    dir_dist = np.zeros(hole_coords.shape)
    try: # surrogate for more than one row in hole_coords
                              # (this is bc one-row is of shape (2,) )
        for idx, hole_coord in enumerate(hole_coords):
            # Find direction
            dir_dist[idx, 0] = np.arctan2( (hole_coord[1] - CTR[1]) , 
                                           (hole_coord[0] - CTR[0]) ) *180/np.pi
            dir_dist[idx, 0] -= 90 # to shift 0deg to like-north
            if dir_dist[idx, 0] <= 0: dir_dist[idx, 0] += 360
            dir_dist[idx, 0] = 360 - dir_dist[idx, 0]
            # Find distance
            dir_dist[idx, 1] = ( (hole_coord[1] - CTR[1])**2 +
                                 (hole_coord[0] - CTR[0])**2 ) ** 0.5
    except: # surrogate condition for only one row in hole_coords
        # Find direction
        dir_dist[0] = np.arctan2( (hole_coords[1] - CTR[1]) , 
                                  (hole_coords[0] - CTR[0]) ) *180/np.pi
        dir_dist[idx, 0] -= 90 # to shift 0deg to like-north
        if dir_dist[0] <= 0: dir_dist[idx, 0] += 360
        dir_dist[0] = 360 - dir_dist[0]
        # Find distance
        dir_dist[1] = ( (hole_coords[1] - CTR[1])**2 +
                        (hole_coords[0] - CTR[0])**2 ) ** 0.5
    dir_ = dir_dist[:,0]
    dist = dir_dist[:,1]
    if returns == 'dir_dist':
        return dir_dist
    if returns == 'dir':
        return dir_
    if returns == 'dist':
        return dist

X = np.array([[13,26],
              [26,26],
              [26,13],
              [26,0],
              [13,0],
              [0,0],
              [0,13],
              [0,26]])
dirdis_from_ctr(X, returns='dist')

# Generate 1000 random coordinates inside barrel radius
water_coords = []
for i in range(10000):
    x = np.random.rand(1,2)[0] * 26
    while ((x[0]-CTR[0])**2 + (x[1]-CTR[1])**2)**0.5 >= bbl_dia/2:
        x = np.random.rand(1,2)[0] * 26
    water_coords.append(x)
water_coords = np.asarray(water_coords)

# Generate random starting location for 5 holes
X0 = []
for i in range(num_holes):
    x = np.random.rand(1,2)[0] * 26
    while ((x[0]-CTR[0])**2 + (x[1]-CTR[1])**2)**0.5 >= bbl_dia/2:
        x = np.random.rand(1,2)[0] * 26
    X0.append(x)
X0 = np.asarray(X0)

def myfun(X, water_coords):
    # X is coordinates of holes
    if X.size > 2:
        X = np.reshape(X, [int(X.size/2), 2]) # Takes in a 1xN and changes it to an N/2x2
    else:
        X = np.array([[X[0], X[1]]])
    return np.sum(cdist(X, water_coords)**2)

soln = minimize(lambda X: myfun(X, water_coords), np.reshape(X0, [1, X0.size])[0], method='nelder-mead')
print(soln)

hole_coords = np.reshape(soln.x, [int(soln.x.size/2), 2])
print(hole_coords)
dir_dis_2_holes = dirdis_from_ctr(np.reshape(soln.x, [int(soln.x.size/2), 2]))
print(dir_dis_2_holes)

idx = 0
while True:
    idx += 1
    print(idx, soln.fun)
    funval_old = soln.fun
    
    # Plot
    x = np.linspace(0, bbl_dia, num=50)
    y_pos = np.zeros(x.shape)
    y_neg = np.zeros(x.shape)
    for i, xi in enumerate(x):
        y_pos[i] = ((bbl_dia/2)**2 - (xi-bbl_dia/2)**2 )**0.5
        y_neg[i] = -y_pos[i]
    y_pos += bbl_dia/2
    y_neg += bbl_dia/2
    plt.plot(x, y_pos) # plot circle upper half
    plt.plot(x, y_neg) # plot circle lower half
    # Plot holes
    for hole_coord in hole_coords:
        plt.plot(hole_coord[0], hole_coord[1], 'bo')
    
    plt.show()
    #time.sleep(.01)
    
    hole_coords_old = hole_coords.copy()
    
    # Allocate
    allocation = np.argmin(cdist(hole_coords, water_coords), 0)
    # Locate
    for i in range(num_holes):
        # Coordinates for water for which hole i is the closest hole
        local_water_coords = water_coords[allocation == i]
        soln = minimize(lambda X: myfun(X, local_water_coords), hole_coords[i, :], method='nelder-mead')
        hole_coords[i, :] = soln.x
        
    if (hole_coords == hole_coords_old).all() or abs(soln.fun - funval_old) < 1e1:
        break

print(hole_coords)
dir_dis_2_holes = dirdis_from_ctr(hole_coords)
print(dir_dis_2_holes)
print(np.mean(dir_dis_2_holes[:,1]))
print(np.pi*np.mean(dir_dis_2_holes[:,1])**2, np.pi*13**2, np.pi*np.mean(dir_dis_2_holes[:,1])**2 / (np.pi*13**2))

















