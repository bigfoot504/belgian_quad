# 16 Oct 2019 Created
# 29 Jan 2020 Added to atom/python/chromebook
# 3 Feb 2020 modified

import numpy as np
import random

eps = 1e-4
f1 = lambda x: x[0] - x[1] + 2*x[0]**2 + 2*x[0]*x[1]
f2 = lambda x: x[0]**2 + 2*x[1]**2 - 2*x[0]*x[1] - 2*x[1]
f = f1
g = lambda x: x[0]**2 + x[1]**2 - 100 # constraint (g(x)<=0)

# starting triangle w/ vertices inside a length 14 square centered at origin
x0 = np.random.rand(2,3)*14 - 7
x = x0

for i in range(500): # main loop
    
    x_y = np.append(x, [f(x)], axis=0) # place function values under each vertex
    x_y = x_y[ :, x_y[2,:].argsort() ] # sort vertices by function value
    xL = x_y[:2,0]; xM = x_y[:2,1]; xH = x_y[:2,2]
    
    # temp
    #print(x_y); print('Flip # ',i) # temp
    #input('Hit ENTER to continue') # temp
    
    # test convergence criteria
    if np.linalg.norm(xH-xM) < eps and np.linalg.norm(xH-xL) < eps and np.linalg.norm(xL-xM) < eps:
        break
    
    # make x0 be halfway between xM and xL
    x0 = (xM + xL) / 2
    
    # alpha is stepsize scalar for triangle flip
    alpha = np.random.rand()*.4 + .8
    
    # Step/flip in direction of x0-xH scaled by alpha
    xR = x0 + (x0-xH)*alpha
        
    # Extension: Extend xR if f(xR) < f(xL) (~line search)
    while f(xR) < f(xL):
        alpha = alpha * 1.2
        xR = x0 + (x0-xH)*alpha
        #print('Extend') # temp
        if g(xR) > 0:
            #print('Extended too far') # temp
            break # keep xR from running away outside FR
    
    # Shortening: Shorten xR to meet constraint
    while g(xR) > 0:
        alpha = alpha * 0.9
        xR = x0 - (xH-x0)*alpha
        #print('Ifeasible, Shorten') # temp
    
    # Contraction: If f(xR) > f(xH), reduce triangle to xL, x0, (x0+xH)/2
    if f(xR) >= f(xH):
        x = np.array([(x0+xH) / 2, x0, xL]).T
        #print('Contract') # temp
    else:
        x = np.array([xR, xM, xL]).T

# Post-processing
x = np.mean(x,1)
y = f(x)

# Display result
print('x =', x, ' f(x) =',y)

# Plot prep
numpts = 50
xvals = np.linspace(x[0]-10,x[0]+10,numpts)
yvals = np.linspace(x[1]-10,x[1]+10,numpts)
idx = range(numpts)
X, Y = np.meshgrid(xvals, yvals)
Z = np.zeros([numpts,numpts])
for i in idx:
    for j in idx:
        #X[i,j] = xvals[i]
        #Y[i,j] = xvals[j]
        Z[i,j] = f( np.array([X[i,j], Y[i,j]]) )
        
# Plot result
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.view_init(60, 10)
surf = ax.plot_surface(X,Y,Z)
#plt.show()
# prep subplot
X2 = np.linspace(-10,10,numpts)
Y2 = np.sqrt(100 - X2**2)
X2 = np.concatenate((X2,  X2))
Y2 = np.concatenate((Y2, -Y2))
Z2 = 100*np.ones(100)
ax.plot3D(X2, Y2, Z2)
ax.scatter3D(x[0], x[1], y)
plt.show()












