# This code runs steepest descent method with the Goldstein-Armijo function
# defined internally and attempting to recode it so that it passes an object
# for optimization.
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:29:23 2019

@author: rmlocke
"""

# Class for optimization problem
class OptProb:
    pass
# Class for step iterations within problem
class Step:
    pass

# Goldstein-Armijo function (conducts a loop) to determine step size.

def GA_fun(step,prob):
    # Prep Goldstein-Armijo Loop
    lam1 = 1 # Initialize candidate lambda to 1
    # (half lambda after each iteration if too large)
    a = 1e-4  # Must be a small number s.t. 0 < alpha < 1
    B = 0.9 # Must be a large number s.t. beta < 1
    # Get candidate "new x" = x1
    x1 = step.x + lam1*step.d

    # Goldstein-Armijo Loop
    # f(x1)-f(x) > alpha*lambda*d*gf(x) % Too large stepsize
    #   d*gf(x1) < beta*d*gf(x)         % Too small stepsize
    while True:
        #print(prob.f(x1) - prob.f(step.x));# temp
        #print(a*lam1*np.dot(step.d,prob.gf(step.x)))# temp
        #print(prob.f(x1) - prob.f(step.x) > a*lam1*np.dot(step.d,prob.gf(step.x)))# temp
        if prob.f(x1) - prob.f(step.x) > a*lam1*np.dot(step.d,prob.gf(step.x)): # Too large stepsize check
            lam1 = lam1 / 2                   # Cut it down if too large
            x1 = step.x + lam1*step.d         # Update x1 for checks
        while np.dot(step.d,prob.gf(x1)) < B*np.dot(step.d,prob.gf(step.x)): # Too small stepsize check
            lam1 = lam1 * 1.1                   # Make bigger if too small
            x1 = step.x + lam1*step.d           # Update x1 for checks
        #print(lam) # temp
        #print(f(x1)-f(x) <= a*lam*np.dot(d,gf(x)) and np.dot(d,gf(x1)) >= B*np.dot(d,gf(x))) # temp
        #np.array([ (f(x1) - f(x)) - (a*lam*np.dot(d,gf(x))), (B*np.dot(d,gf(x))) - (np.dot(d,gf(x1))) ]) # temp
        if prob.f(x1)-prob.f(step.x) <= a*lam1*np.dot(step.d,prob.gf(step.x)) \
        and np.dot(step.d,prob.gf(x1)) >= B*np.dot(step.d,prob.gf(step.x)): # Conditions for a good stepsize
            break
        elif lam1 < 1e-8: # break out of loop in case of emergency
            break
    step.lam = lam1
    # only the step-size lam has been modified in the object
    # the step on x still has not yet been taken
    return step
#print(lam,x1)



# Steepest descent function.

# need x,d,f,gf to start; examples
#x0 = np.array([5,1]); d = np.array([1,1])
#f = lambda x: x[0]**2 + x[1]**2
#gf = lambda x: np.array([2*x[0], 2*x[1]])
#max_d = 5
#eps = 1e-8 # epsilon convergence criteria

def Steep(prob):
    step = Step() # make step object
    step.x = prob.x0 # x is where we are currently (x0 is start pt for the
    # whole problem)

    # Steepest Descent Loop
    for k in range(1000): # iteration cap failsafe in case it doesn't converge

        # Test convergence/stopping criteria
        if ( np.linalg.norm(prob.gf(step.x)) / (1+abs(prob.f(step.x))) <= prob.eps ):
            break

        # Set "d" to the direction of descent (negative gradient)
        step.d = -prob.gf(step.x)
        # Keep direction but shrink magnitude (to avoid overstepping to another
        # valley in case obj fun not convex)
        #d = d / np.linalg.norm(d) / 2 * (0.8+0.4*np.random.rand())*max_d # Randomize exact direction length to avoid cycling into a pattern
        #d = d * (0.99 + 0.02*np.random.rand())
        # Goldstein-Armijo Line Search
        step = GA_fun(step,prob)

        # Take step
        step.x = step.x + step.lam*step.d

    xs = step.x # xs=x* the optimal solution

    return xs



# Sample Implementation Below Here
import numpy as np

# optimization problem object (only problem-based items go here - NOT
# search iteration info)
prob = OptProb()
prob.x0 = np.array([5,1])
prob.f = lambda x: .5*x[0]**2 + x[1]**2
prob.gf = lambda x: np.array([x[0], \
                              2*x[1]])
#prob.max_d = 5
prob.eps = 1e-8 # epsilon convergence criteria

xs = Steep(prob)
print('x*=',xs,', f(x*)=',prob.f(xs),', gradf(x*)=',prob.gf(xs))



# Try some plotting
from mpl_toolkits import mplot3d
%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-10,10,50)
y = np.linspace(-10,10,50)
X, Y = np.meshgrid(x,y)
f = lambda x,y: .5*x**2 + y**2
Z = f(X,Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#ax = plt.axes(projection='3d')
ax.scatter(np.random.rand(10)*20-10,np.random.rand(10)*20-10,np.random.rand(10)*200)
