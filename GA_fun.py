# Attempt to reproduce Goldstein-Armijo loop I previously created in MATLAB.
# For use in optimization.
# 19 Oct 2019 Created
# 29 Jan 2020 Added to atom/python/chromebook
# Updated 30 Jan 2020
# for use with Steep.py

# need x,d,f,gf to start; examples
#  x = np.array([5,1]); d = np.array([1,1])
#  f = lambda x: x[0]**2 + x[1]**2
# gf = lambda x: np.array([2*x[0], 2*x[1]])

def GA_fun(x,d,f,gf):
    """
    Goldstein-Armijo line search.
      x = np.array([x1,x2,...xn])
      f = lambda x: some function of x
     gf = lambda x: some gradient function of f(x)
    lam = returns lam, a stepsize
    """
    import numpy as np
    # Prep Goldstein-Armijo Loop
    lam = 1 # Initialize lambda to 1 (half it after each iteration if too large)
    a = 1e-4 # Must be a small numbrer s.t. 0 < alpha < 1
    B = 0.9 # Must be a large number s.t. beta < 1
    # Get candidate "new x"
    x1 = x + lam*d

    # Goldstein-Armijo Loop
    # f(x1)-f(x) > alpha*lambda*d*gf(x) % Too large stepsize
    #   d*gf(x1) < beta*d*gf(x)         % Too small stepsize
    while True:
        if f(x1) - f(x) > a*lam*np.dot(d,gf(x)): # Too large stepsize check
            lam = lam / 2                        # Cut it down if too large
            x1 = x + lam*d                       # Update x1 for check
        while np.dot(d,gf(x1)) < B*np.dot(d,gf(x)): # Too small stepsize check
            lam = lam * 1.1                         # Make bigger if too small
            x1 = x + lam*d                          # Update x1 for checks
        #print(lam) # temp
        #print(f(x1)-f(x) <= a*lam*np.dot(d,gf(x)) and np.dot(d,gf(x1)) >= B*np.dot(d,gf(x))) # temp
        #np.array([ (f(x1) - f(x)) - (a*lam*np.dot(d,gf(x))), (B*np.dot(d,gf(x))) - (np.dot(d,gf(x1))) ]) # temp
        if f(x1)-f(x) <= a*lam*np.dot(d,gf(x)) and np.dot(d,gf(x1)) >= B*np.dot(d,gf(x)): # Conditions for a good stepsize
            break
        elif lam < 1e-8: # break out of loop in case of emergency
            break
    return lam
    #print(lam,x1)
