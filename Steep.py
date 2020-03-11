# Steepest descent algorithm
# For use with GA_fun.py

# need x,d,f,gf to start; examples
#x0 = np.array([5,1]); d = np.array([1,1])
#f = lambda x: x[0]**2 + x[1]**2
#gf = lambda x: np.array([2*x[0], 2*x[1]])
#max_d = 5
#eps = 1e-8 # epsilon convergence criteria

def Steep(x0,max_d,f,gf,eps):
    import numpy as np
    from GA_fun import GA_fun as ga

    x1 = x0 # starting point

    # Steepest Descent Loop
    for k in range(1000): # iteration cap failsafe in case it doesn't converge

        # Set convergence/stopping criteria
        if ( np.linalg.norm(gf(x1)) / (1+abs(f(x1))) <= eps ):
            break

        # Set "d" to the direction of descent (negative gradient)
        d = -gf(x1)
        # Keep direction but shrink magnitude (to avoid overstepping to another
        # valley in case obj fun not convex)
        #d = d / np.linalg.norm(d) / 2 * (0.8+0.4*np.random.rand())*max_d # Randomize exact direction length to avoid cycling into a pattern
        #d = d * (0.99 + 0.02*np.random.rand())
        # Goldstein-Armijo Line Search
        lam = ga(x1,d,f,gf)

        # Take step
        x1 = x1 + lam*d

    return x1
