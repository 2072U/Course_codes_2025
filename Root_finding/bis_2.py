# Bisection as programmed in class (8pm section). Ontario Tech U, 2025.
# Incorrect as is, see the comment below:
import numpy as np

# Inputs: function handle f, initial left and right boundary l and r, tolerance for error and residual eps_x and eps_f (all of type float), max number of iterations (positive integer).
def bis(f,l,r,eps_x,eps_f,kMax):
    a = l
    b = r
    conv = False
    val = f(a)                          # Correct this mistake! "val" must be updated
    for k in range(kMax):               # when the left boundary is updated!
        m = (a+b)/2.0
        newval = f(m)
        if val * newval < 0.0:          # Bisection step: discard half the domain.
            b = m
        else:
            a = m
        err = abs(b-a)                  # Upper bound for error.
        res = abs(f(m))                 # Residual.
        if err < eps_x and res < eps_f: # If converged, set convergence flag and exit.
            conv = True
            break
        
    return m,err,res,conv
