# A function for Newton iteration. Written in class in week 2. MATH/CSCI 2072U, Ontario Tech U, 2025.
import numpy as np

# In: function handles f and fp (derivative), initial guess x0, tolerance for the error (eps_x) and residual (eps_f) and the max nr of iterations kMax.
def Newton(f,fp,x0,eps_x,eps_f,kMax):
    # Initialize: dummy variables that change in the loop and the convergence flag.
    x = x0
    conv = False
    # The Newton iteration loop.
    for i in range(kMax):
        # Compute the update step as -f/f':
        r = f(x)
        dx = -r/fp(x)
        # Update the approximate solution:
        x += dx
        # Compute the residual, estimate the error:
        err = abs(dx)
        res = abs(r)
        # Check convergence:
        if err < eps_x and res < eps_f:
            conv = True
            break
    return x, err, res, conv
