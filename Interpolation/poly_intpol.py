# Polynomial interpolation as programmed in class. L. van Veen, Ontario Tech U, 2025.
import numpy as np
import matplotlib.pyplot as plt

# Solve a linear system with n by n lower triangular matrix A and n by 1 righ-hand side r (all floats).
def forwardsub(A,r):
    n = np.shape(A)[0]
    x = np.empty((n,))
    x[0] = r[0] / A[0,0]                     # First solve the first equation "A[0,0] x[0] = r[0]".
    for i in range(2,n+1):                   # Work your way back to the last equation, substituting known
        dum = 0.0                            # elements of solution x.
        for j in range(1,i):
            dum += A[i-1,j-1] * x[j-1]
        x[i-1] = (r[i-1] - dum) / A[i-1,i-1]
    return x

# Evaluate a polynomial in the form a0 + a1 phi1 + ... where phi1=(x-xs0), phi2=(x-xs0)(x-xs1) ...
# Input: argument x of the polynomial (float); node locations xs (array of floats) and coefficients a (array of floats). Out: the value of the polynomial at x.
def P(x,xs,a):
    n = np.shape(a)[0]-1
    Q = a[0]
    d = 1
    for i in range(1,n+1):
        d = d * (x - xs[i-1])
        Q = Q + a[i] * d
    return Q

# Funtion for finding a polynomial interpolant. Uses the basis functions 1, (x-x0), (x-x0)*(x-x1), ... to arrive at a triangular linear system.
# Input: nodes locations xs (list of floats) and y-values (list of floats), grid of x-values to evaluate the ploynomial on.
# Output: list of polynomial coefficients b (array of floats) and array of function values y (smae shape and kind as x_grid).
def poly_int(xs,ys,x_grid):
    n = len(xs) - 1
    # Pre-allocate and compute the matrix:
    C = np.zeros((n+1,n+1))
    C[:,0] = 1.0
    for i in range(1,n+1):
        for j in range(1,i+1):
            C[i,j] = C[i,j-1] * (xs[i] - xs[j-1])
    # Solve the linear system:
    b = forwardsub(C,np.array(ys))
    # Allocate and compute the array of y-values:
    m = len(x_grid)
    y = np.copy(x_grid)
    for i in range(m):
        y[i] = P(x_grid[i],xs,b)
    return b, y

