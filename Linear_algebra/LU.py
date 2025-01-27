# Code for LU decomposition, forward and backward substitution. Programmed (mostly) in class.
# Work in progress! We will add the swapping of rows following the same rule as Gauss elimination in the same folder.
import numpy as np

# First code for LU decomposition without row swapping.
def LU(A):
    n = np.shape(A)[0] # n is the nr of rows in A
    U = np.copy(A)
    L = np.identity(n)
    for j in range(1,n):
        for i in range(j+1,n+1):
            L[i-1,j-1] = U[i-1,j-1] / U[j-1,j-1]
            U[i-1,:] = U[i-1,:] - L[i-1,j-1] * U[j-1,:]
    return L, U

# Solve a linear system with upper triangular matrix A and righ-hand side r.
def LUbacksub(A,r):
    n = np.shape(A)[0]
    x = np.empty((n,))
    x[n-1] = r[n-1] / A[n-1,n-1]               # First solve the last equation "A[n-1,n-1] x[n-1] = r[n-1]".
    for i in range(n-1,0,-1):                  # Work your way back to the first equation, substituting known
        dum = 0.0                              # elements of solution x.
        for j in range(i+1,n+1):
            dum += A[i-1,j-1] * x[j-1]
        x[i-1] = (r[i-1] - dum) / A[i-1,i-1]
    return x

# Solve a linear system with lower triangular matrix A and righ-hand side r.
def LUforwardsub(A,r):
    n = np.shape(A)[0]
    x = np.empty((n,))
    x[0] = r[0] / A[0,0]                     # First solve the first equation "A[0,0] x[0] = r[0]".
    for i in range(2,n+1):                   # Work your way back to the last equation, substituting known
        dum = 0.0                            # elements of solution x.
        for j in range(1,i):
            dum += A[i-1,j-1] * x[j-1]
        x[i-1] = (r[i-1] - dum) / A[i-1,i-1]
    return x


