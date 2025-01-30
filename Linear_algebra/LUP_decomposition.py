# Code for LU decomposition, forward and backward substitution. Programmed (mostly) in class.
# Work in progress! We will add the swapping of rows following the same rule as Gauss elimination in the same folder.
import numpy as np

# When a pivot smaller than this number is found, it is considered equal to zero and an error is thrown.
small = 1e-13

# Swap two whole rows. Input: n by m array B, row indices a and b. Output: Array B with rows a nd b swapped.
def swap(B,a,b):
    dum = np.copy(B[a-1,:])
    B[a-1,:] = np.copy(B[b-1,:])
    B[b-1,:] = np.copy(dum)
    return B
# Swap rows a and b for columns 1 through b-1. Note that, in contrast to "swap", the order of the row indices is important as the second is taken to be the current column of the elimination process.
# Input: n by m array B, row idices a and b. Index b should also be the index of the current column.
def swapL(B,a,b):
    dum = np.copy(B[a-1,0:b-1])
    B[a-1,0:b-1] = np.copy(B[b-1,0:b-1])
    B[b-1,0:b-1] = np.copy(dum)
    return B

# PA=LU decomposition with row swapping (pivoting).
# Input: n by n array of floats A. Output: factors L and U, permutation matrix P, boolean succes (set to False if a pivot smaller than variable "small" is encountered).
def LUP(A):
    n = np.shape(A)[0] # n is the nr of rows in A
    U = np.copy(A)
    L = np.identity(n)
    P = np.identity(n)
    success = True     # Most matrices are invertable, so we set the flag to True by default.
    for j in range(1,n):
        # Select the largest element (in abs value) on or below the diagonal.
        k = np.argmax(abs(U[j-1:,j-1])) + j
        U = swap(U,k,j)
        P = swap(P,k,j)
        # Swap rows k and j of L only up to - not including - column j:
        if j > 1:
            L = swapL(L,k,j)
        # If the largest available pivot is too close to zero, warn the user and stop.
        if abs(U[j-1,j-1]) < small:
            success = False
            break
        # The actual elimination: create zeros below the diagonal:
        for i in range(j+1,n+1):
            L[i-1,j-1] = U[i-1,j-1] / U[j-1,j-1]
            U[i-1,:] = U[i-1,:] - L[i-1,j-1] * U[j-1,:]
    return L, U, P, success

# Solve a linear system with n by n upper triangular matrix A and n by 1 righ-hand side r (all floats).
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

# Solve a linear system with n by n lower triangular matrix A and n by 1 righ-hand side r (all floats).
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


