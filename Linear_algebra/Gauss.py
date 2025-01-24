# Functions for Gauss elimination and backsubstitution. Written in lecture 5, CSCI/MATH2072U, OnTechU, 2025.
# Note: written so that i and j are indices starting from 1. Be careful to avoid off-by-one errors because of the Python indexing and the odd behaviour of the 'range' function!
import numpy as np

# This parameter is close to machine accuracy. A warning is printed if a pivot is smaller than this in absolute value.
small = 1e-14

# Auxiliary function to swap two rows in an array. In: n by m array of floats B, indices of rows to be swapped, a and b. Out: n by m array of floats equal to input B but with rows a and b swapped.
def swap(B,a,b):
    dum = np.copy(B[a-1,:])
    B[a-1,:] = np.copy(B[b-1,:])
    B[b-1,:] = np.copy(dum)
    return B

# In: n X m array of floats where m > (n-1). Out: array of floats B of the same shape as A, result of Gauss elimination. 
def Gauss(A):
    # NOTE: we use "np.copy" here to make sure we are copying values, not pointers.
    B = np.copy(A)
    n = np.shape(A)[0]
    for j in range(1,n):                  # Outer loop over columns.
        # Select the best pivot element (better than picking any non-zero one!):
        p = np.argmax(abs(B[j-1,j:])) + j # Note that "argmax" returns the index to the vector B[j-1,j:], we add j to get the index to the full column vector.
        B = swap(B,j,p)                   # Swap rows so that the diagonal element is greater (in abs value) than the vlues belwo it in the column.
        if abs(B[j-1,j-1]) < small:       # If the best available pivot is still close to 0, print a warning.
            print('Warning: small pivot!')
        for i in range(j+1,n+1):          # Compute the multiplyer and Gauss eliminate one row.
            m = B[i-1,j-1] / B[j-1,j-1]
            B[i-1,:] = B[i-1,:] - m * B[j-1,:]
    return B

# Backsubstitution, normally to follow a call to Gauss. Input: n by (n+1) array of floats in row-echelon form (upper triangular nXn block augmented by the right-hand-sides). Out: (n,) shaped array that holds the solution vector.
def backsub(A):
    n = np.shape(A)[0]
    x = np.empty((n,))
    x[n-1] = A[n-1,n] / A[n-1,n-1]             # First solve the last equation "A[n-1,n-1] x[n-1] = A[n-1,n]".
    for i in range(n-1,0,-1):                  # Work your way back to the first equation, substituting known
        dum = 0.0                              # elements of solution x.
        for j in range(i+1,n+1):
            dum += A[i-1,j-1] * x[j-1]
        x[i-1] = (A[i-1,n] - dum) / A[i-1,i-1]
    return x

