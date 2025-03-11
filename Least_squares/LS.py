# Least squares approximation of fixed order. By L. van Veen, Ontario Tech U, 2025.
import numpy as np

# Function to compute the least-squares polynomial approximant of order o to date (x_i,y_i), i=0..n.
# In: np arrays of floats x and y hold the data, o (positive integer) is the order of the approximant.
# Out: function handle P for the least-squares approximant.
def LS(x,y,o):
# Number of data points:
    n = np.shape(x)[0] - 1
# Allocate (first o+1 columns of the) Vandermonde matrix
    V = np.zeros([n+1,o+1])
# Compute elements in order O(on) time
    for i in range(0,n+1):
        V[i,0] = 1.0
        for j in range(1,o+1):
            V[i,j] = V[i,j-1] * x[i]
# Find least-squares solution
    a = np.linalg.lstsq(V,y,rcond=None)[0]  # The [0] at the end serves to select the first output element
    def P(z):                       # Construct the polynomial to return
        S = a[o-1] + a[o]*z   # Use Horner's algorithm so that P(x) is computed in O(o) time (see Lec 9, slide 13).
        for k in range(o-2,-1,-1):
            S = a[k][0] + S * z
        return S
    return P
 
