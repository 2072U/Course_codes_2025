# An example of poorly conditioned linear solving. For MATH/CSCI2072U, OnTechU, 2025. By L. van Veen.
# For background material, see lecture 7.
import numpy as np
from LUP_decomposition import *
import matplotlib.pyplot as plt

# We create a N by N matrix with elements V_{ij}=(-1)^{i+j} / (i+2 j). Its condition number grows rapidly with size N.
def makeV(N):
    V = np.empty((N,N))
    for i in range(1,N+1):
        for j in range(1,N+1):
            V[i-1,j-1] = (-1.0)**(i+j) / float(i+2*j)
    return V

# Now we loop over matrix sizes N and solve the system
# V x = b
# where b is the last column vector of V. The exact solution is then (0,0, ..., 0, 1).
errors = []                                                   # Create an empty list to add resuls to for plotting.
for N in range(3,30):                                         # Loop over matrix sizes.
    V = makeV(N)                                              # Construct the matrix V.
    b = V[:,N-1]                                              # Construct the right-hand side.
    L, U, P, success = LUP(V)                                 # Compute the LUP decomposition.
    y = LUforwardsub(L,P@b)                                   # Solve Ly=Pr with forward substitution.
    x = LUbackwardsub(U,y)                                    # Solve Ux=y with backward substitution.
    exact = np.zeros(N,)                                      # Construct the exact solution.
    exact[N-1] = 1.0
    rerror = np.linalg.norm(x-exact) / np.linalg.norm(exact)  # Compute the relative error and residual (using the 2-norm).
    rresid = np.linalg.norm(V@x - b) / np.linalg.norm(b)
    condi = np.linalg.cond(V)                                 # Compute the condition number.
    errors.append([N,rerror,condi * rresid])                  # Store the matrix size, error and upper bound for the error.
# Plot the results. Is the actual error smaller than the upper bound? Are they close? How does the error behave as the
# matrix size grows? Why (and at what value) doe the error become more or less constant?
errors = np.array(errors)
plt.semilogy(errors[:,0],errors[:,1],'k-*',label='rel. error')
plt.semilogy(errors[:,0],errors[:,2],'-r*',label='upper bound')
plt.xlabel('matrix size')
plt.ylabel('(upper bound of) rel. error')
plt.legend()
plt.show()
