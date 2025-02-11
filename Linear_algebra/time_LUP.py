# Measuring the wall time of LUP decomposition as a function of the matrix size. By L. van Veen, for CSCI/MATH2072U.
# The wall time is a proxy for the complexity and is, in this case, expected to grwo as n^3.
import numpy as np
from time import time
import matplotlib.pyplot as plt
from LUP_decomposition import LUP # We use our own LUP function. If you use an in-built NumPy function, it will
                                  # had off the computation to Blas/LaPack a highly optimized library of linear
                                  # algebra routines and you may see unexpected results, especially for n < 10000.
# Range of matrix sizes to probe:
nStart = 1000
nEnd = 2000
# Create an empty list to add results to:
times = []
# Loop over matrix sizes:
for n in range(nStart,nEnd+1,100):
    A = np.random.rand(n,n)                     # Create a random matrix of size n X n.
    start = time()                              # Get the system clock time.
    L, U, P, success = LUP(A)                   # Perform the decomposition.
    stop = time()                               # Get the system clock time again. The difference is meaningful
    times.append([n,stop-start])                # Store the result for plotting.
# Since we expect the wall time to increase as n^p for fixed p, we plot on a log scale:
times = np.array(times)
plt.loglog(times[:,0],times[:,1],'-*')
plt.loglog(times[:,0],1e-6*times[:,0]**2,'-b') # Plot n^2 and n^3 for comparison.
plt.loglog(times[:,0],1e-9*times[:,0]**3,'-r')
plt.xlabel('matrix size')
plt.ylabel('time for LUP (s)')
plt.show()
