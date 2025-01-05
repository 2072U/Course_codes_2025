# Script to reproduce the PageRank example from lecture 1/Wikipedia (https://en.wikipedia.org/wiki/PageRank).
# MATH/CSCI2072U, Winter 2025. By L. van Veen, Ontario Tech U.
import numpy as np              # The usual imports..
import matplotlib.pyplot as plt

# Set parameters:
# Numerical parameters:
nIter = 100      # Maximal number of iterations.
tol = 1e-4       # Relative error tolerance.
# Algorithm parameters:
d = 0.85         # Damping factor of the PR algorithm (value taken from Wikipedia).
# Input parameters:
n = 11           # Number of nodes in the network.
# Allocate a n X n array of zeros and inset ones for connections i <- j in entry (i,j):
A = np.zeros((n,n),dtype=float)
A[0,3] = 1
for j in range(2,9):
    A[1,j] = 1
A[2,1] = 1
A[3,4] = 1
for j in range(5,11):
    A[4,j] = 1
A[5,4] = 1

print(A)
# Now there is a one for each link. PR requires each column to sum to 1 (the PR flowing out is equally distributed over outgoing links),
# but some columns may have only zeros:
for j in range(n):
    norm = np.sum(A[:,j])
    if norm > 0:
        A[:,j] /= norm

# The simplest, but not the most efficient, method to find the PR is the "power method".
# Of course, you can use LU decomposition, which is part of this course, but that method is not efficient enough for large numbers of web pages.
# The power method is closer to what Google actually uses.
# Initialize a n-vector (any non-zero vector will do):
v = np.ones((n,1)) / float(n)
# Now iterate the map v -> (1-d) I + d * A  v, the core of the PR algorithm...

for k in range(nIter):
    w = np.copy(v)                                    # Make extra sure that v and w are different variables, not different names for the same pointer.
    v = (1.0-d) + d * A @ v                           # The PR step.
    print(v)                                          # Print the intermediary result.
    err = np.linalg.norm(v-w,2) / np.linalg.norm(w,2) # The error is the relative difference between iterates.
    print('Iteration %d, err=%e.' % (k,err))
    if err < tol:                                     # If the error is smaller than the tolerance, exit the loop.
        break
v = 100 * v / np.sum(v)                               # Scale the result to percentages.
print(v)                                              # The entry of this vector are the PR values, scaled to percentages like on the Wiki page.


