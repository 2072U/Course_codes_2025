# Brief script to demonstrate the use of the cubic spline function in SciPy.
# Reproduces the example in the slides of lecture 12. By L van Veen, MATH/CSCI2072U, Ontario Tech U, 2025.
import numpy as np
import scipy
import scipy.interpolate
import matplotlib.pyplot as plt

# Set knots:
N = 4
L = -4.0
R = 4.0
xs = np.linspace(L,R,N)
# Example computation from lecture 12. Splines approximating the arctan function.
def F(x):
    return np.arctan(x)
ys = F(xs)

# Note, that a "PPoly object" is returned. To evaluate it, we can use the method __call__
f = scipy.interpolate.CubicSpline(xs,ys,bc_type='natural',extrapolate=True)

xPlot = np.linspace(L-1.0,R+1.0,1000)
yPlot = f.__call__(xPlot)
yf = F(xPlot)
plt.plot(xPlot,yPlot,'-r')
plt.plot(xPlot,yf,'-k')
plt.plot(xs,ys,'.',markersize=20)
plt.title('%d knots' % (N))
plt.show()
