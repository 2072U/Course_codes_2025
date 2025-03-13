# Demonstration for forward and centered finite differences and their error. By L. van Veen, Ontario Tech U, 2025.
import numpy as np
import matplotlib.pyplot as plt

# A simple test function and its derivative:
def f(x):
    return x*np.sin(x)
def df(x):
    return np.sin(x)+x*np.cos(x)
# Equation of a straight line through points (x0,y0) and (x1,y1):
def lin(x,x0,x1,y0,y1):
    return y0 + (x-x0)*(y1-y0)/(x1-x0)
# Base point where we will approximate the derivative:
xb =0.2
# Fine grid for plotting:
nPlot = 1000
xs = np.linspace(0,1,nPlot)
ys = f(xs)
# Initial step size for forward differncing:
h = 0.5
# The left point is always the same:
x0 = xb
y0 = f(xb)
# Exact derivative for comparison:
exact = df(xb)
# Loop over step sizes and record the error for each of them:
errs = []
while h > 1e-13:
    # Second point on the graph:
    x1 = xb + h
    y1 = f(x1)
    # Compute the linear approximation and plot it along with the graph of f:
    ylin = lin(xs,x0,x1,y0,y1)
    plt.plot(xs,ys,'-')
    plt.plot(xs,ylin,'-r')
    plt.plot([x0,x0],[0,y0],'--k')
    plt.plot([x1,x1],[0,y1],'--k')
    plt.plot([0,1],[0,0],'-k')
    plt.title('h='+str(h))
    plt.show()
    # Estimate the derivative with finite differences and compute the error:
    fd = (y1 - y0) / (x1 - x0)
    errs.append([h,np.abs(exact - fd)])
    h /= 10.0
# Plot on a logarithmic scale because we expect the error to decrease as h^p for fixed p:
errs = np.array(errs)
plt.loglog(errs[:,0],errs[:,1],'-k',label='error of forward FD')
plt.loglog(errs[:,0],errs[:,0],'-r',label='h')
plt.xlabel('h')
plt.ylabel('FD error')
plt.legend()
plt.show()
# Note the V shape. To the right, the error is decreasing because the interpolation error decreases.
# To the left, the error is increasing because of cancellation of significant digits.

# Try again, but with centered differences:
h = 0.2
errs = []
while h > 1e-13:
    x0 = xb -h
    y0 = f(x0)
    x1 = xb + h
    y1 = f(x1)
    ylin = lin(xs,x0,x1,y0,y1)
    plt.plot(xs,ys,'-')
    plt.plot(xs,ylin,'-r')
    plt.plot([x0,x0],[0,y0],'--k')
    plt.plot([x1,x1],[0,y1],'--k')
    plt.plot([0,1],[0,0],'-k')
    plt.title('h='+str(h))
    plt.show()
    fd = (y1 - y0) / (x1 - x0)
    errs.append([h,np.abs(exact - fd)])
    h /= 10.0

errs = np.array(errs)
plt.loglog(errs[:,0],errs[:,1],'-k',label='error of centered FD')
plt.loglog(errs[:,0],errs[:,0]**2,'-r',label='h')
plt.xlabel('h')
plt.ylabel('FD error')
plt.legend()
plt.show()
# Now the error decreases as h^2 instead of h! The cancellation error is still there, of course.
