# A test for the least-squares code. We fit a second-order polynomial to data (t,z(t)) where t is time and z is the height
# of a falling object. Uncertainty of measurements is represented by a random term. The final result is the list of
# parameters (accelleration of gravity, g, initial velocity v0 and iniital height H) estimated from the least-squares fit
# that can be compared to the true value of the parameters used when generating the data.
# CSCI/MATH2072U, Ontario Tech U, 2025, by L. van Veen.
import numpy as np
from LS import LS                                 # Our least-squares function.
import matplotlib.pyplot as plt                   # For plotting.
from poly_intpol import *                         # Lagrange interpolation for comparison.

# This function generates surrogate data, with noise and uncertainty modelled by random numbers.
# Input: time step dt between measurements, noise level epsilon (floats).
# Output: arrays x (time instants) and y (height measurements), time at which the weight hits the ground and the highest point of hte orbit (floats)
def surrogate_data(dt,eps):                       # A mass falling vertically with initial velocity v0 from height H
    g = 9.81                                      # without air friction.
    v0 = 10.0
    H = 56.67
    print("Actual values of the coefficients: a[0]=H="+str(H)+", a[1]=v0="+str(v0)+" and a[2]=-g/2="+str(-g/2.0))
    t_hit = v0/g + np.sqrt(v0**2 +2.0 * g *H)/g   # This is when the mass hits the ground.
    top = H + v0**2/(2.0*g)                       # The highest point of the orbit.
    nd = np.floor(t_hit/dt)                       # Number of data when measuring with intervals dt.
    nd = nd.astype(int)
    x = np.zeros([nd,1])
    y = np.zeros([nd,1])
    for k in range(1,nd+1):                       # "Measured" data with noise of amplitude eps.
        error = eps * (np.random.rand()-0.5)      # Default range of rand is [0,1] so we subtract 0.5 t0 make the pertubation positive or negative with equal probability.
        x[k-1] = float(k) * dt
        y[k-1] = H + v0 * float(k) * dt - g * float(k**2) *dt**2 / 2.0 + error # Solution to Newton's equation of motion with added noise.
    return x,y,t_hit,top

# Set the parameters of the experiment:
dt = 0.6
eps = 10.0
x,y,t_hit,top = surrogate_data(dt,eps)            # Pretend we measure at times stored in x the heights stored in y.

order = 2                                         # Newton's equations tell us the height should depend on time through a second order polynomial.
P,a = LS(x,y,order)                               # Compute the quadratic least-squares fit.
print("Estimated values of the coefficients: a[0]=H="+str(a[0][0])+", a[1]=v0="+str(a[1][0])+" and a[2]=-g/2="+str(a[2][0]))

npo = 100                                         # Take npo points for plotting.
l = 0.0
r = t_hit
dx = (r-l)/float(npo)
xs = np.zeros([npo,1])                            # Pre-allocate arrays for plotting.
ys = np.zeros([npo,1])

for k in range(0,npo):                            # Fill those arrays.
    xs[k] = l+float(k)*dx
    ys[k] = P(xs[k])
# Try naive interpolation:
xp = np.linspace(0,t_hit,1000)
b, yp = poly_int(x,y,xp)
plt.plot(x,y,'*')
plt.plot(xp,yp,'-')
plt.ylim([0,1.2*top])
plt.xlim([0,1.2*t_hit])
plt.title('Lagrange interpolation does not make sense...')
plt.show()
# That doesn't look right.. you probably see the weight moving upward!

plt.plot(xs,ys,'-k',x,y,'*')
plt.xlim([0,1.2*t_hit])
plt.ylim([0,1.2*top])
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('... but a quadratic least-squares fit does!')
plt.show()

