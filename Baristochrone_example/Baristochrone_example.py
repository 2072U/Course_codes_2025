# Simple demonstration of the "brachistochrone problem": how to shape a slide so that a child sliding without friction from
# (x,y)=(0,H) to (L,0) so that the time it takes is minimal. Contains a function to (approximately) solve the equation of motion
# and a script to find the optimal values of the coefficients a in y(x)=H - (H+P(L)) x/L + P(x) with P(x)=a_0 x^2 + ... + a_N x^{N+2} 
# using a rudimentary form of gradient descent. By L. van Veen, Ontario Tech U, 2025.
import numpy as np
import matplotlib.pyplot as plt

# Defines the shape of the slide as y=F(x). In: x (horizontal position), a (array of polynomial coefficients), height and length of the
# slide H and L.
def P(x,a):
    numCoeffs = np.size(a)
    pol = 0.0
    for i in range(numCoeffs):
        pol += a[i] * x**(i+2)
    return pol
def F(x,a,H,L):
    height = H - (H+P(L,a)) * x/L + P(x,a)
    return height

# First derivative of the graph of y over x, which appears in the equation of motion because it determines the component
# of the force of gravity along the slide. Input and output as in the function F.
def DP(x,a):
    numCoeffs = np.size(a)
    pol = 0.0
    for i in range(numCoeffs):
        pol += float(i+2) * a[i] * x**(i+1)
    return pol
def DF(x,a,H,L):
    slope = - (H+P(L,a))/L + DP(x,a)
    return slope

# Second derivative of the graph of y over x, which appears in the equation of motion as a geometric term.
# Input and output as in the function F except that H and L are not needed.
def DDP(x,a):
    numCoeffs = np.size(a)
    pol = 2.0 * a[0]
    for i in range(1,numCoeffs):
        pol += float((i+2)*(i+1)) * a[i] * x**i
    return pol
def DDF(x,a):
    curv = DDP(x,a)
    return curv

# Newton's equations of motion: dx/dt=v; dv/dt = force. Input: horizontal position and velocity x and v, array of coefficients a,
# height and length of the slide.
def RHS(x,v,a,H,L):
    slope = DF(x,a,H,L)
    curve = DDF(x,a)
    dxdt = v
    dvdt = - (g + v**2 * curve) * slope / (1.0 + slope**2) 
    return dxdt, dvdt

# Function to use Euler's method to find an approximate solution to the equations of motion. Input: maximal integration time,
# time step size, initial position and velocity x0 and v0, goal value of x, list of coefficients, height and length of the slide.
def Solve(tMax,dt,x0,v0,x1,a,H,L):
    t = 0
    x = x0
    v = v0
    hist = np.zeros((int(np.ceil(tMax/dt))+1,2)) # Store the intermediate steps for plotting.
    hist[0,:] = [t,x0]
    step = 0
    reach = 1                                    # Set to True by default, set to false if the goal is not reached by tMax.
    while x < x1:
        dxdt, dvdt = RHS(x,v,a,H,L)              # Euler step.
        x += dt * dxdt
        v += dt * dvdt
        t += dt
        step += 1
        if t > tMax:                             # If x does not reach x1 by time tMax, exit.
            print("Goal not reached before t=tMax=%f." % (tMax))
            reach = 0
            break
        hist[step,:] = [t,x]
    if reach == True:                            # If the goal is reached, estimate the transit time by linear interpolation.
        tTot = hist[step-1,0] - (x1-hist[step-1,1]) * (hist[step,0]-hist[step-1,0])/(hist[step,1]-hist[step-1,1])
    else:
        tTot = 0.0
    return tTot,hist

# Set parameters and auxiliary variables:
# .. geometry and physics ..
g = 9.81                                # Acelleration of gravity in m/s^2.
H = 2.0                                 # Height of the slide in metres.
L = 4.0                                 # Lenght of the slide in metres.
# .. initial state and goal ..
x0 = 0.0                                # Start at (x,y)=(0,F(0))=(0,H).
v0 = 0.0                                # No initial velocity.
x1 = L                                  # Target horizontal displacement.
# .. initial shape of the slide ..
nCoeff = 3                              # Number of polynomial coefficients.
a0 = np.zeros((nCoeff,))                # Initial values of the coefficients. All zeros gives a straight ramp.
a0[0] = -0.1                            # Just for fun start with a slightly "n-shaped" slide (parabola with a maximum).
# .. parameters of the numerical algorithms ..
tMax = 10                               # Maximal integration time.
dt = 0.00001                            # Time step size - keep small or gradient descent may fail.
da = 0.001                              # Step size of the finite-difference approximation of the gradient.
stepSize = 0.002                        # Step size of the gradient descent.
nStep = 1000                            # Maximal number of gradient descent steps.
# Define auxiliary variables:
xs = np.linspace(0,L,100)               # For plotting the slide.
res = np.zeros((nStep,2))               # Pre-allocate an array for the slide time after each descent step.
# The actual optimization loop!
a = a0                                  # Fix the array of coefficients to the initial value.
for step in range(nStep):
    tTot0, hist = Solve(tMax,dt,x0,v0,x1,a,H,L)             # Find the slide time for the current design.
    res[step,:] = [step,tTot0]                              # Store the result for plotting.
    if step > 0:                                            # If we have taken at least one step, track the progress.
        gain = (tMem-tTot0)/tMem                            # This is the relative decrease of the slide time.
        print('Step %d T=%f gain=%f.' % (step,tTot0,gain))
        if gain < 0.1 * np.linalg.norm(dTda,2) *da / tMem:  # If there is no more significant progress, stop.
            break
    tMem = tTot0
# Enable the block below for plotting intermediate results...
    ys = F(xs,a,H,L)
    plt.plot(xs,ys,'-')
    plt.title('Time taken is %f.' % (tTot0))
    plt.show()
    xoft = np.array([point for point in hist.tolist() if point[0] > dt and point[1] < L]) # Delete the extra zeros fr plotting.
    plt.plot(xoft[:,0],xoft[:,1])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()
    
    dTda = np.zeros((nCoeff,))                              # Pere-allocate an array for the gradient of the slide time.    
    for j in range(nCoeff):                                 # Estimate the sensitivity to changes to each of the coefficients
        b = np.copy(a)                                      # by finite differencing.
        b[j] += da
        tTot1, hist = Solve(tMax,dt,x0,v0,x1,b,H,L)
        dTda[j] = (tTot1 - tTot0) / da
    dTda /= np.linalg.norm(dTda,2)                          # Normalize the gradient vector.
    print(step,a,stepSize,dTda,tTot0)
    a = a - stepSize * dTda                                 # Adjust the shape of the slide and try again.
plt.plot(res[0:step,0],res[0:step,1],'-*')
plt.show()
ys = F(xs,a,H,L)
plt.plot(xs,ys,'-b')
plt.title('The near-optimal shape:')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

