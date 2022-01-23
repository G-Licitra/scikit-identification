import casadi as ca
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


############ Create RK4 ##########
def RK4integrator(ode, N_steps_per_sample=10, fs=601.1):

    dt = 1/fs/N_steps_per_sample

    # Build an integrator for this system: Runge Kutta 4 integrator
    k1 = ode(x,u, theta)
    k2 = ode(x+dt/2.0*k1, u, theta)
    k3 = ode(x+dt/2.0*k2, u, theta)
    k4 = ode(x+dt*k3, u, theta)
    states_final = x+dt/6.0*(k1+2*k2+2*k3+k4)

    # Create a function that simulates one step propagation in a sample
    one_step = ca.Function('one_step',[x, u, theta],[states_final])

    X = x
    for i in range(N_steps_per_sample):
        X = one_step(X, u, theta)

    # Create a function that simulates all step propagation on a sample
    one_sample = ca.Function('one_sample',[x, u, theta], [X])

    ############ Simulating the system ##########
    all_samples = one_sample.mapaccum("all_samples", N)

    return one_sample, all_samples


############ Create a Gauss-Newton solver ##########
def gauss_newton(e,nlp,V):
    # Use just-in-time compilation to speed up the evaluation
    if ca.Importer.has_plugin('clang'):
        print('just-in-time compilation with compiler= "clang" ')
        with_jit = True
        compiler = 'clang'
    elif ca.Importer.has_plugin('shell'):
        print('just-in-time compilation with compiler= "shell" ')
        with_jit = True
        compiler = 'shell'
    else:
        print("WARNING; running without jit. This may result in very slow evaluation times")
        with_jit = False
        compiler = ''

    J = ca.jacobian(e,V)
    H = ca.triu(ca.mtimes(J.T, J))
    sigma = ca.MX.sym("sigma")
    hessLag = ca.Function('nlp_hess_l',
                          {'x':V,'lam_f':sigma, 'hess_gamma_x_x':sigma*H},
                          ['x','p','lam_f','lam_g'], ['hess_gamma_x_x'],
                          dict(jit=with_jit, compiler=compiler))
    
    return ca.nlpsol("solver","ipopt", nlp, dict(hess_lag=hessLag, jit=with_jit, compiler=compiler))


# In this example, we fit a nonlinear model to measurements
#
# This example uses more advanced constructs than the vdp* examples:
# Since the number of control intervals is potentially very large here,
# we use memory-efficient Map and MapAccum, in combination with
# codegeneration.
#
# We will be working with a 2-norm objective:
# || y_measured - y_simulated ||_2^2
#
# This form is well-suited for the Gauss-Newton Hessian approximation.

############ SETTINGS #####################
N = 10000  # Number of samples
fs = 610.1 # Sampling frequency [hz]
t  = np.linspace(0,(N-1)*(1/fs),N) # time array

nx, nu, ntheta = 2, 1, 4               # number states, input, and parameter

param_truth = ca.DM([5.625e-6, 2.3e-4, 1, 4.69]) # proper scaling 1e-6,1e-4,1,1
param_guess = ca.DM([5,2,1,5])
scale = ca.vertcat(1e-6,1e-4,1,1)

############ MODELING #####################
x = ca.MX.sym("x", nx)

y, dy = x[0], x[1]
u = ca.MX.sym("u", nu)

theta = ca.MX.sym("theta", ntheta)
M, c, k, k_NL = theta[0], theta[1], theta[2], theta[3] 

rhs = ca.vertcat(dy , (u-k_NL*y**3-k*y-c*dy)/M)

# Form an ode function
ode = ca.Function('ode',[x,u,theta],[rhs])

# create integrator
one_sample, all_samples = RK4integrator(ode, N_steps_per_sample=10, fs=601.1)

############ GENERATE GROUND TRUTH #####################

# Choose an excitation signal
np.random.seed(0)
u_data = ca.DM(0.1*np.random.random(N))
x0 = ca.DM([0,0])
X_measured = all_samples(x0, u_data, ca.repmat(param_truth,1,N))
y_data = X_measured.T 
# You may add some noise here
# When noise is absent, the fit will be perfect.
y_data+= 0.001*np.random.random((N, nx))


############ Identifying the simulated system: single shooting strategy ##########

# Note, it is in general a good idea to scale your decision variables such
# that they are in the order of ~0.1..100
X_symbolic = all_samples(x0, u_data, ca.repmat(theta*scale,1,N))
e = y_data-X_symbolic.T
nlp = {'x':theta, 'f':0.5*ca.dot(e,e)}
solver = gauss_newton(e,nlp, theta)
sol = solver(x0=param_guess)

theta_est = sol['x']*scale
print(f"theta est = {theta_est}")
print(f"theta truth = {param_truth}")

X_est = all_samples(x0, u_data, ca.repmat(theta_est,1,N))

df_x = pd.DataFrame(np.array(X_measured).T, t, ["x1", "x2"])
df_y = pd.DataFrame(np.array(y_data), t, ["y1", "y2"])
df_est = pd.DataFrame(np.array(X_est).T, t, ["x_est1", "x_est2"])

fig, ax = plt.subplots()
sns.scatterplot(data=df_y.loc[:,'y2'], ax=ax)
sns.lineplot(data=df_x['x2'], ax=ax)
sns.lineplot(data=df_est['x_est2'], ax=ax)
plt.xlim((0,0.2))

############ Identifying the simulated system: multiple shooting strategy ##########
# TODO
# All states become decision variables
X = ca.MX.sym("X", nx, N)
Xn = one_sample.map(N, 'openmp')(X, u_data.T, ca.repmat(theta*scale,1,N))
gaps = Xn[:,:-1]-X[:,1:]
e = y_data-Xn.T
V = ca.veccat(theta, X)
nlp = {'x':V, 'f':0.5*ca.dot(e,e),'g': ca.vec(gaps)}

# Multipleshooting allows for careful initialization
yd = np.diff(y_data,axis=0)*fs
X_guess = ca.horzcat(y_data[:,0] , ca.vertcat(yd,yd[-1])).T

x0 = ca.veccat(param_guess,X_guess)

solver = gauss_newton(e,nlp, V)

sol = solver(x0=x0,lbg=0,ubg=0)

print(sol["x"][:4]*scale)

assert(norm_inf(sol["x"][:4]*scale-param_truth)<1e-8)
