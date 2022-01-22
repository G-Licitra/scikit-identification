# In this example, we fit a nonlinear model to measurements
# Model: Chemical Reactor
#      xdot1 = u1 - k1*x1
#      xdot2 = u1*u2/x1 - u1*x2/x1 - k2*x2
# x = [x1;x2] | u = [u1;u2] | p = [ka;kb]    
# This example uses more advanced constructs than the vdp* examples:
# Since the number of control intervals is potentially very large here,
# we use memory-efficient Map and MapAccum, in combination with
# codegeneration.
# We will be working with a 2-norm objective:
# || y_measured - y_simulated ||_2^2
# This form is well-suited for the Gauss-Newton Hessian approximation.
# Author:      Giovanni Licitra
# Data:        22-01-2022


import casadi as ca
import numpy as np
import seaborn as sns
from numpy.matlib import repmat
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

# SETTINGS ===============================================================
N  = 100                           # Number of samples
fs = 50                            # Sampling frequency [hz]
t  = np.linspace(0,(N-1)*(1/fs),N) # time array

N_steps_per_sample = 4   
dt = 1/fs/N_steps_per_sample       # integration step for ode
nx, nu, ntheta = 2, 2, 2               # number states, input, and parameter

# Model parameters =======================================================
theta_truth = [0.1, 0.5];   # True parameters
theta_guess = [0.3, 0.4];   # guess for NLP
#scale       = [1e-1;1e-1]; % scaling factor

# Model generation via casADi/OPTistack ==================================
x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 2)
p = ca.MX.sym("p", 2)

# define uknow parametes as design variable
ka = p[0]
kb = p[1]

# xdot = f(x,u,p) <==> rhs = f(x,u,p)
rhs = ca.vertcat(u[0] - ka*x[0], 
                 u[0]*u[1]/x[0] - u[0] * x[1]/x[0]  -kb*x[1]) # xdot = f(x,u,p)
       
# Form an ode function
ode = ca.Function('ode', [x, u, p], [rhs])

ode.print_dimensions()

# numerical evaluation ===================================================
x_test     = [0.1,-0.1]
u_test     = [0.2,-0.1]
theta_test = [0.1,0.5]
f_out      = ode(x_test,u_test,theta_test)
print(f_out)

# build integrator: RK4 ==================================================
k1 = ode(x          , u, p)
k2 = ode(x+dt/2.0*k1, u, p)
k3 = ode(x+dt/2.0*k2, u, p)
k4 = ode(x+dt*k3    , u, p)
xf = x + dt/6.0*(k1+2*k2+2*k3+k4)

# Create a function that simulates one step propagation in a sample
one_step = ca.Function('one_step',[x, u, p], [xf])

X = x
for i in range(N_steps_per_sample):
    X = one_step(X, u, p)

# Create a function that simulates all step propagation on a sample
one_sample = ca.Function('one_sample', [x, u, p], [X])

# speedup trick: expand into scalar operations
one_sample = one_sample.expand()

# Compute Forward Simulation =============================================
# choose number of simulation step
all_samples = one_sample.mapaccum('all_samples', N)

# Choose an excitation signal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% experiment 1
u_data1  = 1 * np.random.randn(N,1)  #2 * np.random.randn(N,1) #2*chirp(t,1,10,5,'logarithmic') # Choose signal excitation [chirp]
u_data2  = 2 * np.random.randn(N,1)          # Choose signal excitation [random noise]
Udata    = np.concatenate((u_data1, u_data2), axis=1)              
x0       = np.array([1,-1], ndmin = nx)                          # Initial Condition x0 = [0;0]; [nx = 2]

# perform forward simulation
X_truth = all_samples(x0, np.transpose(Udata), ca.repmat(theta_truth, 1, N))
X_truth = np.concatenate((x0.T, X_truth) ,axis=1) # add the initial condition
X_truth = np.delete(X_truth, N, axis=1) # remove the last sample to keep 500 

# Add gaussian noise to states
ny = 0.04*np.random.randn(nx, N)                       
# sum noise measurements to the state
y_data = X_truth + ny 

df_x = pd.DataFrame(X_truth.T, t, ["x1", "x2"])
df_y = pd.DataFrame(y_data.T, t, ["y1", "y2"])

fig, ax = plt.subplots()
sns.scatterplot(data=df_y, ax=ax)
sns.lineplot(data=df_x, ax=ax)

# Set Identification Algorithm =========================================== 
print('Multiple shooting Approach')


# All states become decision variables
X = ca.MX.sym("X", nx, N)
Xn = one_sample.map(N, 'openmp')(X, Udata.T, ca.repmat(p, 1, N))

# continuity condition
gaps = Xn[:,:-1]-X[:,1:]
e = y_data-Xn
V = ca.veccat(p, X)
nlp = {'x':V, 'f':0.5*ca.dot(e,e), 'g': ca.vec(gaps)}

# Multipleshooting allows for careful initialization
#yd = np.diff(y_data, axis=0)*fs
#X_guess = ca.horzcat(y_data , vertcat(yd,yd[-1])).T;
#x0 = veccat(param_guess,X_guess)
x0 = ca.vertcat(0.3, 0.4, y_data[0, :], y_data[1, :])
solver = ca.nlpsol("solver","ipopt", nlp)
sol = solver(x0=x0,lbg=0,ubg=0)

print("solution found")
print(f'theta_est = {sol["x"][:ntheta]}')

#print(sol["x"][:4]*scale)
#assert(norm_inf(sol["x"][:4]*scale-param_truth)<1e-8)

X_est = np.array(sol["x"][ntheta:]).reshape((N,nx))


df_x = pd.DataFrame(X_truth.T, t, ["x1", "x2"])
df_y = pd.DataFrame(y_data.T, t, ["y1", "y2"])
df_est = pd.DataFrame(X_est, t, ["x_est1", "x_est2"])

fig, ax = plt.subplots()
sns.scatterplot(data=df_y, ax=ax)
sns.lineplot(data=df_x, ax=ax)
sns.lineplot(data=df_est, ax=ax)





X_est = full(all_samples(x0, Udata, repmat(theta_est,1,N)));
%% Plot residuals [only time information] =================================
figure;
title('data fitting');
plot(t,full(y_data)  ,'LineWidth',2,'Color','b');hold on;grid on;
plot(t,full(X_truth)','LineWidth',2.5,'Color','r');
plot(t,X_est         ,'LineWidth',2,'Color','g');
legend('y_{1}','y_{2}','x true_{1}','x true_{2}','x est_{1}','x est_{2}');
xlabel('time [s]');ylabel('y_measurement vs y_simulated');

% Print some information ==================================================
disp('');
disp('True parameters')
str = '%s* = %f \n';Cdisp = {'ka','kb';theta_truth(1),theta_truth(2)};
disp(sprintf(str,Cdisp{:}));

disp('Estimated parameters')
str = '%s* = %f \n';Cdisp = {'ka','kb';theta_est(1),theta_est(2)};
disp(sprintf(str,Cdisp{:}));

% Inspect Jacobian sparsity
Jacobian = jacobian(g, w);

% Inspect Hessian of the Lagrangian sparsity
Lambda     = MX.sym('lam', g.sparsity());
Lagrancian = w_sol.f + dot(Lambda, g);
Hessian    = hessian(Lagrancian, w);

figure;
subplot(1,2,1);title('Jacobian sparsity');hold on;
spy(sparse(DM.ones(Jacobian.sparsity())))
subplot(1,2,2);title('Hessian sparsity');hold on;
spy(sparse(DM.ones(Hessian.sparsity())))