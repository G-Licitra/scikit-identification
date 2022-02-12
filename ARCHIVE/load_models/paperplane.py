import numpy as np
import casadi as ca


# model:
# input:  u = angle of attack [rad]
# states: x = [px;pz;vx;vz] with p = position [m] v = velocity [m/s]
# nx = 4 | nu = 1

# define parameter p, input u and differential states x
p = {'m': 2,   # mass [kg]
     'AR': 10,    # Aspect ratio
     'rho': 1.2,  # air density [kg/m3]
     'g': 9.81,   # gravity [m/s^2]
     'Sref': 0.5} # area wing [m^2]

u = ca.MX.sym("u", 1)
x = ca.MX.sym("x", 4)

px = x[0]  # position x
pz = x[1]  # position z = -h (altitude)
vx = x[2]  # velocity x [m/s]
vz = x[3]  # velocity z [m/s]
alpha = u # angle of attack [rad]

# define model dynamics
CL = 2*np.pi*alpha*(10/12)         # Lift coefficient
CD = 0.01+CL**2 / (p['AR']*np.pi)  # Draft coefficient
V  = ca.sqrt(vx**2 + vz**2)        # Velocity

eL = 1/V * ca.vertcat(vz,-vx)      # Lift vector
eD = 1/V * ca.vertcat(-vx, -vz)    # Drag vector

Flift = 0.5 * p['rho'] * V**2 * CL * p['Sref'] * eL
Fdrag = 0.5 * p['rho'] * V**2 * CD * p['Sref'] * eD
Fgravity = ca.vertcat(0, p['m']* p['g'])
F = Flift + Fdrag + Fgravity;
Fx = F[0]
Fz = F[1]

rhs = ca.vertcat(vx, vz, Fx/p['m'], Fz/p['m'])

# Construction Symbolic Ode
ode = ca.Function('ode', [x, u], [rhs])

# numerical evaluation
x_test     = np.array([0, 100, 5, 5])
u_test     = np.deg2rad(5)
f_out      = ode(x_test,u_test)


#p = [m;AR;rho;g;Sref];
#x0 = [0;100;5;5];           % [px0;pz0;vx0;vz0] i.c.

# First, define the time-dependent parameters
#T = 10;
#ts = 0.01;
#t = [0:ts:T-ts]';                               % define range
#ut  = t;                                        % time varing parameter
#u = 0*pi/180 + 0*sin(2*pi*0.5*t);               % angle of attack [rad]
#u(1) = deg2rad(5);                              % 5 deg i.c.


fs = 100 # sample rate
f  = 1   # the frequency of the signal
Toss = 4   # observation time
t    = np.linspace(0, Toss, Toss*fs, endpoint=False)
# generate a signal with 2 harmonics
data_time = pd.DataFrame(data={'x_sine': np.sin(2*np.pi*f* t) + 0.5*np.cos(2*np.pi*(10*f)* t)}, index=t)


N  = 50;                           # number of samples
fs = 50;                           # sampling frequency [hz]
t0, tf = 0, N/fs                   # initial and final time [s]
t  = np.linspace(0,(N-1)*(1/fs),N) # time array


N_steps_per_sample = 4;
dt = 1/fs/N_steps_per_sample;      # integration step for ode

%% build integrator: RK4 ==================================================
k1 = ode(states          ,controls,theta);
k2 = ode(states+dt/2.0*k1,controls,theta);
k3 = ode(states+dt/2.0*k2,controls,theta);
k4 = ode(states+dt*k3    ,controls,theta);
xf = states + dt/6.0*(k1+2*k2+2*k3+k4);
% Create a function that simulates one step propagation in a sample
one_step = Function('one_step',{states, controls, theta},{xf});

X = states;
for i=1:N_steps_per_sample
    X = one_step(X, controls, theta);
end

% Create a function that simulates all step propagation on a sample
one_sample = Function('one_sample',{states, controls, theta}, {X});
% speedup trick: expand into scalar operations
one_sample = one_sample.expand();
