import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters

############ SETTINGS #####################
N = 10000  # Number of samples
fs = 610.1  # Sampling frequency [hz]

param_truth = ca.DM([5.625e-6, 2.3e-4, 1, 4.69])
param_guess = ca.DM([5, 2, 1, 5])
scale = ca.vertcat(1e-6, 1e-4, 1, 1)

############ MODELING #####################

(state, input, param) = generate_model_parameters(nstate=2, ninput=1, nparam=4)


# rhs = [-alpha * x[0] + p[0] * x[0] * x[1], gamma * x[1] - p[1] * x[0] * x[1
#                                                                           ]]


y, dy = state[0], state[1]

u = input[0]

M, c, k, k_NL = param[0], param[1], param[2], param[3]

# M = ca.MX.sym("x")
# c = ca.MX.sym("c")
# k = ca.MX.sym("k")
# k_NL = ca.MX.sym("k_NL")

# params = ca.vertcat(M, c, k, k_NL)

rhs = [dy, (u - k_NL * y**3 - k * y - c * dy) / M]


model = DynamicModel(
    states=state,
    inputs=input,
    param=param,
    model_dynamics=rhs,
)

model.print_summary()

rhs_num, y_num = model.evaluate(
    state_num=[1.0, 1.0], input_num=0, param_num=param_truth
)
print(f"rhs = {rhs_num}, \ny = {y_num}")


# Form an ode function
# ode = ca.Function("ode", [states, controls, params], [rhs])

############ Creating a simulator ##########
N_steps_per_sample = 10
dt = 1 / fs / N_steps_per_sample

# Build an integrator for this system: Runge Kutta 4 integrator
k1, _ = model.model_function(state, input, param)
k2, _ = model.model_function(state + dt / 2.0 * k1, input, param)
k3, _ = model.model_function(state + dt / 2.0 * k2, input, param)
k4, _ = model.model_function(state + dt * k3, input, param)

states_final = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# Create a function that simulates one step propagation in a sample
one_step = ca.Function("one_step", [state, input, param], [states_final])

X = state

for i in range(N_steps_per_sample):
    X = one_step(X, input, param)

# Create a function that simulates all step propagation on a sample
one_sample = ca.Function("one_sample", [state, input, param], [X])

############ Simulating the system ##########
all_samples = one_sample.mapaccum("all_samples", N)

# Choose an excitation signal
np.random.seed(0)
u_data = ca.DM(0.1 * np.random.random(N))

x0 = ca.DM([0, 0])
X_measured = all_samples(x0, u_data, ca.repmat(param_truth, 1, N))

y_data = X_measured[0, :].T

# You may add some noise here
# y_data+= 0.001*numpy.random.random(N)
# When noise is absent, the fit will be perfect.

# Use just-in-time compilation to speed up the evaluation
if ca.Importer.has_plugin("clang"):
    with_jit = True
    compiler = "clang"
elif ca.Importer.has_plugin("shell"):
    with_jit = True
    compiler = "shell"
else:
    print("WARNING; running without jit. This may result in very slow evaluation times")
    with_jit = False
    compiler = ""

############ Create a Gauss-Newton solver ##########
def gauss_newton(e, nlp, V):
    J = ca.jacobian(e, V)
    H = ca.triu(ca.mtimes(J.T, J))
    sigma = ca.MX.sym("sigma")
    hessLag = ca.Function(
        "nlp_hess_l",
        {"x": V, "lam_f": sigma, "hess_gamma_x_x": sigma * H},
        ["x", "p", "lam_f", "lam_g"],
        ["hess_gamma_x_x"],
        dict(jit=with_jit, compiler=compiler),
    )
    return ca.nlpsol(
        "solver", "ipopt", nlp, dict(hess_lag=hessLag, jit=with_jit, compiler=compiler)
    )


############ Identifying the simulated system: single shooting strategy ##########

# Note, it is in general a good idea to scale your decision variables such
# that they are in the order of ~0.1..100
X_symbolic = all_samples(x0, u_data, ca.repmat(param * scale, 1, N))

e = y_data - X_symbolic[0, :].T
nlp = {"x": param, "f": 0.5 * ca.dot(e, e)}

solver = gauss_newton(e, nlp, param)

sol = solver(x0=param_guess)

print(sol["x"][:4] * scale)
print(param_truth)

assert ca.norm_inf(sol["x"] * scale - param_truth) < 1e-8

############ Identifying the simulated system: multiple shooting strategy ##########

# All states become decision variables
X = ca.MX.sym("X", 2, N)

Xn = one_sample.map(N, "openmp")(X, u_data.T, ca.repmat(param * scale, 1, N))

gaps = Xn[:, :-1] - X[:, 1:]

e = y_data - Xn[0, :].T

V = ca.veccat(param, X)

nlp = {"x": V, "f": 0.5 * ca.dot(e, e), "g": ca.vec(gaps)}

# Multipleshooting allows for careful initialization
yd = np.diff(y_data, axis=0) * fs
X_guess = ca.horzcat(y_data, ca.vertcat(yd, yd[-1])).T

x0 = ca.veccat(param_guess, X_guess)

solver = gauss_newton(e, nlp, V)

sol = solver(x0=x0, lbg=0, ubg=0)

print(sol["x"][:4] * scale)
print(param_truth)


assert ca.norm_inf(sol["x"][:4] * scale - param_truth) < 1e-8
