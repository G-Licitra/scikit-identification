import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from skmid.integrator import RungeKutta4
from skmid.models import DynamicModel
from skmid.models import generate_model_parameters


def gauss_newton(e, nlp, V):
    r"""Gauss-Newton solver`

    Parameters
    ---------
    e : casadi
        cost function
    nlp: dict
        Steps forward within one integration step
    V : casadi
        optimization variables

    Returns
    ---------
    solver : casadi
        nlp solver

    """

    # Use just-in-time compilation to speed up the evaluation
    if ca.Importer.has_plugin("clang"):
        print('just-in-time compilation with compiler= "clang" ')
        with_jit = True
        compiler = "clang"
    elif ca.Importer.has_plugin("shell"):
        print('just-in-time compilation with compiler= "shell" ')
        with_jit = True
        compiler = "shell"
    else:
        print(
            "WARNING; running without jit. This may result in very slow evaluation times"
        )
        with_jit = False
        compiler = ""

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


class EquationErrorMethod:
    def __init__(self, *, model: DynamicModel, fs: int = 1):
        pass

    def fit(self, U, Y):
        pass  #

    def predict(self, U, Y):
        pass

    def summary(self):
        pass


if __name__ == "__main__":  # when run for testing only

    T = np.linspace(0, 10, 11)

    yN = np.array(
        [
            [
                1.0,
                0.9978287,
                2.366363,
                6.448709,
                5.225859,
                2.617129,
                1.324945,
                1.071534,
                1.058930,
                3.189685,
                6.790586,
            ],
            [
                1.0,
                2.249977,
                3.215969,
                1.787353,
                1.050747,
                0.2150848,
                0.109813,
                1.276422,
                2.493237,
                3.079619,
                1.665567,
            ],
        ]
    )

    param_truth = [0.693379029, 0.341128482]

    sigma_x1 = 0.1
    sigma_x2 = 0.2

    (x, _, p) = generate_model_parameters(nstate=2, ninput=0, nparam=2)

    alpha = 1.0
    gamma = 1.0

    rhs = [-alpha * x[0] + p[0] * x[0] * x[1], gamma * x[1] - p[1] * x[0] * x[1]]

    model = DynamicModel(
        states=x,
        param=p,
        model_dynamics=rhs,
    )

    model.print_summary()

    rhs_num, y_num = model.evaluate(state_num=[1.0, 1.0], param_num=param_truth)
    print(f"rhs = {rhs_num}, \ny = {y_num}")

    # The weightings for the measurements errors given to scikit-identification are calculated
    # from the standard deviations of the measurements, so that the least squares
    # estimator is the maximum likelihood estimator for the estimation problem.

    wv = np.zeros((2, yN.shape[1]))
    wv[0, :] = 1.0 / sigma_x1**2
    wv[1, :] = 1.0 / sigma_x2**2

    # invoke integrator
    rk4 = RungeKutta4(model=model, fs=1, n_steps_per_sample=1)

    _ = rk4.simulate(x0=[1.0, 1.0], param=p, N_steps=11)
    y_symbolic = rk4.state_sim_

    #    X_symbolic[0,:]
    # MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]', repmat(([1e-06, 0.0001, 1, 1]*vertcat(x, c, k, k_NL)), 10000)){0}[:20000:2])
    # MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [[0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]], repmat(([1e-06, 0.0001, 1, 1]*param)                , 10000)){0}'[:10000])

    # patch
    # X_symbolic = all_samples(x0, u_data, repmat(params*scale,1,N))

    e = yN - y_symbolic
    nlp = {"x": p, "f": 0.5 * ca.dot(e, e)}

    solver = gauss_newton(e, nlp, p)

    sol = solver(x0=[0, 0])

    p_est = sol["x"].full()

    print(p_est)
    print(param_truth)

    _ = rk4.simulate(x0=[1.0, 1.0], param=p_est, N_steps=11)
    x_sim = rk4.state_sim_

    plt.figure()

    plt.scatter(T, yN[0, :], color="b", label="$x_{1,meas}$")
    plt.scatter(T, yN[1, :], color="r", label="$x_{2,meas}$")

    plt.plot(x_sim.index, x_sim["x1"].values, color="b", label="$x_{1,sim}$")
    plt.plot(x_sim.index, x_sim["x2"].values, color="r", label="$x_{2,sim}$")

    plt.xlabel("$t$")
    plt.ylabel("$x_1, x_2$", rotation=0)
    plt.xlim(0.0, 10.0)

    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

    # param_truth =[5.625e-06, 0.00023, 1, 4.69]

    # # Settings
    # N = 10000  # Number of samples
    # fs = 610.1 # Sampling frequency [hz]

    # # model
    # param_truth = [5.625e-6,2.3e-4,1,4.69]
    # param_guess = [5,2,1,5]
    # scale = ca.vertcat(1e-6,1e-4,1,1)

    # (states, input, param) = generate_model_parameters(nstate=2, ninput=1, nparam=4)

    # y, dy  = states[0], states[1]
    # u  = input[0]
    # M, c, k, k_NL = param[0], param[1], param[2], param[3]

    # rhs = [dy ,
    #        (u - k_NL*y**3-k*y-c*dy)/M
    #        ]

    # output = [y]

    # model = DynamicModel(
    #     states=states,
    #     inputs=input,
    #     param = param, # <------- patch
    #     model_dynamics=rhs,
    #     output=output,
    #     state_name=["y", "dy"],
    #     input_name=["u"],
    #     output_name=["y"],
    #     param_name = ["M", "c", "k", "k_NL"]
    # )

    # model.print_summary()

    # rhs_num, y_num = model.evaluate(state_num=[1,0.1], input_num=1, param_num=param_truth)
    # print(f"rhs = {rhs_num}, \ny = {y_num}")

    # # invoke integrator
    # rk4 = RungeKutta4(model=model, fs=fs, n_steps_per_sample=10)

    # # Choose an excitation signal
    # np.random.seed(0)
    # u_data = 0.1*np.random.random(N)
    # x0 = ca.DM([0,0])

    # _ = rk4.simulate(x0=x0, input=u_data, param=param_truth)

    # y_data = rk4.output_sim_
    # y_data+= y_data.applymap(lambda x:x + 0*np.random.random()) #0.001*np.random.random(N+1)

    # _ = rk4.simulate(x0=x0, input=u_data, param=param) # param=param*scale
    # y_symbolic = rk4.state_sim_

    # #    X_symbolic[0,:]
    # #MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]', repmat(([1e-06, 0.0001, 1, 1]*vertcat(x, c, k, k_NL)), 10000)){0}[:20000:2])
    # #MX(one_sample_acc10_acc10_acc10_acc10(zeros(2x1), [[0.0548814, 0.0715189, 0.0602763, ..., 0.075843, 0.00237874, 0.0813575]], repmat(([1e-06, 0.0001, 1, 1]*param)                , 10000)){0}'[:10000])

    # # patch
    # y_symbolic = y_symbolic[:,0]
    # #X_symbolic = all_samples(x0, u_data, repmat(params*scale,1,N))

    # e = y_data.iloc[1:].values-y_symbolic
    # nlp = {'x':param, 'f':0.5*ca.dot(e,e)}

    # solver = gauss_newton(e,nlp, param)

    # sol = solver(x0=param_guess)

    # print(sol["x"]*scale)
    # # param_truth =[5.625e-06, 0.00023, 1, 4.69]

    # assert(ca.norm_inf(sol["x"]*scale-param_truth)<1e-8)

# -----------------------------------------------
