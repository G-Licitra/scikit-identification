import casadi as ca
import numpy as np


def gauss_newton(e, nlp, V):
    r"""Gauss-Newton solver`

    Parameters
    ----------
    e : casadi
        cost function
    nlp: dict
        Steps forward within one integration step
    V : casadi
        optimization variables

    Returns
    -------
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
