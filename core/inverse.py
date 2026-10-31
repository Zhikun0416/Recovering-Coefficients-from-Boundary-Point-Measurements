"""Bounded least squares and measurement sensitivities."""

import numpy as np
from scipy.optimize import least_squares

from .backend import parameters


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def recover(problem, inputs, observed, truth=None, initial=None):
    """Fit scalar point observations with the discrete measurement map."""
    _, default_initial, bounds, _ = parameters(problem.model)
    initial = default_initial if initial is None else np.asarray(initial)
    use_adjoint = problem.model != "radial"
    cache = {"parameters": None}

    def evaluate(p):
        if cache["parameters"] is None or not np.array_equal(cache["parameters"], p):
            measured = problem.measure(p, inputs, gradient=use_adjoint)
            if use_adjoint:
                cache["values"], cache["jacobian"] = measured
            else:
                cache["values"] = measured
            cache["parameters"] = p.copy()
        return cache["values"]

    if use_adjoint:
        def jacobian(p):
            evaluate(p)
            return cache["jacobian"]
        derivative = {"jac": jacobian}
    else:
        derivative = {"jac": "2-point", "diff_step": 1e-4}
    optimum = least_squares(
        lambda p: evaluate(p) - observed,
        initial,
        bounds=bounds,
        x_scale="jac",
        max_nfev=200,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-12,
        **derivative,
    )
    if not optimum.success:
        raise RuntimeError(f"{problem.model} inversion failed: {optimum.message}")
    result = {"recovered": optimum.x, "fit_rms": rms(evaluate(optimum.x) - observed)}
    if truth is not None:
        error = np.abs((optimum.x - truth) / truth)
        result.update(
            truth=np.asarray(truth),
            relative_error=error,
            parameter_relative_RMS=rms(error),
            maximum_parameter_relative_error=float(error.max()),
        )
    return result


def sensitivity(problem, inputs, truth):
    """Singular values of the unscaled observation Jacobian."""
    if problem.model == "radial":
        columns = []
        for direction in np.eye(len(truth)) * 5e-4:
            columns.append(
                (problem.measure(truth + direction, inputs)
                 - problem.measure(truth - direction, inputs)) / 1e-3
            )
        jacobian = np.column_stack(columns)
    else:
        _, jacobian = problem.measure(truth, inputs, gradient=True)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if np.linalg.matrix_rank(jacobian) != len(truth):
        raise RuntimeError("The measurement Jacobian is not full rank.")
    return {
        "smallest_singular_value": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }
