"""Evaluate independent inputs together with shared Poisson factorizations."""
import numpy as np


def measure(problem, parameters, inputs, gradient=False):
    solver = problem.disk
    if getattr(problem, "_batch_source", None) is not inputs:
        if problem.case == 2:
            f = np.column_stack([row["f"] for row in inputs])
            u0 = np.column_stack([row["u0"].ravel() for row in inputs])
            obs0 = np.array([row["obs0"] for row in inputs])
        else:
            f = np.column_stack([row[0] for row in inputs])
            lifted = [solver.lifting(*row) for row in inputs]
            u0 = np.column_stack([item[0].ravel() for item in lifted])
            obs0 = np.array([item[1] for item in lifted])
        problem._batch_source = inputs
        problem._batch_lifting = f, u0, obs0
    f, u0, obs0 = problem._batch_lifting
    if not hasattr(problem, "_batch_basis"):
        from .backend import coefficient_fields
        basis = [coefficient_fields(problem.model, unit, problem.X, problem.Y, problem.case)
                 for unit in np.eye(len(parameters))]
        qb = np.column_stack([item[0].ravel() for item in basis])
        ab = np.column_stack([item[1].ravel() for item in basis])
        bb = np.array([item[2] for item in basis])
        if problem.case == 2:
            boundary_basis = solver.q_boundary, solver.a_boundary, solver.b_boundary
        else:
            boundary = [coefficient_fields(problem.model, unit, np.cos(solver.theta),
                                            np.sin(solver.theta), 1) for unit in np.eye(len(parameters))]
            boundary_basis = (np.column_stack([item[0] for item in boundary]),
                              np.column_stack([item[1] for item in boundary]),
                              np.broadcast_to(bb, (solver.nt, len(bb))))
        problem._batch_basis = qb, ab, bb, *boundary_basis
    q_basis, a_basis, b_basis, qb_basis, ab_basis, bb_basis = problem._batch_basis
    q, a, b = q_basis@parameters, a_basis@parameters, float(b_basis@parameters)
    qb, ab, bb = qb_basis@parameters, ab_basis@parameters, bb_basis@parameters
    power = 3 if problem.model == "cubic" else 2
    jac = solver.jac if problem.case == 2 else np.ones(len(q))
    jb = solver.jb if problem.case == 2 else np.ones(solver.nt)
    normal = solver.normal_scale if problem.case == 2 else 1.
    denominator = 4*solver.r[-1]*(1+solver.h/4)
    bg = np.zeros_like(u0)
    bg[-solver.nt:] = jb[:, None]*(ab[:, None]*f**power+bb[:, None]*f**3-qb[:, None]*f)/denominator
    v = np.zeros_like(u0)
    active = np.ones(len(inputs), dtype=bool)
    for _ in range(130):
        u = u0+v
        z = solver.lu.solve(bg+jac[:, None]*(q[:, None]*u-a[:, None]*u**power-b*u**3))
        new = solver.lu.solve(-jac[:, None]*z)
        if not np.all(np.isfinite(new)):
            raise RuntimeError("Non-finite batched Picard iterate")
        step = np.max(abs(new-v), axis=0)
        v[:, active] = new[:, active]
        active &= step >= 2e-12
        if not np.any(active):
            break
    else:
        raise RuntimeError("Batched Picard iteration did not converge")
    index = (solver.nr-1)*solver.nt
    factor = -2/(solver.h*(1+solver.h/4)*normal)
    observed = obs0+factor*v[index]
    if not gradient:
        return observed
    u = u0+v
    c = np.zeros_like(u)
    c[index] = factor
    potential = q[:, None]-power*a[:, None]*u**(power-1)-3*b*u*u
    adjoint = np.zeros_like(u)
    active = np.ones(len(inputs), dtype=bool)
    for _ in range(130):
        mid = solver.lu.solve(c-potential*adjoint, trans="T")
        new = jac[:, None]*solver.lu.solve(jac[:, None]*mid, trans="T")
        step = np.max(abs(new-adjoint), axis=0)
        adjoint[:, active] = new[:, active]
        active &= step >= 2e-14
        if not np.any(active):
            break
    else:
        raise RuntimeError("Batched adjoint iteration did not converge")
    derivative = -(adjoint*u).T@q_basis+(adjoint*u**power).T@a_basis
    derivative += np.sum(adjoint*u**3, axis=0)[:, None]*b_basis
    weight = adjoint[-solver.nt:]/jac[-solver.nt:, None]*jb[:, None]/denominator
    derivative += (weight*f).T@qb_basis-(weight*f**power).T@ab_basis-(weight*f**3).T@bb_basis
    return observed, derivative
