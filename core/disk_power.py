"""Linear null model and quadratic/cubic candidates with spatial coefficients.

An analytic biharmonic lifting handles the known boundary input. The
outer face uses the prescribed boundary Laplacian in a quadratic Taylor
reconstruction. The resulting correction has homogeneous Navier data.
All inverse data are scalar outward derivatives at (1, 0).
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

MODELS = ("linear", "quadratic", "cubic")




def coefficients(parameters, r, theta):
    q0, q2, qx = parameters[:3]
    q = q0+q2*r*r+qx*r*np.cos(theta)
    c = np.zeros_like(q) if len(parameters) == 3 else parameters[3]+parameters[4]*r*r+parameters[5]*r*np.sin(theta)
    return q, c


class DiskNavier:
    def __init__(self, nr, nt, model='cubic'):
        if model not in MODELS:
            raise ValueError('Unknown candidate model')
        self.model = model
        self.power = {'linear': 1, 'quadratic': 2, 'cubic': 3}[model]
        self.npars = 3 if model == 'linear' else 6
        if nr < 4 or nt < 8:
            raise ValueError("Require nr >= 4 and nt >= 8")
        self.nr, self.nt = nr, nt
        self.h, self.dt = 1. / nr, 2 * np.pi / nt
        self.r = (np.arange(nr) + .5) * self.h
        self.theta = np.arange(nt) * self.dt
        self.rr = np.broadcast_to(self.r[:, None], (nr, nt))
        self.x = self.rr*np.cos(self.theta)
        self.y = self.rr*np.sin(self.theta)
        ones, zeros = np.ones(nr*nt), np.zeros(nr*nt)
        self.q_basis = np.stack((ones, self.rr.ravel()**2, self.x.ravel(), zeros, zeros, zeros), axis=1)[:, :self.npars]
        self.c_basis = np.stack((zeros, zeros, zeros, ones, self.rr.ravel()**2, self.y.ravel()), axis=1)[:, :self.npars]
        self.q_boundary = np.stack((np.ones(nt), np.ones(nt), np.cos(self.theta),
                                    np.zeros(nt), np.zeros(nt), np.zeros(nt)), axis=1)[:, :self.npars]
        self.c_boundary = np.stack((np.zeros(nt), np.zeros(nt), np.zeros(nt), np.ones(nt),
                                    np.ones(nt), np.sin(self.theta)), axis=1)[:, :self.npars]
        rows, cols, vals = [], [], []
        for i, ri in enumerate(self.r):
            for j in range(nt):
                k = i * nt + j
                angular = 1 / (ri * self.dt)**2
                diagonal = 2 * angular
                for jj in ((j - 1) % nt, (j + 1) % nt):
                    rows.append(k); cols.append(i * nt + jj); vals.append(-angular)
                if i:
                    inner = i / ((i + .5) * self.h**2)
                    diagonal += inner
                    rows.append(k); cols.append(k - nt); vals.append(-inner)
                if i < nr - 1:
                    outer = (i + 1) / ((i + .5) * self.h**2)
                    diagonal += outer
                    rows.append(k); cols.append(k + nt); vals.append(-outer)
                else:
                    diagonal += 2 / (ri * self.h**2 * (1 + self.h / 4))
                rows.append(k); cols.append(k); vals.append(diagonal)
        self.A = csc_matrix((vals, (rows, cols)), shape=(nr * nt, nr * nt))
        self.lu = splu(self.A)
        self._lifting_cache = {}

    def angular_second(self, boundary):
        return (np.roll(boundary, 1) - 2 * boundary + np.roll(boundary, -1)) / self.dt**2

    def boundary_rhs(self, boundary, boundary_laplacian):
        rhs = np.zeros((self.nr, self.nt))
        # At r=1: z_rr = Delta z - z_r - z_theta_theta.
        rhs[-1] = (2 * boundary / self.h**2 +
                   (boundary_laplacian - self.angular_second(boundary)) / 4)
        rhs[-1] /= self.r[-1] * (1 + self.h / 4)
        return rhs.ravel()

    def lifting(self, f, g):
        """Exact disk biharmonic extension of the trigonometric input.

        Mode m: u0 = r^m [f_m + g_m (r^2-1)/(4(m+1))].
        """
        key = (np.asarray(f).tobytes(), np.asarray(g).tobytes())
        if key in self._lifting_cache:
            return self._lifting_cache[key]
        modes = np.abs(np.fft.fftfreq(self.nt) * self.nt)
        fh, gh = np.fft.fft(f), np.fft.fft(g)
        r = self.r[:, None]
        coefficients = r**modes * (fh + gh * (r*r-1) / (4*(modes+1)))
        U0 = np.fft.ifft(coefficients, axis=1).real
        derivative = float(np.real(np.sum(modes*fh + gh/(2*(modes+1))) / self.nt))
        self._lifting_cache[key] = (U0, derivative)
        return U0, derivative

    def solve(self, parameters, f, g, source=None, source_boundary=0., tol=2e-12, gradient=False):
        """Return (M,U,checks), or (M,U,checks,dM/dp) with gradient=True."""
        if len(parameters) != self.npars:
            raise ValueError('Parameter dimension does not match model')
        qfield, cfield = coefficients(parameters, self.rr, self.theta)
        q, nonlinear = qfield.ravel(), cfield.ravel()
        m = self.power
        forcing = np.zeros_like(q) if source is None else np.asarray(source).ravel()
        U0, obs0 = self.lifting(f, g)
        u0 = U0.ravel()
        qb, cb = coefficients(parameters, 1., self.theta)
        lap_w = cb*f**m - qb*f + source_boundary
        # v=u-u0 and z=Delta v have homogeneous Navier data.
        bg = self.boundary_rhs(np.zeros(self.nt), lap_w)
        v = np.zeros_like(u0)
        for iteration in range(1, 131):
            u = u0 + v
            w = self.lu.solve(bg + q*u - nonlinear*u**m - forcing)
            vnew = self.lu.solve(-w)
            if not np.all(np.isfinite(vnew)):
                raise RuntimeError("Non-finite Picard iterate")
            step = np.max(np.abs(vnew - v))
            v = vnew
            if step < tol:
                break
        else:
            raise RuntimeError("Picard iteration did not converge in 130 steps")
        u = u0 + v
        U = u.reshape(self.nr, self.nt)
        flux0 = -2 * v.reshape(self.nr, self.nt)[-1, 0] / (self.h * (1 + self.h / 4))
        w_check = self.lu.solve(bg + q*u - nonlinear*u**m - forcing)
        residual = np.max(np.abs(self.A @ v + w_check))
        check = {"iterations": iteration, "residual_inf": float(residual), "last_step_inf": float(step)}
        result = (float(obs0+flux0), U, check)
        if not gradient:
            return result
        # (A^2 + diag(q-m*c*u^(m-1))) dv = -dbg - dq*u + dc*u^m.
        # Solve the transpose equation; A is NOT symmetric on the polar grid.
        c = np.zeros_like(u); c[(self.nr-1)*self.nt] = -2/(self.h*(1+self.h/4))
        tangent_potential = q-m*nonlinear*u**(m-1)
        adjoint = np.zeros_like(u)
        for adj_it in range(1, 131):
            new = self.lu.solve(self.lu.solve(c-tangent_potential*adjoint, trans='T'), trans='T')
            if not np.all(np.isfinite(new)):
                raise RuntimeError("Non-finite adjoint iterate")
            adj_step = np.max(np.abs(new-adjoint))
            adjoint = new
            if adj_step < 2e-14:
                break
        else:
            raise RuntimeError("Adjoint iteration did not converge")
        # Include parameter dependence of the boundary Laplacian correction.
        rhs = -self.q_basis*u[:, None]+self.c_basis*u[:, None]**m
        dbg = (-self.q_boundary*f[:, None]+self.c_boundary*f[:, None]**m)/(4*self.r[-1]*(1+self.h/4))
        rhs[-self.nt:] -= dbg
        deriv = adjoint @ rhs
        check.update(adjoint_iterations=adj_it, adjoint_step_inf=float(adj_step))
        return (*result, deriv)
