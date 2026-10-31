"""Analytic rounded rectangle and conformally mapped Navier solver.

The map is a Schwarz-Christoffel rectangle map restricted to an inner
disk. Its singularities lie strictly outside the closed computational
disk, so the physical boundary is analytic and regular everywhere.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.optimize import brentq
from scipy.special import eval_legendre

from .disk_mixed import DiskNavier

KAPPA = .9
SERIES_TERMS = 320


@lru_cache(maxsize=1)
def map_parameters():
    n = np.arange(SERIES_TERMS)

    def series(alpha):
        return KAPPA**n * eval_legendre(n, np.cos(2*alpha)) / (2*n+1)

    def aspect(alpha):
        c = series(alpha)
        return c.sum()/np.sum(c*(-1.)**n)-2

    alpha = brentq(aspect, 0., np.pi/4, xtol=1e-14)
    c = series(alpha)
    scale = .5/np.sum(c*(-1.)**n)
    return alpha, scale, c*scale


def conformal_map(w):
    return 1.+.5j + w*polyval(w*w, map_parameters()[2])


def map_derivative(w):
    alpha, scale, _ = map_parameters()
    return scale/np.sqrt(1-2*KAPPA*np.cos(2*alpha)*w*w+KAPPA**2*w**4)


def fourier_basis(theta):
    return np.r_[np.cos(np.arange(4)[:, None]*theta),
                 np.sin(np.arange(4)[:, None]*theta)]


class RoundedRectangleNavier:
    """Polar finite volumes for Delta U=J V, Delta V=J(F-qU)."""

    def __init__(self, nr, nt, model="quadratic"):
        polar = DiskNavier(nr, nt)
        for name in ("nr", "nt", "h", "dt", "r", "theta", "rr", "A", "lu"):
            setattr(self, name, getattr(polar, name))
        self.boundary_rhs = polar.boundary_rhs
        self.model = model
        self.w = self.rr*np.exp(1j*self.theta)
        self.z = conformal_map(self.w)
        self.X, self.Y = self.z.real, self.z.imag
        self.metric = abs(map_derivative(self.w))**2
        self.jac = self.metric.ravel()
        self.zb = conformal_map(np.exp(1j*self.theta))
        self.jb = abs(map_derivative(np.exp(1j*self.theta)))**2
        self.normal_scale = float(abs(map_derivative(1.)))
        self.basis = fourier_basis(self.theta)
        self._lifting_basis = None
        self.q_basis, self.a_basis, self.b_basis = self.coefficient_basis(self.z)
        self.q_boundary, self.a_boundary, self.b_boundary = self.coefficient_basis(self.zb)

    def coefficient_basis(self, z):
        x, y = (z.real-1).ravel(), (z.imag-.5).ravel()
        rho2 = x*x+y*y
        count = 4 if self.model == "radial" else 7 if self.model == "mixed" else 3 if self.model == "linear" else 6
        q = np.zeros((len(x), count))
        a, b = q.copy(), q.copy()
        q[:, :2] = np.stack((np.ones_like(x), rho2), axis=1)
        if self.model == "radial":
            a[:, 2:] = np.stack((np.ones_like(x), rho2), axis=1)
        else:
            q[:, 2] = x
            if count > 3:
                a[:, 3:6] = np.stack((np.ones_like(x), rho2, y), axis=1)
            if count == 7:
                b[:, 6] = 1.
        return q, a, b

    def lifting(self, f, g):
        """Analytic physical biharmonic lifting from boundary Fourier samples.

        If V is the harmonic extension of g and Z is the centered map,
        Delta_w Re(A_m(w)*conj(Z(w)))/4 = |Z'|^2 Re(w^m),
        where A_m'=Z' w^m. Subtracting the harmonic boundary trace
        of this particular solution enforces the prescribed f.
        """
        samples = len(f)
        fh, gh = np.fft.fft(f)/samples, np.fft.fft(g)/samples
        modes = np.arange(samples//2)
        active = modes[np.abs(gh[:len(modes)]) > 1e-14]
        n = np.arange(SERIES_TERMS)
        c = map_parameters()[2]
        points = 2048
        tb = 2*np.pi*np.arange(points)/points
        wb = np.exp(1j*tb)
        zb = conformal_map(wb)-(1.+.5j)
        z = self.z-(1.+.5j)
        particular = np.zeros_like(self.X)
        pb = np.zeros(points)
        derivative = 0.
        z1 = conformal_map(1.)-(1.+.5j)
        dz1 = map_derivative(1.)
        for m in active:
            ac = c*(2*n+1)/(2*n+m+1)
            ai = self.w**(m+1)*polyval(self.w*self.w, ac)
            ab = wb**(m+1)*polyval(wb*wb, ac)
            weight = gh[m]*(.25 if m == 0 else .5)
            particular += np.real(weight*ai*np.conj(z))
            pb += np.real(weight*ab*np.conj(zb))
            derivative += float(np.real(weight*(dz1*np.conj(z1)+ac.sum()*np.conj(dz1))))
        ph = np.fft.fft(pb)/points
        # Harmonic extension of f minus the particular boundary trace.
        kh = -ph
        kh[:samples//2] += fh[:samples//2]
        kh[-(samples//2-1):] += fh[-(samples//2-1):]
        k = np.arange(points//2)
        active = k[np.abs(kh[:len(k)]) > 1e-15]
        harmonic = np.zeros_like(self.X)
        harmonic_derivative = 0.
        for m in active:
            weight = kh[m]*(1 if m == 0 else 2)
            harmonic += np.real(weight*self.w**m)
            harmonic_derivative += float(np.real(m*weight))
        return particular+harmonic, (derivative+harmonic_derivative)/self.normal_scale

    def prepare(self, rows):
        if self._lifting_basis is None:
            zero = np.zeros(self.nt)
            self._lifting_basis = [self.lifting(self.basis[i] if component == 0 else zero,
                                               self.basis[i] if component == 1 else zero)
                                   for component in (0, 1) for i in range(8)]
        fields = np.array([item[0] for item in self._lifting_basis])
        observations = np.array([item[1] for item in self._lifting_basis])
        prepared = []
        for row in rows:
            coeff = np.r_[row["f"], row["g"]]
            prepared.append(dict(f=np.asarray(row["f"])@self.basis,
                                 g=np.asarray(row["g"])@self.basis,
                                 u0=np.einsum("k,kij->ij", coeff, fields),
                                 obs0=float(coeff@observations)))
        return prepared

    def solve(self, parameters, boundary, source=None, source_boundary=0., tol=2e-12, gradient=False):
        q, a, b = self.q_basis@parameters, self.a_basis@parameters, self.b_basis@parameters
        qb, ab, bb = self.q_boundary@parameters, self.a_boundary@parameters, self.b_boundary@parameters
        power = 3 if self.model == "cubic" else 2
        f = boundary["f"]
        u0 = boundary["u0"].ravel()
        forcing = np.zeros_like(q) if source is None else np.asarray(source).ravel()
        lap_z = self.jb*(ab*f**power+bb*f**3-qb*f+source_boundary)
        bg = self.boundary_rhs(np.zeros(self.nt), lap_z)
        v = np.zeros_like(q)
        for iteration in range(1, 131):
            u = u0+v
            z = self.lu.solve(bg+self.jac*(q*u-a*u**power-b*u**3-forcing))
            new = self.lu.solve(-self.jac*z)
            if not np.all(np.isfinite(new)):
                raise RuntimeError("Non-finite Picard iterate")
            step = float(np.max(abs(new-v)))
            v = new
            if step < tol:
                break
        else:
            raise RuntimeError("Picard iteration did not converge")
        u = u0+v
        flux = -2*v[(self.nr-1)*self.nt]/(self.h*(1+self.h/4)*self.normal_scale)
        z = self.lu.solve(bg+self.jac*(q*u-a*u**power-b*u**3-forcing))
        residual = float(np.max(abs(self.A@v+self.jac*z)))
        check = dict(iterations=iteration, residual_inf=residual, last_step_inf=step)
        result = (float(boundary["obs0"]+flux), u.reshape(self.nr, self.nt), check)
        if not gradient:
            return result
        c = np.zeros_like(u)
        c[(self.nr-1)*self.nt] = -2/(self.h*(1+self.h/4)*self.normal_scale)
        potential = q-power*a*u**(power-1)-3*b*u*u
        adjoint = np.zeros_like(u)
        for adj_it in range(1, 131):
            intermediate = self.lu.solve(c-potential*adjoint, trans="T")
            zeta = self.lu.solve(self.jac*intermediate, trans="T")
            new = self.jac*zeta
            adj_step = float(np.max(abs(new-adjoint)))
            adjoint = new
            if adj_step < 2e-14:
                break
        else:
            raise RuntimeError("Adjoint iteration did not converge")
        rhs = -self.q_basis*u[:, None]+self.a_basis*u[:, None]**power+self.b_basis*u[:, None]**3
        dbg = self.jb[:, None]*(-self.q_boundary*f[:, None]+self.a_boundary*f[:, None]**power+
                               self.b_boundary*f[:, None]**3)/(4*self.r[-1]*(1+self.h/4))
        deriv = adjoint@rhs-zeta[-self.nt:]@dbg
        check.update(adjoint_iterations=adj_it, adjoint_step_inf=adj_step)
        return (*result, deriv)
