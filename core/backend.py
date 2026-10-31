"""Common model, input design and solver interface for the two geometries."""
import numpy as np

from . import disk_radial
from . import disk_mixed
from . import disk_power
from . import rounded_rectangle


MODELS = {1: "radial", 2: "mixed", 3: "cubic", 4: "quadratic"}
SEEDS = {1: 20261014, 2: 20261018, 3: 20261019, 4: 20261020}


def parameters(model):
    if model == "radial":
        return (np.array([1.1, .8, 2.4, 1.5]), np.array([.85, .45, 1.7, .95]),
                ([.1, -.1, .05, -.1], [4., 4., 8., 8.]), ["q0", "q2", "a0", "a2"])
    truth = np.array([1.1, .8, .2, 2.4, 1.5, .3, .15])
    initial = np.array([.9, .5, .05, 1.8, 1., .1, .05])
    lower, upper = [.5, 0., -.4, -6., -5., -2., -2.], [3., 3., .4, 6., 5., 2., 2.]
    count = 7 if model == "mixed" else 3 if model == "linear" else 6
    if model == "mixed":
        lower[3:5] = [.1, 0.]
    names = ["q0", "q2", "qx", "a0", "a2", "ay", "b3"]
    if model == "cubic":
        names[3:6] = ["c0", "c2", "cy"]
    return truth[:count], initial[:count], (lower[:count], upper[:count]), names[:count]


def coefficient_fields(model, p, x, y, case):
    xi, eta = (x, y) if case == 1 else (x-1., y-.5)
    rho2 = xi*xi+eta*eta
    q = p[0]+p[1]*rho2
    if model == "radial":
        a = p[2]+p[3]*rho2
    else:
        q = q+p[2]*xi
        a = np.zeros_like(q) if model == "linear" else p[3]+p[4]*rho2+p[5]*eta
    return q, a, p[6] if model == "mixed" else 0.


def nonlinear(model, a, b, u):
    return a*u**(3 if model == "cubic" else 2)+b*u**3


def boundary_basis(case, x, y):
    theta = np.arctan2(y, x)
    return np.concatenate((np.cos(np.arange(4)[:, None]*theta.ravel()),
                           np.sin(np.arange(4)[:, None]*theta.ravel())), axis=0)


def reference_boundary(case):
    if case == 1:
        theta = 2*np.pi*np.arange(8192)/8192
        basis = boundary_basis(1, np.cos(theta), np.sin(theta))
        gram = np.diag([1., .5, .5, .5, 0., .5, .5, .5])
    else:
        theta = 2*np.pi*np.arange(8192)/8192
        basis = rounded_rectangle.fourier_basis(theta)
        speed = abs(rounded_rectangle.map_derivative(np.exp(1j*theta)))
        gram = (basis*speed)@basis.T/speed.sum()
    return basis, gram


def make_design(case, ex, n, seed, amplitude=None):
    amplitude = (.4 if ex == 1 else .6) if amplitude is None else amplitude
    peak_basis, gram = reference_boundary(case)
    rng = np.random.default_rng(seed)
    scales = (1., -1.) if ex == 1 else (.5, -.5, 1., -1.)
    result = {name: [] for name in ("f_only", "g_only", "dual")} if ex == 4 else []
    for direction in range(n//len(scales)):
        fc, gc = rng.normal(0, .75, (2, len(gram)))
        fc[4] = gc[4] = 0.
        if ex == 4:
            fc *= amplitude/np.sqrt(fc@gram@fc)
            gc *= 3*amplitude/np.sqrt(gc@gram@gc)
            groups = [("f_only", fc, np.zeros_like(gc)), ("g_only", np.zeros_like(fc), gc),
                      ("dual", fc/np.sqrt(2), gc/np.sqrt(2))]
        else:
            fc *= amplitude/np.max(abs(fc@peak_basis))
            gc *= amplitude/np.max(abs(gc@peak_basis))
            kind = direction % 4
            if kind == 0:
                gc *= 0
            elif kind == 1:
                fc *= 0; gc *= 3
            else:
                gc *= 2.5 if kind == 2 else -2.5
            groups = [(["f_only", "g_only", "mixed", "mixed"][kind], fc, gc)]
        for name, f, g in groups:
            for scale in scales:
                row = dict(f=(scale*f).tolist(), g=(scale*g).tolist(), scale=scale,
                           direction=direction, kind=name,
                           weighted_energy=float(scale**2*(f@gram@f+(g@gram@g)/9)))
                (result[name] if ex == 4 else result).append(row)
    return result


class Problem:
    def __init__(self, case, model, level):
        self.case, self.model, self.level = case, model, level
        if case == 1:
            module = disk_radial if model == "radial" else disk_mixed if model == "mixed" else disk_power
            self.disk = module.DiskNavier(level, 2*level) if model in ("radial", "mixed") else module.DiskNavier(level, 2*level, model)
            self.X, self.Y = self.disk.rr*np.cos(self.disk.theta), self.disk.rr*np.sin(self.disk.theta)
            self.weights = np.broadcast_to(self.disk.rr, self.X.shape)
            self.grid = [level, 2*level]
        else:
            self.disk = rounded_rectangle.RoundedRectangleNavier(level, 2*level, model=model)
            self.X, self.Y = self.disk.X, self.disk.Y
            self.weights = self.disk.rr*self.disk.metric
            self.grid = [level, 2*level]

    def prepare(self, design):
        if self.case == 1:
            theta = self.disk.theta
            basis = np.r_[np.cos(np.arange(4)[:, None]*theta), np.sin(np.arange(4)[:, None]*theta)]
            return [(np.array(row["f"])@basis, np.array(row["g"])@basis) for row in design]
        return self.disk.prepare(design)

    def solve(self, p, boundary, gradient=False):
        if self.case == 2:
            return self.disk.solve(p, boundary, gradient=gradient)
        if self.model == "radial":
            return self.disk.solve(p, *boundary)
        return self.disk.solve(p, *boundary, gradient=gradient)

    def measure(self, p, inputs, gradient=False):
        from .batch import measure
        return measure(self, p, inputs, gradient=gradient)

    def fields(self, p):
        return coefficient_fields(self.model, p, self.X, self.Y, self.case)

    def field_errors(self, truth, recovered):
        qt, at, _ = self.fields(truth)
        qr, ar, _ = self.fields(recovered)
        return {name: float(np.sqrt(np.sum((rec-ref)**2*self.weights)/np.sum(ref**2*self.weights)))
                for name, ref, rec in (("q_relative_L2", qt, qr), ("coefficient_relative_L2", at, ar))
                if np.any(ref)}

    def manufactured(self, nonlinear_case=True):
        p = parameters(self.model)[0] if nonlinear_case else np.zeros(len(parameters(self.model)[0]))
        scale = .1 if nonlinear_case else 1.
        if self.case == 1:
            r, t = self.disk.rr, self.disk.theta
            exact = scale*(1-r**4+.1*r**3*np.cos(3*t)+.08*r**4*np.cos(2*t))
            f = scale*(.1*np.cos(3*t)+.08*np.cos(2*t))
            g = scale*(-16+.96*np.cos(2*t))
            q, a, b = self.fields(p)
            qb, ab, bb = coefficient_fields(self.model, p, np.cos(t), np.sin(t), 1)
            source = -64*scale+q*exact-nonlinear(self.model, a, b, exact)
            source_b = -64*scale+qb*f-nonlinear(self.model, ab, bb, f)
            obs, field, check = self.disk.solve(p, f, g, source=source, source_boundary=source_b)
            obs_exact = -3.38*scale
        else:
            def values(z):
                zz = z-(1.+.5j)
                rho2 = abs(zz)**2
                u = scale*(1+.1*zz.real+.08*zz.imag+.05*(zz**3).real+.1*rho2**2)
                return u, 1.6*scale*rho2
            exact, _ = values(self.disk.z)
            theta = 2*np.pi*np.arange(2048)/2048
            bf, bg = values(rounded_rectangle.conformal_map(np.exp(1j*theta)))
            u0, obs0 = self.disk.lifting(bf, bg)
            f, g = values(self.disk.zb)
            boundary = dict(f=f, g=g, u0=u0, obs0=obs0)
            q, a, b = self.fields(p)
            qb, ab, bb = coefficient_fields(self.model, p, self.disk.zb.real, self.disk.zb.imag, 2)
            source = 6.4*scale+q*exact-nonlinear(self.model, a, b, exact)
            source_b = 6.4*scale+qb*f-nonlinear(self.model, ab, bb, f)
            obs, field, check = self.disk.solve(p, boundary, source=source, source_boundary=source_b)
            obs_exact = .65*scale
        return dict(field_error_inf=float(abs(field-exact).max()), measurement_error=float(abs(obs-obs_exact)), **check)
