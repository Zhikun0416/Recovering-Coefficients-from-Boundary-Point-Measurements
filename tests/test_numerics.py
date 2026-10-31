"""Check analytic boundary responses, convergence, and discrete derivatives."""

import os
import unittest

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from core.backend import Problem, make_design, parameters
from core.inverse import recover
from core.protocol import CorrectedProblem


class NumericalTests(unittest.TestCase):
    def test_exact_biharmonic_boundary_responses(self):
        disk = Problem(1, "quadratic", 12)
        f, g = np.zeros(8), np.zeros(8)
        f[0], f[1], g[2] = .08, .1, -.2
        boundary = disk.prepare([{"f": f, "g": g}])
        observed = disk.measure(np.zeros(6), boundary)[0]
        self.assertAlmostEqual(observed, .1 - .2 / 6, places=12)

        rounded = Problem(2, "quadratic", 12)
        from core.rounded_rectangle import map_derivative
        f, g = np.zeros(8), np.zeros(8)
        f[1] = .1
        boundary = rounded.prepare([{"f": f, "g": g}])
        observed = rounded.measure(np.zeros(6), boundary)[0]
        self.assertAlmostEqual(observed, .1 / abs(map_derivative(1.)), places=12)

    def test_analytic_rounded_boundary(self):
        from core.rounded_rectangle import conformal_map, map_derivative, KAPPA
        w = np.exp(2j*np.pi*np.arange(8192)/8192)
        z = conformal_map(w)
        self.assertLess(KAPPA, 1.)
        self.assertGreater(abs(map_derivative(w)).min(), 0.)
        np.testing.assert_allclose([z.real.min(), z.real.max(), z.imag.min(), z.imag.max()],
                                   [0., 2., 0., 1.], atol=1e-12)
        self.assertAlmostEqual(conformal_map(1.).real, 2., places=12)

    def test_manufactured_solution_convergence(self):
        for case in (1, 2):
            with self.subTest(case=case):
                coarse = Problem(case, "mixed", 12).manufactured()
                fine = Problem(case, "mixed", 24).manufactured()
                self.assertLess(fine["field_error_inf"], coarse["field_error_inf"] / 2)
                self.assertLess(fine["measurement_error"], coarse["measurement_error"] / 2)

    def test_corrected_adjoint_matches_central_differences(self):
        for case in (1, 2):
            rows = make_design(case, 4, 4, 123)["dual"]
            for model in ("linear", "quadratic", "cubic", "mixed"):
                with self.subTest(case=case, model=model):
                    problem = CorrectedProblem(case, model, level=12, data_level=18)
                    truth = parameters(model)[0]
                    inputs = problem.prepare(rows)
                    _, adjoint = problem.measure(truth, inputs, gradient=True)
                    central = np.column_stack([
                        (problem.measure(truth + direction, inputs)
                         - problem.measure(truth - direction, inputs)) / 2e-4
                        for direction in np.eye(len(truth)) * 1e-4
                    ])
                    error = np.linalg.norm(adjoint - central) / np.linalg.norm(central)
                    self.assertLess(error, 2e-5)

    def test_batched_measurements_match_individual_inputs(self):
        for case in (1, 2):
            rows = make_design(case, 2, 8, 456)
            for model in ("radial", "mixed", "linear", "quadratic", "cubic"):
                with self.subTest(case=case, model=model):
                    problem = Problem(case, model, 12)
                    truth = parameters(model)[0]
                    inputs = problem.prepare(rows)
                    gradient = model != "radial"
                    scalar = [problem.solve(truth, row, gradient=gradient) for row in inputs]
                    batch = problem.measure(truth, inputs, gradient=gradient)
                    values = batch[0] if gradient else batch
                    np.testing.assert_allclose(values, [item[0] for item in scalar], atol=1e-12, rtol=1e-12)
                    if gradient:
                        np.testing.assert_allclose(batch[1], [item[3] for item in scalar], atol=1e-12, rtol=1e-10)

    def test_noiseless_joint_recovery(self):
        for case in (1, 2):
            with self.subTest(case=case):
                problem = Problem(case, "mixed", 12)
                truth = parameters("mixed")[0]
                inputs = problem.prepare(make_design(case, 2, 32, 123))
                observed = problem.measure(truth, inputs)
                fit = recover(problem, inputs, observed, truth)
                self.assertLess(fit["fit_rms"], 1e-9)
                self.assertLess(fit["parameter_relative_RMS"], 1e-3)


if __name__ == "__main__":
    unittest.main()
