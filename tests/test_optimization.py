from dataclasses import replace
import csv
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from optimization import METHODS, OBJECTIVES, OptimizationSettings, minimize
from optimization.line_search import golden_section
from optimization.newton import newton_direction
from optimization.quasi_newton import broyden_update
from optimization.reference import reference_solutions


ROOT = Path(__file__).resolve().parents[1]


class Quadratic:
    key = "quadratic"

    def __init__(self, diagonal):
        self.matrix = np.diag(diagonal)
        self.value_calls = self.gradient_calls = self.hessian_calls = 0

    def value(self, point):
        self.value_calls += 1
        return float(point @ self.matrix @ point / 2)

    def gradient(self, point):
        self.gradient_calls += 1
        return self.matrix @ point

    def hessian(self, point):
        self.hessian_calls += 1
        return self.matrix


class DerivativeAndLineTests(unittest.TestCase):
    def test_derivatives_and_uniform_hessian_bounds(self):
        for key, objective in OBJECTIVES.items():
            for point in (np.array([0.25, -0.75]), np.array([0.0, 10.0]), np.array([10.0, 0.0])):
                h = 1e-5
                axes = np.eye(2) * h
                gradient_fd = [(objective.value(point + e) - objective.value(point - e)) / (2 * h)
                               for e in axes]
                hessian_fd = np.column_stack([
                    (objective.gradient(point + e) - objective.gradient(point - e)) / (2 * h)
                    for e in axes
                ])
                hessian = objective.hessian(point)
                assert_allclose(objective.gradient(point), gradient_fd, rtol=1e-7, atol=1e-8)
                assert_allclose(hessian, hessian_fd, rtol=1e-7, atol=1e-8)
                assert_allclose(hessian, hessian.T, atol=0)
                eigenvalues = np.linalg.eigvalsh(hessian)
                self.assertGreaterEqual(eigenvalues[0], 2 if key == "f1" else 2 - np.sqrt(2) - 1e-14)
                if key == "f2":
                    self.assertLessEqual(eigenvalues[-1], 4 + np.sqrt(2) + 1e-14)

    def test_golden_reuses_values_and_selects_closed_endpoints(self):
        for minimizer, expected in ((0.37, 0.37), (-1.0, 0.01), (1.0, 1.0), (2.0, 1.0)):
            calls = []

            def parabola(alpha):
                calls.append(alpha)
                return (alpha - minimizer) ** 2

            line = golden_section(parabola, 0.01, 1.0, include_endpoints=True)
            self.assertTrue(line.converged)
            self.assertAlmostEqual(line.alpha, expected, delta=1e-9)
            self.assertEqual(len(calls), 4 + line.iterations)
            self.assertEqual(calls[-2:], [0.01, 1.0])
        open_line = golden_section(lambda a: (a - 2.0) ** 2)
        self.assertLess(open_line.alpha, 1.0)

    def test_line_and_outer_limits_are_independent(self):
        objective = Quadratic([2, 3])
        settings = OptimizationSettings(initial_point=(1, 1), max_line_iterations=1, max_iterations=1)
        result = minimize(objective, "newton_solve_golden", settings)
        self.assertEqual((result.status, result.iterations, result.counts.function_evaluations),
                         ("line_search_limit", 0, 6))
        result = minimize(objective, "newton_solve_fixed", settings)
        self.assertEqual((result.status, result.iterations), ("converged", 1))


class NewtonTests(unittest.TestCase):
    def test_equivalent_directions_and_independent_residual(self):
        for objective in OBJECTIVES.values():
            for point in ((0, 0), (0, 10), (10, 0), (0.75, -0.4)):
                h, g = objective.hessian(point), objective.gradient(point)
                inv = newton_direction(h, g, "inverse")
                solve = newton_direction(h, g, "solve")
                assert_allclose(inv, solve, rtol=1e-13, atol=1e-14)
                assert_allclose(h @ inv, -g, rtol=1e-13, atol=1e-14)
                assert_allclose(h @ solve, -g, rtol=1e-13, atol=1e-14)

    def test_quadratic_in_three_dimensions_and_actual_call_counters(self):
        settings = OptimizationSettings(initial_point=(1, 2, -1))
        for method in METHODS:
            if not method.startswith("newton"):
                continue
            with self.subTest(method=method):
                objective = Quadratic([2, 5, 10])
                result = minimize(objective, method, settings)
                self.assertEqual((result.status, result.iterations), ("converged", 1))
                assert_allclose(result.history[-1].point, 0.0, atol=1e-14)
                self.assertEqual(result.history[-1].alpha, 1.0)
                counts = result.counts
                expected_f = 2 if method.endswith("fixed") else 6 + result.history[-1].line_iterations
                self.assertEqual((counts.function_evaluations, counts.gradient_evaluations,
                                  counts.hessian_evaluations), (expected_f, 2, 1))
                self.assertEqual((objective.value_calls, objective.gradient_calls,
                                  objective.hessian_calls), (expected_f, 2, 1))
                self.assertEqual(counts.matrix_inversions, int("inverse" in method))
                self.assertEqual(counts.linear_solves, int("solve" in method))

    def test_matrix_failures_and_nonfinite_objective(self):
        settings = OptimizationSettings(initial_point=(1, 1))
        for diagonal, status in (([1, 0], "singular_matrix"), ([1, 1e-15], "singular_matrix"),
                                 ([1, -1], "invalid_search_direction"), ([1, np.nan], "nonfinite_matrix")):
            objective = Quadratic(diagonal)
            if np.isnan(diagonal[1]):
                objective.value = lambda x: float(x @ x)
                objective.gradient = lambda x: x
            result = minimize(objective, "newton_solve_fixed", settings)
            self.assertEqual((result.status, result.iterations, result.counts.linear_solves), (status, 0, 0))

        objective = Quadratic([1, 1])
        objective.value = lambda point: 1.0 if point[0] else float("inf")
        result = minimize(objective, "newton_inverse_fixed", settings)
        self.assertEqual((result.status, result.iterations, result.counts.function_evaluations),
                         ("nonfinite_objective", 0, 2))

    def test_stopping_uses_strict_gradient_threshold(self):
        objective = Quadratic([1, 1, 1])
        settings = OptimizationSettings(initial_point=(1, 0, 0), tolerance=1.0, max_iterations=1)
        result = minimize(objective, "newton_solve_fixed", settings)
        self.assertEqual((result.status, result.iterations), ("converged", 1))
        optimal = minimize(objective, "newton_solve_fixed", replace(settings, initial_point=(0, 0, 0)))
        self.assertEqual((optimal.status, optimal.iterations, optimal.counts.hessian_evaluations),
                         ("converged", 0, 0))


class QuasiNewtonTests(unittest.TestCase):
    def test_update_preserves_secant_and_spd_and_has_correct_dfp_endpoint(self):
        matrix = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.2], [0.0, 0.2, 1.0]])
        step = np.array([0.4, -0.5, 0.8])
        change = np.array([1.1, -0.7, 0.6])
        bfgs = broyden_update(matrix, step, change)
        rho = 1.0 / (change @ step)
        # Independent Hessian form of the DFP endpoint.
        transform = np.eye(step.size) - rho * np.outer(change, step)
        dfp = transform @ matrix @ transform.T + rho * np.outer(change, change)
        for phi in (0, 0.5, 1):
            with self.subTest(phi=phi):
                update = broyden_update(matrix, step, change, phi)
                self.assertEqual(update.status, "accepted")
                assert_allclose(update.matrix @ step, change, rtol=1e-14, atol=1e-15)
                assert_allclose(update.matrix, update.matrix.T, atol=0)
                self.assertGreater(np.linalg.eigvalsh(update.matrix).min(), 0)
                assert_allclose(update.matrix, (1 - phi) * bfgs.matrix + phi * dfp, atol=2e-15)
        self.assertGreater(np.linalg.norm(bfgs.matrix - dfp), 0.01)
        update = broyden_update(matrix, step, change, 0.5, adaptive=True)
        self.assertTrue(0.1 <= update.phi <= 0.9)
        assert_allclose(update.matrix @ step, change, atol=1e-14)

    def test_curvature_rejections_are_reported(self):
        matrix = np.eye(2)
        for change in (np.array([-1.0, 1.0]), np.array([0.0, 1.0]), np.zeros(2)):
            rejected = broyden_update(matrix, np.array([1.0, 0.0]), change, 0.5)
            self.assertEqual(rejected.status, "skipped_curvature")
            self.assertIsNone(rejected.matrix)

        class Linear:
            key = "linear"

            def value(self, point):
                return float(point[0])

            def gradient(self, point):
                return np.array([1.0, 0.0])

        result = minimize(Linear(), "bfgs_fixed",
                          OptimizationSettings(initial_point=(0, 0), max_iterations=2))
        self.assertEqual((result.status, result.iterations, result.counts.skipped_updates),
                         ("max_iterations", 2, 2))
        self.assertEqual([item.update_status for item in result.history[1:]], ["skipped_curvature"] * 2)

    def test_quasi_newton_requires_no_hessian_and_fixed_step_can_be_uphill(self):
        class HessianForbidden(Quadratic):
            def hessian(self, point):
                raise AssertionError("Quasi-Newton called the analytical Hessian.")

        for method in ("bfgs_fixed", "bfgs_golden", "broyden_fixed", "broyden_golden"):
            objective = HessianForbidden([2, 4, 3])
            result = minimize(objective, method, OptimizationSettings(initial_point=(1, 2, -1)))
            self.assertEqual(result.status, "converged")
            self.assertEqual(result.counts.hessian_evaluations, 0)
            self.assertEqual(result.counts.linear_solves, result.iterations)
            self.assertEqual(result.counts.quasi_newton_updates + result.counts.skipped_updates,
                             result.iterations)
            if method.endswith("fixed"):
                self.assertTrue(all(item.alpha == 1 for item in result.history[1:]))
                self.assertGreater(result.history[1].value, result.history[0].value)

    def test_generic_gradient_descent_accepts_a_gradient_only_three_dimensional_objective(self):
        goal = np.array([1, 2, -1], dtype=float)

        class Shifted:
            key = "shifted"

            def value(self, point):
                return np.sum((point - goal) ** 2) / 2

            def gradient(self, point):
                return point - goal

        result = minimize(Shifted(), "division", OptimizationSettings(initial_point=(0, 0, 0)))
        self.assertEqual(result.status, "converged")
        assert_allclose(result.history[-1].point, goal, atol=1e-6)


class ExperimentTests(unittest.TestCase):
    def test_both_objectives_and_all_starting_points(self):
        references = reference_solutions()
        for key, objective in OBJECTIVES.items():
            mu = 2 if key == "f1" else 2 - np.sqrt(2)
            for point in ((0, 0), (0, 10), (10, 0)):
                settings = OptimizationSettings(initial_point=point)
                for method in METHODS:
                    with self.subTest(objective=key, point=point, method=method):
                        result = minimize(objective, method, settings)
                        last = result.history[-1]
                        if key == "f1" and point == (0, 10) and method in ("bfgs_fixed", "broyden_fixed"):
                            self.assertEqual(result.status, "nonfinite_objective")
                            self.assertGreater(result.counts.skipped_updates, 0)
                        else:
                            self.assertEqual(result.status, "converged")
                            self.assertLess(last.gradient_norm, settings.tolerance)
                            distance = np.linalg.norm(np.array(last.point) - references[key]["point"])
                            self.assertLessEqual(distance, last.gradient_norm / mu + 1e-14)
                        if METHODS[method].step == "unit":
                            self.assertTrue(all(it.alpha == 1.0 for it in result.history[1:]))
                        if METHODS[method].step == "golden_closed":
                            self.assertTrue(all(settings.line_lower <= it.alpha <= settings.line_upper
                                                for it in result.history[1:]))

    def test_regression_against_published_first_order_snapshots(self):
        for folder in ("results", "demo_0_0", "demo_0_10", "demo_10_0"):
            with (ROOT / folder / "summary.csv").open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    key = row["function"]
                    method = "division" if row["method"] == "Step Division" else "golden"
                    point = (float(row["initial_x"]), float(row["initial_y"]))
                    with self.subTest(function=key, point=point, method=method):
                        run = minimize(OBJECTIVES[key], method, OptimizationSettings(initial_point=point))
                        self.assertEqual((run.status, run.iterations), (row["status"], int(row["iterations"])))
                        self.assertEqual(run.counts.function_evaluations, int(row["function_evaluations"]))
                        self.assertEqual(run.counts.gradient_evaluations, int(row["gradient_evaluations"]))
                        assert_allclose(run.history[-1].point, [float(row["final_x"]), float(row["final_y"])],
                                        rtol=0, atol=1e-7)
                        with (ROOT / folder / "histories" / f"{key}_{method}.csv").open(
                            encoding="utf-8", newline="",
                        ) as trace:
                            old_points = [[float(r["x"]), float(r["y"])] for r in csv.DictReader(trace)]
                        assert_allclose([entry.point for entry in run.history], old_points, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
