from concurrent.futures import CancelledError
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose

from optimization import OBJECTIVES
from optimization.desktop import charts
from optimization.desktop.session import DEFAULT_SETTINGS, Experiment, execute, export_session
from optimization.exports import read_csv
from optimization.reference import reference_for_objective


class DesktopExperimentTests(unittest.TestCase):
    def test_custom_coefficients_update_derivatives_and_reference(self):
        for original in OBJECTIVES.values():
            with self.subTest(objective=original.key):
                objective = replace(original, a=1.7, b=2.8, c=-0.4, d=0.9, k=2.3, m=-1.1)
                point = np.array([0.3, -0.7])
                h = 1e-5
                offsets = np.eye(2) * h
                gradient = [(objective.value(point + step) - objective.value(point - step)) / (2 * h)
                            for step in offsets]
                hessian = np.column_stack([
                    (objective.gradient(point + step) - objective.gradient(point - step)) / (2 * h)
                    for step in offsets
                ])
                assert_allclose(objective.gradient(point), gradient, rtol=1e-8, atol=1e-8)
                assert_allclose(objective.hessian(point), hessian, rtol=1e-8, atol=1e-8)
                shift = np.array([0.2, -0.3])
                self.assertAlmostEqual(objective.change(point, shift, objective.gradient(point)),
                                       objective.value(point + shift) - objective.value(point), places=13)
                reference = reference_for_objective(objective)
                self.assertIsNotNone(reference)
                assert_allclose(objective.gradient(reference["point"]), [0, 0], atol=1e-13)
                session = execute(Experiment(objective, DEFAULT_SETTINGS, ("newton_solve_fixed",)))
                run = session.runs[0]
                self.assertEqual(run.result.status, "converged")
                assert_allclose(run.result.history[-1].point, reference["point"], atol=1e-6)
                self.assertEqual(run.trace[-1]["objective"],
                                 objective.value(run.result.history[-1].point))

    def test_unverified_global_reference_leaves_error_columns_empty(self):
        objective = replace(OBJECTIVES["f2"], a=0.15)
        session = execute(Experiment(objective, DEFAULT_SETTINGS, ("newton_solve_fixed",)))
        self.assertIsNone(session.reference)
        self.assertEqual(session.runs[0].result.status, "invalid_search_direction")
        row = session.runs[0].trace[0]
        self.assertIsNone(row["point_error"])
        self.assertIsNone(row["objective_gap"])

    def test_experiment_keeps_stops_and_compares_adaptive_variant(self):
        settings = replace(DEFAULT_SETTINGS, initial_point=(0, 10))
        session = execute(Experiment(OBJECTIVES["f1"], settings,
                                     ("bfgs_fixed", "bfgs_golden", "broyden_golden"),
                                     repeats=2, compare_adaptive=True))
        self.assertEqual(
            [(run.id, run.result.status) for run in session.runs],
            [("bfgs_fixed", "nonfinite_objective"), ("bfgs_golden", "converged"),
             ("broyden_golden", "converged"), ("broyden_golden_adaptive", "converged")],
        )
        self.assertEqual(len(session.runs[0].trace), 4)
        self.assertEqual(len(session.runs[1].trace), 6)
        self.assertTrue(all(run.timings and len(run.timings) == 2 for run in session.runs))

    def test_export_records_inputs_outcomes_histories_and_figures(self):
        settings = replace(DEFAULT_SETTINGS, initial_point=(-1, 2), tolerance=1e-7)
        session = execute(Experiment(OBJECTIVES["f2"], settings, ("division", "bfgs_golden")))
        with TemporaryDirectory() as directory:
            output = export_session(session, directory, {
                "trajectories": charts.trajectories(session),
                "convergence": charts.convergence(session),
                "work": charts.work(session),
            })
            setup = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(setup["experiment"]["settings"]["initial_point"], [-1, 2])
            self.assertEqual(setup["experiment"]["settings"]["tolerance"], 1e-7)
            results = read_csv(output / "summary.csv")
            self.assertEqual([row["method"] for row in results], ["division", "bfgs_golden"])
            for row in results:
                trace = read_csv(output / row["history_csv"])
                self.assertEqual(len(trace), 1 + int(row["iterations"]))
                self.assertEqual(trace[-1]["gradient_norm"], row["gradient_norm"])
            self.assertEqual(len(read_csv(output / "benchmark_times.csv")), 2)
            for plot in Path(output / "figures").glob("*.png"):
                self.assertTrue(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 3)

    def test_stop_request_can_prevent_next_solver_call(self):
        experiment = Experiment(OBJECTIVES["f1"], DEFAULT_SETTINGS, ("division", "bfgs_golden"))
        self.assertRaises(CancelledError, execute, experiment, cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
