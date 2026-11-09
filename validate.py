import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from optimization import OBJECTIVES, Settings, solve
from optimization.line_search import golden_section
from optimization.reference import reference_solutions


ROOT = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise AssertionError(message)


class Quadratic:
    key = "quadratic"

    def value(self, point):
        return float(np.dot(point, point)/2)

    def gradient(self, point):
        return np.array(point, dtype=float)

    def change(self, point, displacement, gradient):
        return float(np.dot(gradient, displacement)+np.dot(displacement, displacement)/2)


def main():
    parser = argparse.ArgumentParser(description="Independent numerical checks")
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    metadata_file = args.results/"experiment.json"
    if metadata_file.exists():
        values = json.loads(metadata_file.read_text(encoding="utf-8"))["settings"]
        values["initial_point"] = tuple(values["initial_point"])
        settings = Settings(**values)
    else:
        settings = Settings()
    checks = []
    rng = np.random.default_rng(2026)
    gradient_error = hessian_error = difference_error = 0.0
    for obj in OBJECTIVES.values():
        for point in rng.uniform(-1.5, 2.0, (12, 2)):
            basis = np.eye(2)*1e-5
            finite_gradient = np.array([(obj.value(point+h)-obj.value(point-h))/(2e-5) for h in basis])
            finite_hessian = np.column_stack([(obj.gradient(point+h)-obj.gradient(point-h))/(2e-5) for h in basis])
            g_error = float(np.linalg.norm(finite_gradient-obj.gradient(point))/(1+np.linalg.norm(finite_gradient)))
            h_error = float(np.linalg.norm(finite_hessian-obj.hessian(point))/(1+np.linalg.norm(finite_hessian)))
            gradient_error, hessian_error = max(gradient_error, g_error), max(hessian_error, h_error)
            displacement = rng.normal(0, 0.2, 2)
            direct = obj.value(point+displacement)-obj.value(point)
            stable = obj.change(point, displacement, obj.gradient(point))
            difference_error = max(difference_error, float(abs(direct-stable)/(1+abs(direct))))
    check(gradient_error < 1e-7 and hessian_error < 1e-7, "Analytical derivatives failed finite differences")
    check(difference_error < 1e-12, "Stable objective increments disagree with independent direct evaluation")
    checks.append({"name": "Derivatives and stable objective increments", "passed": True,
                   "max_gradient_relative_error": gradient_error, "max_hessian_relative_error": hessian_error,
                   "max_increment_relative_error": difference_error})
    calls = []

    def parabola(alpha):
        calls.append(alpha)
        return (alpha-0.37)**2

    line = golden_section(parabola, tolerance=1e-10)
    check(abs(line.alpha-0.37) < 1e-9 and line.width <= 1e-10, "Golden section failed known minimum")
    check(len(calls) == 2+line.iterations, "Golden section evaluation reuse is incorrect")
    fixture = solve(Quadratic(), "division", Settings(initial_point=(1.0, 0.0), initial_step=0.5, tolerance=0.75))
    check((fixture.iterations, fixture.function_evaluations, fixture.gradient_evaluations) == (1, 3, 2),
          "Evaluation counters failed an independently countable quadratic example")
    checks.append({"name": "Known scalar minimum and evaluation accounting", "passed": True,
                   "golden_alpha": line.alpha, "fixture_counts": [1, 3, 2]})
    refs = reference_solutions()
    check(abs(refs["f1"]["t_brent"]-refs["f1"]["t_lambert"]) < 1e-13, "Lambert W reference mismatch")
    checks.append({"name": "Independent scalar and stationary-system references", "passed": True,
                   "references": refs})
    max_step_difference = 0.0
    interior_steps = boundary_steps = 0
    run_checks = []
    for key, obj in OBJECTIVES.items():
        for method in ["division", "golden"]:
            r = solve(obj, method, settings)
            check(r.status == "converged" and r.history[-1]["gradient_norm"] < settings.tolerance, f"{key}/{method} did not converge")
            final = np.array([r.history[-1]["x"], r.history[-1]["y"]])
            mu = 2.0 if key == "f1" else 2.0-np.sqrt(2.0)
            distance = np.linalg.norm(final-refs[key]["point"])
            check(distance <= r.history[-1]["gradient_norm"]/mu+2e-14, "Strong convexity error bound failed")
            check(r.gradient_evaluations == r.iterations+1, "Gradient count does not match accepted points")
            if method == "division":
                expected = 2+sum(h["backtracks_from_previous"]+1 for h in r.history[1:])
            else:
                expected = 2+sum(h["line_iterations_from_previous"]+2 for h in r.history[1:])
            check(r.function_evaluations == expected, "Function count does not match trace")
            for old, new in zip(r.history, r.history[1:]):
                point = np.array([old["x"], old["y"]])
                next_point = np.array([new["x"], new["y"]])
                alpha = new["alpha_from_previous"]
                gradient = obj.gradient(point)
                check(np.allclose(next_point, point-alpha*gradient, atol=1e-14, rtol=0), "Incorrect update")
                change = new["objective_change_from_previous"]
                check(change < 0, "Accepted step does not decrease the objective")
                direct_change = obj.value(next_point)-obj.value(point)
                check(abs(direct_change-change) < 1e-11*(1+abs(obj.value(point))), "Trace increment mismatch")
                if method == "division":
                    check(change <= -settings.armijo_epsilon*alpha*np.dot(gradient, gradient)*(1-1e-12), "Armijo condition failed")
                else:
                    check(0 < alpha < 1, "Golden section left the lecture interval")
                    derivative = lambda a: float(np.dot(obj.gradient(point-a*gradient), -gradient))
                    check(derivative(0) < 0, "Line direction is not descent")
                    if derivative(1) > 0:
                        interior_steps += 1
                        exact_alpha = brentq(derivative, 0, 1, xtol=1e-14)
                        difference = abs(alpha-exact_alpha)
                        max_step_difference = max(max_step_difference, difference)
                        check(difference < max(1e-6, 2*settings.line_tolerance), "Golden step disagrees with independent derivative root")
                        if settings.line_tolerance <= 1e-7 and new["gradient_norm"] > 1e-10:
                            check(abs(new["gradient_cosine_previous"]) < 1e-3, "Interior line search orthogonality is inaccurate")
                    else:
                        boundary_steps += 1
                        check(alpha > 1-max(1e-7, 2*settings.line_tolerance), "Boundary infimum was not approached")
            run_checks.append({"function": key, "method": method, "status": r.status,
                               "point_error": float(distance), "iterations": r.iterations})
    checks.append({"name": "Four descent runs and line search geometry", "passed": True,
                   "max_alpha_difference_from_brent": max_step_difference, "runs": run_checks,
                   "main_interior_line_steps": interior_steps, "main_boundary_line_steps": boundary_steps})
    direction = np.linalg.eigh(np.array([[1.0, -1.0], [-1.0, 3.0]]))[1][:, 0]
    start = np.array([0.5, -0.5])+0.01*direction
    gradient = OBJECTIVES["f2"].gradient(start)
    derivative_at_upper = float(np.dot(OBJECTIVES["f2"].gradient(start-gradient), -gradient))
    boundary = solve(OBJECTIVES["f2"], "golden", replace(Settings(), initial_point=tuple(start)))
    first = boundary.history[1]
    check(derivative_at_upper < 0 and first["alpha_from_previous"] > 1-1e-7,
          "Boundary example did not approach the upper interval endpoint")
    check(abs(first["gradient_cosine_previous"]) > 0.9, "Boundary case incorrectly treated as orthogonal")
    checks.append({"name": "Open interval boundary limitation", "passed": True,
                   "initial_point": start.tolist(), "derivative_at_alpha_one": derivative_at_upper,
                   "first_alpha": first["alpha_from_previous"],
                   "first_cosine": first["gradient_cosine_previous"]})
    sweep_files = ["tolerance_sweep.csv", "step_sweep.csv", "line_accuracy_sweep.csv", "initial_point_sweep.csv"]
    sweep_count = 0
    for name in sweep_files:
        file = args.results/name
        if file.exists():
            with file.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    check(row["status"] == "converged", f"A saved sweep failed: {name}")
                    check(float(row["gradient_norm"]) < float(row.get("tolerance", settings.tolerance)), "Sweep stopping check failed")
                    sweep_count += 1
    checks.append({"name": "Saved sensitivity experiments", "passed": True, "runs_checked": sweep_count})
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results/"validation.json").write_text(json.dumps({"passed": True, "checks": checks}, indent=2)+"\n", encoding="utf-8")
    for item in checks:
        print("PASS  "+item["name"])
    print(f"All {len(checks)} validation groups passed.")


if __name__ == "__main__":
    main()
