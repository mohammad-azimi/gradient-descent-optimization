import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
from scipy.optimize import brentq

from optimization import METHODS, OBJECTIVES
from optimization.exports import read_csv, write_json
from optimization.reference import reference_solutions


ROOT = Path(__file__).resolve().parent


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def check_trace(row, trace, settings, reference):
    case = row["case"]
    method = METHODS[row["method"]]
    objective = OBJECTIVES[row["function"]]
    iterations = int(row["iterations"])
    converged = row["status"] == "converged"
    points = np.array([[float(item["x"]), float(item["y"])] for item in trace])
    check(len(trace) == iterations + 1, f"{case}: trace length")
    check(np.array_equal(points[0], [float(row["initial_x"]), float(row["initial_y"])]),
          f"{case}: initial point")
    check(np.array_equal(points[-1], [float(row["final_x"]), float(row["final_y"])]),
          f"{case}: final point")
    check(float(row["objective"]) == float(trace[-1]["objective"]), f"{case}: reported objective")
    check(float(row["gradient_norm"]) == float(trace[-1]["gradient_norm"]), f"{case}: reported norm")
    check(float(row["point_error"]) == float(trace[-1]["point_error"]), f"{case}: reported error")
    max_line_alpha_error = max_direction_residual = max_secant_residual = 0.0
    total_line_calls = 0
    updates = Counter()
    for k, (item, point) in enumerate(zip(trace, points)):
        check(int(item["iteration"]) == k, f"{case}: iteration index")
        gradient = objective.gradient(point)
        gradient_norm = np.linalg.norm(gradient)
        check(math.isclose(gradient_norm, float(item["gradient_norm"]), rel_tol=1e-12, abs_tol=1e-13),
              f"{case}: gradient norm")
        check(math.isclose(objective.value(point), float(item["objective"]), rel_tol=1e-12, abs_tol=1e-12),
              f"{case}: objective")
        check(math.isclose(np.linalg.norm(point - reference["point"]), float(item["point_error"]),
                           rel_tol=1e-12, abs_tol=1e-14), f"{case}: reference error")
        if not k:
            continue
        previous = points[k - 1]
        direction = np.array([float(item["direction_x"]), float(item["direction_y"])])
        alpha = float(item["alpha"])
        check(np.allclose(previous + alpha * direction, point, rtol=1e-13, atol=1e-13),
              f"{case}: iterate does not follow its step")
        if method.step == "unit":
            check(alpha == 1 and int(item["line_contractions"]) == 0, f"{case}: changed unit step")
        elif method.step == "division":
            total_line_calls += 1 + int(item["backtracks"])
        elif method.step == "golden_open":
            check(0 < alpha < 1, f"{case}: left the original open bracket")
            total_line_calls += 2 + int(item["line_contractions"])
        else:
            total_line_calls += 4 + int(item["line_contractions"])
            lower, upper = settings["line_lower"], settings["line_upper"]
            check(lower <= alpha <= upper, f"{case}: left the closed bracket")
            check(float(item["line_width"]) <= settings["line_tolerance"], f"{case}: line width")

            def derivative(a):
                return float(objective.gradient(previous + a * direction) @ direction)

            if derivative(lower) >= 0:
                expected = lower
            elif derivative(upper) <= 0:
                expected = upper
            else:
                expected = brentq(derivative, lower, upper, xtol=1e-14)
            disagreement = abs(alpha - expected)
            max_line_alpha_error = max(max_line_alpha_error, disagreement)
            check(disagreement <= max(1e-6, 4 * settings["line_tolerance"]),
                  f"{case}: line minimum differs from independent derivative root")
        if method.family != "gradient":
            check(float(item["direction_matrix_min_eigenvalue"]) > 0, f"{case}: positive direction matrix")
            check(float(item["direction_matrix_rcond"]) > settings["matrix_rcond"], f"{case}: matrix condition")
            max_direction_residual = max(max_direction_residual, float(item["relative_direction_residual"]))
        if item["update_status"]:
            updates[item["update_status"]] += 1
            if item["update_status"] == "accepted":
                residual = float(item["relative_secant_residual"])
                check(residual <= 1e-8, f"{case}: secant residual")
                check(float(item["secant_curvature"]) > 0 and float(item["model_curvature"]) > 0,
                      f"{case}: curvature")
                max_secant_residual = max(max_secant_residual, residual)
                phi = float(item["phi"])
                if row["variant"] == "adaptive":
                    check(0.1 <= phi <= 0.9, f"{case}: adaptive parameter outside its bounds")
                elif method.family == "broyden":
                    check(phi == settings["broyden_phi"], f"{case}: constant phi")
                else:
                    check(phi == 0.0, f"{case}: BFGS phi")
    check(json.loads(row["update_status_counts"]) == dict(updates), f"{case}: update event accounting")
    check(int(row["quasi_newton_updates"]) == updates["accepted"], f"{case}: accepted updates")
    check(int(row["skipped_updates"]) == sum(updates.values()) - updates["accepted"], f"{case}: skipped updates")
    if method.family == "newton":
        attempts = int(row["hessian_evaluations"])
        check(attempts >= iterations, f"{case}: missing Hessian evaluation")
        active = "matrix_inversions" if method.direction == "inverse" else "linear_solves"
        inactive = "linear_solves" if active == "matrix_inversions" else "matrix_inversions"
        check(int(row[inactive]) == 0, f"{case}: direction routine mismatch")
        if converged:
            check(attempts == int(row[active]) == iterations, f"{case}: Newton operation count")
    else:
        check(int(row["hessian_evaluations"]) == int(row["matrix_inversions"]) == 0, f"{case}: analytic Hessian use")
        if method.family != "gradient":
            check(sum(updates.values()) == iterations, f"{case}: each accepted step needs an update event")
            check(int(row["linear_solves"]) >= iterations, f"{case}: quasi-Newton solve count")
    if converged:
        check(float(row["gradient_norm"]) < settings["tolerance"], f"{case}: strict stopping rule")
        mu = 2.0 if row["function"] == "f1" else 2.0 - math.sqrt(2.0)
        check(float(row["point_error"]) <= float(row["gradient_norm"]) / mu + 2e-14, f"{case}: strong convexity bound")
        expected_f = 2 + total_line_calls if method.family == "gradient" else 1 + total_line_calls + iterations
        check(int(row["function_evaluations"]) == expected_f, f"{case}: objective count")
        check(int(row["gradient_evaluations"]) == iterations + 1, f"{case}: gradient count")
    return max_line_alpha_error, max_direction_residual, max_secant_residual


def main():
    parser = argparse.ArgumentParser(description="Verify stored comparison runs and timings.")
    parser.add_argument("--results", type=Path, default=ROOT / "comparison_results")
    parser.add_argument("--check-only", action="store_true", help="Validate without writing validation.json.")
    args = parser.parse_args()
    path = args.results
    meta = json.loads((path / "experiment.json").read_text(encoding="utf-8"))
    main_rows = read_csv(path / "summary.csv")
    adaptive = read_csv(path / "adaptive_summary.csv") if meta["adaptive_runs"] else []
    settings = meta["settings"]
    refs = reference_solutions()
    for key, reference in refs.items():
        check(np.allclose(meta["reference_solutions"][key]["point"], reference["point"], atol=1e-14, rtol=0),
              f"Independent reference mismatch for {key}")
    expected = {(f, f"s{i}", m) for f in meta["functions"] for m in meta["methods"]
                for i in range(1, len(meta["initial_points"]) + 1)}
    actual = {(r["function"], r["start_id"], r["method"]) for r in main_rows}
    check(len(main_rows) == len(expected) == meta["primary_runs"] and expected == actual,
          "Main results do not cover exactly the configured cases.")
    check(len(adaptive) == meta["adaptive_runs"], "Adaptive count mismatch.")
    if adaptive:
        check(len(adaptive) == len({r["case"] for r in adaptive})
              == sum(r["method"].startswith("broyden_") for r in main_rows),
              "Adaptive cases must be paired with the constant Broyden runs.")
    rows = main_rows + adaptive
    histories = {r["case"]: read_csv(path / r["history_csv"]) for r in rows}
    timings = defaultdict(list)
    for record in read_csv(path / "benchmark_times.csv"):
        timings[record["case"]].append(float(record["elapsed_seconds"]))
    check(set(timings) == set(histories), "Benchmark case set mismatch.")
    check({r["case"] for r in read_csv(path / "convergence_rates.csv")} == set(histories), "Rate table case set mismatch.")
    maxima = np.zeros(3)
    for row in rows:
        case = row["case"]
        start = meta["initial_points"][int(row["start_id"][1:]) - 1]
        check([float(row["initial_x"]), float(row["initial_y"])] == start, f"{case}: expected experiment start")
        maxima = np.maximum(maxima, check_trace(row, histories[case], settings, refs[row["function"]]))
        samples = timings[case]
        check(len(samples) == int(row["repeats"]) == meta["benchmark_repeats"], f"{case}: count of timing samples")
        check(math.isclose(1000 * median(samples), float(row["median_ms"]), rel_tol=1e-13), f"{case}: median timing")
    check(maxima[1] < 1e-5, "Direction solve residual too large.")
    rows_by_method = {(r["function"], r["start_id"], r["method"]): r for r in main_rows}
    pair_distances = []
    for (function, start, method), item in rows_by_method.items():
        if not method.startswith("newton_inverse"):
            continue
        paired = rows_by_method.get((function, start, method.replace("inverse", "solve")))
        if paired:
            check(item["status"] == paired["status"] and item["iterations"] == paired["iterations"],
                  f"{item['case']}: Newton methods disagree")
            a = np.array([[float(r["x"]), float(r["y"])] for r in histories[item["case"]]])
            b = np.array([[float(r["x"]), float(r["y"])] for r in histories[paired["case"]]])
            distance = float(np.max(np.linalg.norm(a - b, axis=1)))
            check(distance <= 1e-7, f"{item['case']}: Newton trajectories disagree")
            pair_distances.append(distance)
    validation = {
        "passed": True,
        "cases_checked": len(rows),
        "primary_cases": len(main_rows),
        "adaptive_cases": len(adaptive),
        "timed_samples_checked": sum(map(len, timings.values())),
        "newton_pairs_checked": len(pair_distances),
        "max_newton_trajectory_distance": max(pair_distances, default=None),
        "max_closed_line_alpha_error": float(maxima[0]),
        "max_relative_direction_residual": float(maxima[1]),
        "max_relative_secant_residual": float(maxima[2]),
        "statuses": dict(Counter(r["status"] for r in main_rows)),
    }
    if not args.check_only:
        write_json(path / "validation.json", validation)
    print(f"PASS: {len(rows)} exported traces, {sum(map(len, timings.values()))} timing samples and "
          f"{len(pair_distances)} paired Newton comparisons.")
    print("Main statuses:", dict(Counter(r["status"] for r in main_rows)))
    print(f"Maximum closed line-step disagreement: {maxima[0]:.3e}")


if __name__ == "__main__":
    main()
