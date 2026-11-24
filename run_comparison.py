import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys

import numpy as np

from optimization import METHODS, OBJECTIVES, OptimizationSettings, minimize
from optimization.analysis import convergence_rows
from optimization.exports import write_csv, write_json
from optimization.reference import reference_solutions


ROOT = Path(__file__).resolve().parent


def run_case(method, function, start_id, settings, references, repeats, local_radius, output):
    objective = OBJECTIVES[function]
    method_spec = METHODS[method]
    variant = "adaptive" if settings.adaptive_phi else "main"
    case = f"{function}_{start_id}_{method}" + ("_adaptive" if settings.adaptive_phi else "")
    result = minimize(objective, method, settings)
    signature = (result.status, result.iterations, result.counts)
    minimize(objective, method, settings)
    times = []
    for _ in range(repeats):
        trial = minimize(objective, method, settings)
        if (trial.status, trial.iterations, trial.counts) != signature:
            raise RuntimeError(f"Inconsistent repeated benchmark for {case}.")
        times.append(trial.elapsed_seconds)
    trace = convergence_rows(result, references[function], local_radius)
    filename = f"histories/{case}.csv"
    write_csv(output / filename, trace)
    last = trace[-1]
    updates = [entry for entry in result.history if entry.update_status]
    statuses = Counter(entry.update_status for entry in updates)
    alphas = Counter(entry.line_location for entry in result.history[1:])
    row = {
        "case": case,
        "function": function,
        "start_id": start_id,
        "method": method,
        "algorithm": method_spec.label,
        "direction": method_spec.direction,
        "step_strategy": method_spec.step,
        "variant": variant,
        "initial_x": settings.initial_point[0],
        "initial_y": settings.initial_point[1],
        "final_x": last["x"],
        "final_y": last["y"],
        "objective": last["objective"],
        "gradient_norm": last["gradient_norm"],
        "point_error": last["point_error"],
        "objective_gap": last["objective_gap"],
        "iterations": result.iterations,
        **asdict(result.counts),
        "objective_increases": sum(a.value < b.value for a, b in zip(result.history, result.history[1:])),
        "upper_endpoint_steps": alphas["upper"],
        "lower_endpoint_steps": alphas["lower"],
        "update_status_counts": json.dumps(dict(sorted(statuses.items())), separators=(",", ":")),
        "elapsed_seconds": result.elapsed_seconds,
        "median_ms": 1000 * median(times),
        "min_ms": 1000 * min(times),
        "max_ms": 1000 * max(times),
        "repeats": repeats,
        "status": result.status,
        "detail": result.detail,
        "history_csv": filename,
    }
    raw_times = [{"case": case, "variant": variant, "repeat": i, "elapsed_seconds": seconds}
                 for i, seconds in enumerate(times, 1)]
    return row, raw_times, trace


def rate_table(rows, traces):
    records = []
    for row in rows:
        local = [step for step in traces[row["case"]] if step["local_order"] is not None]
        last = local[-1] if local else {}
        data = {
            "case": row["case"],
            "function": row["function"],
            "start_id": row["start_id"],
            "method": row["method"],
            "status": row["status"],
            "local_order_observations": len(local),
            "last_order_iteration": last.get("iteration"),
            "last_local_order": last.get("local_order"),
            "last_local_error_ratio": last.get("error_ratio"),
            "last_local_quadratic_ratio": last.get("quadratic_ratio"),
        }
        records.append(data)
    return records


def paired_quasi_table(main, adaptive):
    constants = {(r["function"], r["start_id"], r["method"]): r for r in main}
    pairs = []
    for probe in adaptive:
        key = (probe["function"], probe["start_id"], probe["method"])
        fixed = constants[key]
        row = {k: probe[k] for k in ("function", "start_id", "method", "initial_x", "initial_y")}
        for label, source in (("constant", fixed), ("adaptive", probe)):
            row.update({label + "_" + k: source[k] for k in (
                "status", "iterations", "function_evaluations", "gradient_norm",
                "point_error", "median_ms", "skipped_updates",
            )})
        pairs.append(row)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Compare first-order, Newton and quasi-Newton configurations.")
    parser.add_argument("--config", type=Path, default=ROOT / "comparison_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "comparison_results")
    parser.add_argument("--function", choices=("all", *OBJECTIVES), default="all")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--x0", type=float, nargs=2)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--line-lower", type=float)
    parser.add_argument("--line-upper", type=float)
    parser.add_argument("--line-tolerance", type=float)
    parser.add_argument("--max-line-iterations", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--broyden-phi", type=float)
    parser.add_argument("--benchmark-repeats", type=int)
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-iterates", action="store_true")
    args = parser.parse_args()
    original = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    comparison = json.loads(args.config.read_text(encoding="utf-8"))
    solver_values = original | comparison["solver"]
    solver_values.update({
        key: getattr(args, key) for key in (
            "tolerance", "line_lower", "line_upper", "line_tolerance",
            "max_line_iterations", "max_iterations", "broyden_phi",
        ) if getattr(args, key) is not None
    })
    solver_values["initial_point"] = tuple(solver_values["initial_point"])
    settings = OptimizationSettings(**solver_values)
    settings.validate()
    repeats = comparison["benchmark_repeats"] if args.benchmark_repeats is None else args.benchmark_repeats
    local_radius = comparison["local_error_radius"]
    if not isinstance(repeats, int) or repeats < 1:
        parser.error("Benchmark repeats must be a positive integer.")
    if not math.isfinite(local_radius) or local_radius <= 0.0:
        parser.error("Local error radius must be positive and finite.")
    points = [tuple(args.x0)] if args.x0 is not None else [tuple(point) for point in comparison["initial_points"]]
    if not points or any(len(point) != 2 for point in points):
        parser.error("The comparison requires at least one two-coordinate starting point.")
    for point in points:
        replace(settings, initial_point=point).validate()
    functions = list(OBJECTIVES) if args.function == "all" else [args.function]
    methods = list(METHODS) if args.method == "all" else [args.method]
    refs = reference_solutions()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    main_rows, adaptive_rows, time_rows = [], [], []
    traces = {}
    for function in functions:
        for index, point in enumerate(points, 1):
            start_id = f"s{index}"
            for method in methods:
                policies = (False, True) if method.startswith("broyden_") and not args.no_adaptive else (False,)
                for adapt in policies:
                    case_settings = replace(settings, initial_point=point, adaptive_phi=adapt)
                    summary, timings, trace = run_case(
                        method, function, start_id, case_settings, refs, repeats, local_radius, output,
                    )
                    (adaptive_rows if adapt else main_rows).append(summary)
                    time_rows.extend(timings)
                    traces[summary["case"]] = trace
                    print(f"{summary['case']:38s} {summary['status']:22s} k={summary['iterations']:3d} "
                          f"Nf={summary['function_evaluations']:4d} Ng={summary['gradient_evaluations']:3d} "
                          f"Nh={summary['hessian_evaluations']:3d}")
                    if args.show_iterates:
                        for entry in trace:
                            print(f"  X_{entry['iteration']:<3d} ({entry['x']:.10g}, {entry['y']:.10g})")
    write_csv(output / "summary.csv", main_rows)
    write_csv(output / "adaptive_summary.csv", adaptive_rows)
    write_csv(output / "adaptive_comparison.csv", paired_quasi_table(main_rows, adaptive_rows))
    write_csv(output / "benchmark_times.csv", time_rows)
    write_csv(output / "convergence_rates.csv", rate_table(main_rows + adaptive_rows, traces))
    write_json(output / "experiment.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__},
        "settings": asdict(settings),
        "initial_points": points,
        "functions": functions,
        "methods": methods,
        "reference_solutions": refs,
        "benchmark_repeats": repeats,
        "local_error_radius": local_radius,
        "point_error_reliability_floor": "64 machine epsilons times max(1, norm(reference))",
        "primary_runs": len(main_rows),
        "adaptive_runs": len(adaptive_rows),
        "counting": ("Objective values and stable objective increments each count once. The legacy solvers "
                     "accumulate increments and verify the final value. The new solvers evaluate the objective "
                     "directly at each accepted point. Closed Golden Section counts both endpoint trials. "
                     "Matrix checks, Hessians, inversions and solves count attempts. Reference, plotting and "
                     "analysis work are outside these counters."),
        "timing": ("Single process; one warm-up per case then repeated solver calls. "
                   "The solver timer includes history and safeguards. It excludes reference calculations, "
                   "reformatting the legacy traces, plots and file I/O."),
    })
    if not args.no_plots:
        from optimization.comparison_plots import make_plots
        make_plots(main_rows, adaptive_rows, traces, refs, settings.tolerance, output / "figures")
    converged = sum(row["status"] == "converged" for row in main_rows)
    print(f"Main: {converged}/{len(main_rows)} converged. Adaptive sensitivity runs: {len(adaptive_rows)}.")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
