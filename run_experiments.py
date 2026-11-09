import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import scipy

from optimization import OBJECTIVES, Settings, solve
from optimization.reference import reference_solutions


ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def enhanced_summary(result, references):
    row = result.summary()
    ref = references[result.objective]
    row["point_error"] = float(np.linalg.norm(np.array([row["final_x"], row["final_y"]])-ref["point"]))
    row["absolute_objective_error"] = abs(row["minimum_value"]-ref["value"])
    cosines = [abs(h["gradient_cosine_previous"]) for h in result.history[1:]
               if h["gradient_cosine_previous"] is not None]
    row["max_abs_successive_gradient_cosine"] = max(cosines, default=0.0)
    row["near_upper_boundary_steps"] = sum(h["alpha_from_previous"] > 1-1e-7 for h in result.history[1:])
    return row


def main():
    parser = argparse.ArgumentParser(description="Group 1 unconstrained minimization experiments")
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--function", choices=["all", "f1", "f2"], default="all")
    parser.add_argument("--method", choices=["all", "division", "golden"], default="all")
    parser.add_argument("--x0", type=float, nargs=2)
    parser.add_argument("--initial-step", type=float)
    parser.add_argument("--reduction-factor", type=float)
    parser.add_argument("--armijo-epsilon", type=float)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--line-tolerance", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--benchmark-repeats", type=int, default=31)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ["initial_step", "reduction_factor", "armijo_epsilon", "tolerance", "line_tolerance", "max_iterations"]:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if args.x0 is not None:
        values["initial_point"] = args.x0
    values["initial_point"] = tuple(values["initial_point"])
    settings = Settings(**values)
    settings.validate()
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = reference_solutions()
    functions = list(OBJECTIVES) if args.function == "all" else [args.function]
    methods = ["division", "golden"] if args.method == "all" else [args.method]
    runs, summary, benchmarks = [], [], []
    for key in functions:
        for method in methods:
            objective = OBJECTIVES[key]
            result = solve(objective, method, settings)
            runs.append(result)
            row = enhanced_summary(result, refs)
            write_csv(out/"histories"/f"{key}_{method}.csv", result.history)
            solve(objective, method, settings)
            timings = []
            for repeat in range(args.benchmark_repeats):
                measured = solve(objective, method, settings)
                if (measured.status, measured.iterations, measured.function_evaluations) != (
                        result.status, result.iterations, result.function_evaluations):
                    raise RuntimeError("Benchmark execution was inconsistent with the main run.")
                timings.append(measured.elapsed_seconds)
                benchmarks.append({"function": key, "method": method, "repeat": repeat+1,
                                   "elapsed_seconds": measured.elapsed_seconds})
            row["median_time_ms"] = float(1000*np.median(timings))
            row["min_time_ms"] = float(1000*min(timings))
            row["max_time_ms"] = float(1000*max(timings))
            row["timing_repeats"] = len(timings)
            summary.append(row)
            print(f"{key} {method:8s}  status={result.status:12s}  k={result.iterations:3d} "
                  f" f={row['minimum_value']:.12f}  |g|={row['gradient_norm']:.3e} "
                  f" Nf={result.function_evaluations} Ng={result.gradient_evaluations}")
    write_csv(out/"summary.csv", summary)
    write_csv(out/"benchmark_times.csv", benchmarks)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings), "references": refs, "summary": summary,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__, "scipy": scipy.__version__},
        "function_evaluation_convention": "One call per scalar objective or stable objective-increment evaluation. Includes rejected trials and final direct verification. Excludes plots, references, diagnostics and other runs.",
        "interval": "0 < alpha < 1, as in lecture slide 25",
        "timing_convention": "perf_counter inside solve, including history, excluding I/O, plots, references and diagnostics; one warm-up then repeated solves",
    }
    write_json(out/"experiment.json", metadata)
    if not args.no_sweeps:
        tolerance_rows, step_rows, line_rows, start_rows = [], [], [], []
        for key in functions:
            objective = OBJECTIVES[key]
            for method in methods:
                for tolerance in [1e-4, 1e-6, 1e-8]:
                    row = enhanced_summary(solve(objective, method, replace(settings, tolerance=tolerance)), refs)
                    row["tolerance"] = tolerance
                    tolerance_rows.append(row)
                for point in [(0.0, 0.0), (-2.0, 1.0), (4.0, -3.0), (2.0, 2.0)]:
                    start_rows.append(enhanced_summary(solve(objective, method, replace(settings, initial_point=point)), refs))
            if "division" in methods:
                for step in [0.2, 0.5, 0.8, 0.95]:
                    row = enhanced_summary(solve(objective, "division", replace(settings, initial_step=step)), refs)
                    row["initial_step"] = step
                    step_rows.append(row)
            if "golden" in methods:
                for line_tolerance in [1e-4, 1e-7, 1e-10]:
                    row = enhanced_summary(solve(objective, "golden", replace(settings, line_tolerance=line_tolerance)), refs)
                    row["line_tolerance"] = line_tolerance
                    line_rows.append(row)
        write_csv(out/"tolerance_sweep.csv", tolerance_rows)
        write_csv(out/"step_sweep.csv", step_rows)
        write_csv(out/"line_accuracy_sweep.csv", line_rows)
        write_csv(out/"initial_point_sweep.csv", start_rows)
        direction = np.linalg.eigh(np.array([[1.0, -1.0], [-1.0, 3.0]]))[1][:, 0]
        boundary_start = tuple(np.array([0.5, -0.5])+0.01*direction)
        boundary = solve(OBJECTIVES["f2"], "golden", replace(settings, initial_point=boundary_start))
        write_csv(out/"boundary_example.csv", boundary.history)
        write_json(out/"boundary_example.json", enhanced_summary(boundary, refs))
    if not args.no_plots:
        from optimization.plots import make_plots
        make_plots(runs, refs, out/"figures", settings.tolerance)
    if any(r.status != "converged" for r in runs):
        raise SystemExit("At least one main run did not converge. Inspect the saved status.")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
