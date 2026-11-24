from concurrent.futures import CancelledError
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from uuid import uuid4

from ..analysis import convergence_rows, finite
from ..exports import write_csv, write_json
from ..minimizer import METHODS, OptimizationResult, OptimizationSettings, minimize
from ..objectives import Objective
from ..reference import reference_for_objective


COEFFICIENTS = ("a", "b", "c", "d", "k", "m")
DEFAULT_SETTINGS = OptimizationSettings(initial_point=(0.0, 0.0))

METHOD_PRESETS = {
    "Gradient descent pair": ("division", "golden"),
    "Newton methods": tuple(key for key in METHODS if key.startswith("newton")),
    "BFGS pair": ("bfgs_fixed", "bfgs_golden"),
    "Broyden pair": ("broyden_fixed", "broyden_golden"),
    "Compare all methods": tuple(METHODS),
}


@dataclass(frozen=True)
class Experiment:
    objective: Objective
    settings: OptimizationSettings
    methods: tuple[str, ...]
    repeats: int = 1
    compare_adaptive: bool = False
    local_radius: float = 0.01

    def validate(self):
        self.settings.validate()
        if len(self.settings.initial_point) != 2:
            raise ValueError("Enter two coordinates for the starting point.")
        for name in COEFFICIENTS:
            if not math.isfinite(getattr(self.objective, name)):
                raise ValueError(f"Coefficient {name} must be finite.")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("Select at least one method and select each method once.")
        if any(key not in METHODS for key in self.methods):
            raise ValueError("Choose methods from the available optimizers.")
        if not isinstance(self.repeats, int) or self.repeats < 1:
            raise ValueError("Timing repeats must be a positive integer.")
        if not math.isfinite(self.local_radius) or self.local_radius <= 0:
            raise ValueError("Local error radius must be positive and finite.")

    def configurations(self):
        for key in self.methods:
            yield key, self.settings
            if self.compare_adaptive and METHODS[key].family == "broyden":
                yield key, replace(self.settings, adaptive_phi=True)


@dataclass(frozen=True)
class MethodRun:
    result: OptimizationResult
    trace: tuple[dict, ...]
    timings: tuple[float, ...]

    @property
    def adaptive(self):
        return self.result.settings.adaptive_phi and METHODS[self.result.method].family == "broyden"

    @property
    def id(self):
        return self.result.method + ("_adaptive" if self.adaptive else "")

    @property
    def short(self):
        return METHODS[self.result.method].short + (" adaptive" if self.adaptive else "")

    @property
    def median_ms(self):
        return 1000 * median(self.timings)

    def summary(self):
        last = self.result.history[-1]
        point_x, point_y = last.point
        return {
            "method": self.result.method,
            "variant": "adaptive" if self.adaptive else "standard",
            "label": METHODS[self.result.method].label,
            "status": self.result.status,
            "detail": self.result.detail,
            "iterations": self.result.iterations,
            "final_x": finite(point_x),
            "final_y": finite(point_y),
            "objective": finite(last.value),
            "gradient_norm": finite(last.gradient_norm),
            "point_error": self.trace[-1]["point_error"],
            **asdict(self.result.counts),
            "median_ms": self.median_ms,
            "timing_repeats": len(self.timings),
            "history_csv": f"histories/{self.id}.csv",
        }


@dataclass(frozen=True)
class Session:
    experiment: Experiment
    reference: dict | None
    runs: tuple[MethodRun, ...]


def execute(experiment, progress=None, cancelled=None):
    experiment.validate()
    reference = reference_for_objective(experiment.objective)
    configurations = list(experiment.configurations())
    runs = []

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise CancelledError()

    for index, (key, settings) in enumerate(configurations):
        check_cancelled()
        if progress is not None:
            progress(index, len(configurations), METHODS[key].label)
        result = minimize(experiment.objective, key, settings)
        timings = []
        if experiment.repeats == 1:
            timings.append(result.elapsed_seconds)
        else:
            signature = result.status, result.iterations, result.counts
            for _ in range(experiment.repeats):
                check_cancelled()
                trial = minimize(experiment.objective, key, settings)
                if (trial.status, trial.iterations, trial.counts) != signature:
                    raise RuntimeError(f"Inconsistent timing run for {METHODS[key].label}.")
                timings.append(trial.elapsed_seconds)
        trace = convergence_rows(result, reference, experiment.local_radius)
        runs.append(MethodRun(result, tuple(trace), tuple(timings)))
    check_cancelled()
    return Session(experiment, reference, tuple(runs))


def export_session(session, directory, figures):
    """Write the session to a new directory once all export files are ready."""
    directory = Path(directory)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"optimization-{timestamp}-{uuid4().hex[:6]}"
    with TemporaryDirectory(prefix=".optimization-export-", dir=directory) as temporary:
        output = Path(temporary) / name
        output.mkdir()
        write_json(output / "experiment.json", {
            "experiment": asdict(session.experiment),
            "reference": session.reference,
        })
        write_csv(output / "summary.csv", [run.summary() for run in session.runs])
        write_csv(output / "benchmark_times.csv", [
            {"method": run.id, "repeat": index, "elapsed_seconds": elapsed}
            for run in session.runs for index, elapsed in enumerate(run.timings, 1)
        ])
        for run in session.runs:
            write_csv(output / run.summary()["history_csv"], run.trace)
        plot_dir = output / "figures"
        plot_dir.mkdir()
        for name_stem, figure in figures.items():
            figure.savefig(plot_dir / f"{name_stem}.png", dpi=185, facecolor=figure.get_facecolor())
        result = directory / name
        output.rename(result)
    return result
