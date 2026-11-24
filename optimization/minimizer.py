from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np

from .line_search import golden_section
from .newton import MatrixInfo, OptimizationFailure, check_positive_matrix, newton_direction
from .quasi_newton import broyden_update
from .solvers import Oracle, Settings, solve


@dataclass(frozen=True)
class Method:
    label: str
    short: str
    family: str
    direction: str
    step: str


METHODS = {
    "division": Method("Gradient descent with step division", "GD division", "gradient", "negative_gradient", "division"),
    "golden": Method("Gradient descent with Golden Section", "GD golden", "gradient", "negative_gradient", "golden_open"),
    "newton_inverse_fixed": Method("Newton inverse with unit step", "N inverse 1", "newton", "inverse", "unit"),
    "newton_inverse_golden": Method("Newton inverse with Golden Section", "N inverse G", "newton", "inverse", "golden_closed"),
    "newton_solve_fixed": Method("Newton solve with unit step", "N solve 1", "newton", "solve", "unit"),
    "newton_solve_golden": Method("Newton solve with Golden Section", "N solve G", "newton", "solve", "golden_closed"),
    "bfgs_fixed": Method("BFGS with unit step", "BFGS 1", "bfgs", "approximate_hessian", "unit"),
    "bfgs_golden": Method("BFGS with Golden Section", "BFGS G", "bfgs", "approximate_hessian", "golden_closed"),
    "broyden_fixed": Method("Broyden class with unit step", "Broyden 1", "broyden", "approximate_hessian", "unit"),
    "broyden_golden": Method("Broyden class with Golden Section", "Broyden G", "broyden", "approximate_hessian", "golden_closed"),
}


@dataclass(frozen=True)
class OptimizationSettings(Settings):
    line_lower: float = 1e-8
    line_upper: float = 1.0
    matrix_rcond: float = 1e-12
    curvature_tolerance: float = 1e-12
    broyden_phi: float = 0.5
    adaptive_phi: bool = False

    def validate(self):
        super().validate()
        if not (0.0 < self.line_lower < self.line_upper <= 1.0):
            raise ValueError("The closed line interval must satisfy 0 < lower < upper <= 1.")
        for name in ("matrix_rcond", "curvature_tolerance"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if not 0.0 <= self.broyden_phi <= 1.0:
            raise ValueError("broyden_phi must be in [0, 1].")


@dataclass
class Counts:
    function_evaluations: int = 0
    gradient_evaluations: int = 0
    hessian_evaluations: int = 0
    matrix_inversions: int = 0
    linear_solves: int = 0
    matrix_checks: int = 0
    quasi_newton_updates: int = 0
    skipped_updates: int = 0
    nonfinite_line_trials: int = 0


@dataclass(frozen=True)
class Iteration:
    point: tuple
    value: float
    gradient: tuple
    gradient_norm: float
    counts: Counts
    alpha: float | None = None
    direction: tuple | None = None
    line_location: str = ""
    line_iterations: int = 0
    line_width: float | None = None
    backtracks: int = 0
    objective_change: float | None = None
    previous_line_derivative: float | None = None
    relative_direction_residual: float | None = None
    matrix_min_eigenvalue: float | None = None
    matrix_rcond: float | None = None
    update_status: str = ""
    phi: float | None = None
    curvature: float | None = None
    model_curvature: float | None = None
    secant_residual: float | None = None


@dataclass(frozen=True)
class OptimizationResult:
    objective: str
    method: str
    settings: OptimizationSettings
    status: str
    detail: str
    elapsed_seconds: float
    history: tuple
    counts: Counts

    @property
    def iterations(self):
        return len(self.history) - 1


def _first_order(objective, method, settings):
    result = solve(objective, method, settings)
    history = []
    for row in result.history:
        dimensions = len(settings.initial_point)
        point = (row["x"], row["y"]) if dimensions == 2 else row["point"]
        gradient = (row["gradient_x"], row["gradient_y"]) if dimensions == 2 else row["gradient"]
        previous = history[-1] if history else None
        history.append(Iteration(
            point, row["value"], gradient, row["gradient_norm"],
            Counts(row["function_evaluations"], row["gradient_evaluations"]),
            alpha=row["alpha_from_previous"] if previous else None,
            direction=tuple(-value for value in previous.gradient) if previous else None,
            line_location=METHODS[method].step if previous else "",
            line_iterations=row["line_iterations_from_previous"],
            line_width=row["line_width_from_previous"],
            backtracks=row["backtracks_from_previous"],
            objective_change=row["objective_change_from_previous"] if previous else None,
            previous_line_derivative=row["previous_line_derivative_at_current"],
        ))
    return OptimizationResult(result.objective, method, settings, result.status, "",
                              result.elapsed_seconds, tuple(history), history[-1].counts)


def _check_state(point, value, gradient):
    if not np.all(np.isfinite(point)):
        raise OptimizationFailure("numerical_failure", "The trial point is non-finite.")
    if not math.isfinite(value):
        raise OptimizationFailure("nonfinite_objective", "The objective value is non-finite.")
    if gradient.shape != point.shape:
        raise OptimizationFailure("invalid_gradient", "The gradient shape does not match the point.")
    if not (np.all(np.isfinite(gradient)) and math.isfinite(float(np.linalg.norm(gradient)))):
        raise OptimizationFailure("nonfinite_gradient", "The gradient or its norm is non-finite.")


def _line_step(oracle, point, value, gradient, direction, settings, counts):
    def evaluate(alpha):
        try:
            change = float(oracle.change(point, alpha * direction, gradient, value))
        except ArithmeticError:
            change = math.inf
        if not math.isfinite(change):
            counts.nonfinite_line_trials += 1
            return math.inf
        return change

    line = golden_section(evaluate, settings.line_lower, settings.line_upper,
                          settings.line_tolerance, settings.max_line_iterations,
                          include_endpoints=True)
    if not line.converged:
        raise OptimizationFailure("line_search_limit", "The Golden Section bracket did not reach its target width.")
    if not math.isfinite(line.value):
        raise OptimizationFailure("nonfinite_objective", "Every tested line value was non-finite.")
    if line.value >= 0.0:
        raise OptimizationFailure("line_search_no_decrease", "The closed interval supplied no decreasing step.")
    locations = {settings.line_lower: "lower", settings.line_upper: "upper"}
    return line.alpha, line.value, locations.get(line.alpha, "interior"), line.iterations, line.width


def _second_order(objective, method, settings):
    spec = METHODS[method]
    counts = Counts()
    oracle = Oracle(objective)
    x = np.array(settings.initial_point, dtype=float)
    quasi = spec.family in ("bfgs", "broyden")
    matrix = np.eye(x.size) if quasi else None
    info = MatrixInfo(1.0, 1.0) if quasi else None
    history = []
    start = perf_counter()

    def snapshot():
        return replace(counts, function_evaluations=oracle.function_evaluations,
                       gradient_evaluations=oracle.gradient_evaluations)

    def read_value(point):
        try:
            return float(oracle.value(point))
        except ArithmeticError:
            return math.nan

    def read_gradient(point):
        try:
            return np.asarray(oracle.gradient(point), dtype=float)
        except ArithmeticError:
            return np.full_like(point, math.nan)

    value, gradient = read_value(x), read_gradient(x)
    initial_gradient = tuple(gradient) if gradient.shape == x.shape else ()
    initial_norm = float(np.linalg.norm(gradient)) if initial_gradient else math.nan
    history.append(Iteration(tuple(x), value, initial_gradient, initial_norm, snapshot()))

    try:
        _check_state(x, value, gradient)
        for iteration in range(settings.max_iterations + 1):
            if history[-1].gradient_norm < settings.tolerance:
                status, detail = "converged", ""
                break
            if iteration == settings.max_iterations:
                status, detail = "max_iterations", "The outer iteration limit was reached."
                break
            if not quasi:
                hessian = getattr(objective, "hessian", None)
                if hessian is None:
                    raise OptimizationFailure("missing_hessian", "Newton requires an analytical Hessian callable.")
                counts.hessian_evaluations += 1
                matrix = np.asarray(hessian(x), dtype=float)
                counts.matrix_checks += 1
                info = check_positive_matrix(matrix, x.size, settings.matrix_rcond)
            approach = "solve" if quasi else spec.direction
            if approach == "inverse":
                counts.matrix_inversions += 1
            else:
                counts.linear_solves += 1
            direction = newton_direction(matrix, gradient, approach)
            slope = float(gradient @ direction)
            if not (np.all(np.isfinite(direction)) and math.isfinite(slope) and slope < 0.0):
                raise OptimizationFailure("invalid_search_direction", "The direction is not finite and strictly descending.")
            residual = float(np.linalg.norm(matrix @ direction + gradient) / history[-1].gradient_norm)
            used_info = info
            if spec.step == "unit":
                alpha, change, location, contractions, width = 1.0, None, "unit", 0, None
            else:
                alpha, change, location, contractions, width = _line_step(
                    oracle, x, value, gradient, direction, settings, counts,
                )
            next_point = x + alpha * direction
            if np.array_equal(next_point, x):
                raise OptimizationFailure("floating_point_stagnation", "The trial point rounded to the current point.")
            next_value = read_value(next_point)
            if not math.isfinite(next_value):
                raise OptimizationFailure("nonfinite_objective", "The trial objective value is non-finite.")
            next_gradient = read_gradient(next_point)
            _check_state(next_point, next_value, next_gradient)
            update = None
            if quasi:
                phi = settings.broyden_phi if spec.family == "broyden" else 0.0
                update = broyden_update(matrix, next_point - x, next_gradient - gradient, phi,
                                        adaptive=spec.family == "broyden" and settings.adaptive_phi,
                                        curvature_tolerance=settings.curvature_tolerance,
                                        matrix_rcond=settings.matrix_rcond)
                counts.matrix_checks += update.matrix_checked
                if update.matrix is None:
                    counts.skipped_updates += 1
                else:
                    counts.quasi_newton_updates += 1
                    matrix, info = update.matrix, update.info
            history.append(Iteration(
                tuple(next_point), next_value, tuple(next_gradient),
                float(np.linalg.norm(next_gradient)), snapshot(), alpha,
                direction=tuple(direction), line_location=location, line_iterations=contractions,
                line_width=width, objective_change=next_value - value if change is None else change,
                previous_line_derivative=float(next_gradient @ direction),
                relative_direction_residual=residual,
                matrix_min_eigenvalue=used_info.min_eigenvalue,
                matrix_rcond=used_info.reciprocal_condition,
                update_status=update.status if update else "", phi=update.phi if update else None,
                curvature=update.curvature if update else None,
                model_curvature=update.model_curvature if update else None,
                secant_residual=update.residual if update else None,
            ))
            x, value, gradient = next_point, next_value, next_gradient
    except OptimizationFailure as error:
        status, detail = error.status, str(error)
    except ArithmeticError as error:
        status, detail = "numerical_failure", str(error)
    return OptimizationResult(objective.key, method, settings, status, detail,
                              perf_counter() - start, tuple(history), snapshot())


def minimize(objective, method, settings=None):
    """Run one registered configuration on a differentiable objective and a finite vector."""
    if method not in METHODS:
        raise ValueError(f"Unknown optimization method: {method}")
    settings = OptimizationSettings() if settings is None else settings
    if not isinstance(settings, OptimizationSettings):
        raise TypeError("minimize expects OptimizationSettings.")
    settings.validate()
    if METHODS[method].family == "gradient":
        return _first_order(objective, method, settings)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        return _second_order(objective, method, settings)
