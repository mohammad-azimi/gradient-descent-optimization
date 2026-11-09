from dataclasses import asdict, dataclass
from time import perf_counter
import math

import numpy as np

from .line_search import golden_section


METHOD_NAMES = {
    "division": "Step Division",
    "golden": "One-Dimensional Minimization",
}


@dataclass(frozen=True)
class Settings:
    initial_point: tuple = (2.0, 2.0)
    initial_step: float = 0.8
    reduction_factor: float = 0.5
    armijo_epsilon: float = 0.1
    tolerance: float = 1e-6
    max_iterations: int = 10000
    line_tolerance: float = 1e-10
    max_line_iterations: int = 200
    max_backtracks: int = 100

    def validate(self):
        point = np.asarray(self.initial_point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("The initial point must contain two finite coordinates.")
        for key in ["initial_step", "reduction_factor", "armijo_epsilon"]:
            value = getattr(self, key)
            if not (math.isfinite(value) and 0 < value < 1):
                raise ValueError(f"{key} must belong to the lecture interval (0, 1).")
        for key in ["tolerance", "line_tolerance"]:
            value = getattr(self, key)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{key} must be positive and finite.")
        for key in ["max_iterations", "max_line_iterations", "max_backtracks"]:
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")


class Oracle:
    def __init__(self, objective):
        self.objective = objective
        self.function_evaluations = 0
        self.gradient_evaluations = 0

    def value(self, point):
        self.function_evaluations += 1
        return self.objective.value(point)

    def gradient(self, point):
        self.gradient_evaluations += 1
        return self.objective.gradient(point)

    def change(self, point, displacement, gradient):
        self.function_evaluations += 1
        return self.objective.change(point, displacement, gradient)


@dataclass
class Result:
    objective: str
    method: str
    settings: dict
    status: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    elapsed_seconds: float
    history: list

    def summary(self):
        last = self.history[-1]
        return {
            "function": self.objective,
            "method": METHOD_NAMES[self.method],
            "initial_x": self.settings["initial_point"][0],
            "initial_y": self.settings["initial_point"][1],
            "final_x": last["x"],
            "final_y": last["y"],
            "minimum_value": last["value"],
            "gradient_norm": last["gradient_norm"],
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
        }


def solve(objective, method, settings=None):
    settings = Settings() if settings is None else settings
    settings.validate()
    if method not in METHOD_NAMES:
        raise ValueError("Choose division or golden.")
    start = perf_counter()
    oracle = Oracle(objective)
    x = np.asarray(settings.initial_point, dtype=float).copy()
    value = oracle.value(x)
    gradient = oracle.gradient(x)
    history = []
    previous_gradient = None
    status = "max_iterations"
    alpha = 0.0
    line_iterations = 0
    line_width = 0.0
    backtracks = 0
    change = 0.0
    step_norm = 0.0

    def record(k):
        norm = float(np.linalg.norm(gradient))
        if previous_gradient is None:
            dot = cosine = angle = derivative = None
        else:
            dot = float(np.dot(previous_gradient, gradient))
            denominator = float(np.linalg.norm(previous_gradient)) * norm
            cosine = float(np.clip(dot/denominator, -1, 1)) if denominator > 0 else None
            angle = float(np.degrees(np.arccos(cosine))) if cosine is not None else None
            derivative = -dot
        history.append({
            "iteration": k, "x": float(x[0]), "y": float(x[1]),
            "value": value, "gradient_x": float(gradient[0]),
            "gradient_y": float(gradient[1]), "gradient_norm": norm,
            "alpha_from_previous": alpha, "step_norm_from_previous": step_norm,
            "objective_change_from_previous": change,
            "backtracks_from_previous": backtracks,
            "line_iterations_from_previous": line_iterations,
            "line_width_from_previous": line_width,
            "gradient_dot_previous": dot, "gradient_cosine_previous": cosine,
            "gradient_angle_degrees_previous": angle,
            "previous_line_derivative_at_current": derivative,
            "function_evaluations": oracle.function_evaluations,
            "gradient_evaluations": oracle.gradient_evaluations,
        })

    record(0)
    for k in range(settings.max_iterations):
        if not (math.isfinite(value) and np.all(np.isfinite(gradient))):
            status = "nonfinite_state"
            break
        if float(np.linalg.norm(gradient)) < settings.tolerance:
            status = "converged"
            break
        direction = -gradient
        squared_norm = float(np.dot(gradient, gradient))
        if not math.isfinite(squared_norm):
            status = "nonfinite_state"
            break
        if method == "division":
            alpha = settings.initial_step
            backtracks = 0
            line_iterations = 0
            line_width = 0.0
            for attempt in range(settings.max_backtracks + 1):
                change = oracle.change(x, alpha*direction, gradient)
                if math.isfinite(change) and change <= -settings.armijo_epsilon*alpha*squared_norm:
                    break
                if attempt == settings.max_backtracks:
                    status = "backtracking_limit"
                    break
                alpha *= settings.reduction_factor
                backtracks += 1
            if status == "backtracking_limit":
                break
        else:
            line = golden_section(
                lambda a: oracle.change(x, a*direction, gradient),
                0.0, 1.0, settings.line_tolerance, settings.max_line_iterations,
            )
            alpha, change = line.alpha, line.value
            line_iterations, line_width = line.iterations, line.width
            backtracks = 0
            if not line.converged:
                status = "line_search_limit"
                break
            if not (math.isfinite(change) and change < 0):
                status = "line_search_no_decrease"
                break
        next_x = x+alpha*direction
        if np.array_equal(next_x, x):
            status = "floating_point_stagnation"
            break
        previous_gradient = gradient.copy()
        step_norm = float(np.linalg.norm(next_x-x))
        x = next_x
        value = float(value + change)
        gradient = oracle.gradient(x)
        record(k+1)
    if np.all(np.isfinite(gradient)) and float(np.linalg.norm(gradient)) < settings.tolerance:
        status = "converged"
    value = oracle.value(x)
    history[-1]["value"] = value
    history[-1]["function_evaluations"] = oracle.function_evaluations
    history[-1]["gradient_evaluations"] = oracle.gradient_evaluations
    elapsed = perf_counter()-start
    return Result(objective.key, method, asdict(settings), status, len(history)-1,
                  oracle.function_evaluations, oracle.gradient_evaluations, elapsed, history)
