import numpy as np
from scipy.optimize import brentq, root
from scipy.special import lambertw, wrightomega

from .objectives import OBJECTIVES


def reference_solutions():
    t = brentq(lambda z: np.log(z) + 0.75*z, 0.1, 1.0, xtol=5e-15)
    x1 = np.array([(1+t)/2, (t-2)/4])
    t_lambert = float((4.0/3.0)*lambertw(0.75).real)
    x2 = np.array([0.5, -0.5])

    def stationary_one(z):
        v = np.exp(-z[0]-z[1])
        return [2*z[0]-v-1, 4*z[1]-v+2]

    def stationary_two(z):
        v = np.sin(z[0]+z[1])
        return [2*z[0]-v-1, 4*z[1]-v+2]

    out = {}
    for key, point, stationarity in [("f1", x1, stationary_one), ("f2", x2, stationary_two)]:
        independent = root(stationarity, [0.0, 0.0], method="hybr", tol=1e-11)
        residual = float(np.linalg.norm(stationarity(independent.x)))
        difference = float(np.linalg.norm(independent.x-point))
        if not (residual < 1e-10 and difference < 1e-10):
            raise RuntimeError(f"Independent reference validation failed for {key}.")
        x, y = point
        v = np.exp(-x-y) if key == "f1" else np.cos(x+y)
        out[key] = {
            "point": point.tolist(), "value": float(x*x+2*y*y+v-x+2*y),
            "root_point": independent.x.tolist(), "root_residual": residual,
            "root_distance": difference, "root_solver_success": bool(independent.success),
        }
    out["f1"].update({"t_brent": float(t), "t_lambert": t_lambert,
                       "scalar_residual": float(abs(np.log(t)+0.75*t))})
    out["f2"]["scalar_residual"] = 0.0
    return out


def reference_for_objective(objective):
    """Solve the scalar stationarity equation when strong convexity can be verified."""
    if objective == OBJECTIVES.get(objective.key):
        return reference_solutions()[objective.key]
    diagonal = 2 * np.array([objective.a, objective.b], dtype=float)
    axis = np.array([objective.c, objective.d], dtype=float)
    linear = np.array([objective.k, objective.m], dtype=float)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center, weighted_axis = -linear / diagonal, axis / diagonal
        curvature, phase = float(axis @ weighted_axis), float(axis @ center)
        if not np.isfinite(curvature) or not np.isfinite(phase):
            return None
        if objective.kind == "cos" and curvature >= 1.0:
            return None
        if curvature == 0:
            point = center
        elif objective.kind == "exp":
            z = wrightomega(np.log(curvature) - phase)
            point = center + weighted_axis * (z / curvature)
        else:
            shift = brentq(lambda q: q - curvature*np.sin(phase + q), -curvature, curvature,
                           xtol=1e-14)
            point = center + weighted_axis * np.sin(phase + shift)
        value = objective.value(point)
        residual = float(np.linalg.norm(objective.gradient(point), ord=np.inf))
        scale = max(1.0, float(np.linalg.norm(linear, ord=np.inf)))
    if np.all(np.isfinite(point)) and np.isfinite(value) and residual <= 1e-8 * scale:
        return {"point": point.tolist(), "value": value, "root_residual": residual}
    return None
