from dataclasses import dataclass
import math

import numpy as np

from .newton import MatrixInfo, OptimizationFailure, check_positive_matrix


@dataclass(frozen=True)
class Update:
    status: str
    phi: float
    curvature: float
    model_curvature: float
    residual: float | None = None
    matrix: np.ndarray | None = None
    info: MatrixInfo | None = None
    matrix_checked: bool = False


def broyden_update(matrix, step, gradient_change, phi=0.0, *, adaptive=False,
                   curvature_tolerance=1e-12, matrix_rcond=1e-12):
    if not (math.isfinite(phi) and 0.0 <= phi <= 1.0):
        raise ValueError("The Hessian Broyden class uses phi in [0, 1].")
    step_norm = float(np.linalg.norm(step))
    change_norm = float(np.linalg.norm(gradient_change))
    model_change = matrix @ step
    model_norm = float(np.linalg.norm(model_change))
    curvature = float(gradient_change @ step)
    model_curvature = float(step @ model_change)
    measures = (step_norm, change_norm, model_norm, curvature, model_curvature)
    if not all(math.isfinite(value) for value in measures):
        return Update("skipped_nonfinite_curvature", phi, curvature, model_curvature)
    tiny = float(np.finfo(float).tiny)
    if (curvature <= max(tiny, curvature_tolerance * step_norm * change_norm)
            or model_curvature <= max(tiny, curvature_tolerance * step_norm * model_norm)):
        return Update("skipped_curvature", phi, curvature, model_curvature)
    if adaptive:
        mismatch = float(np.linalg.norm(gradient_change - model_change))
        phi = 0.1 + 0.8 * min(1.0, mismatch / (change_norm + model_norm))
    scaled_model = model_change / math.sqrt(model_curvature)
    scaled_change = gradient_change / math.sqrt(curvature)
    updated = matrix - np.outer(scaled_model, scaled_model) + np.outer(scaled_change, scaled_change)
    if phi:
        correction = (math.sqrt(model_curvature) / math.sqrt(curvature)) * scaled_change - scaled_model
        updated += phi * np.outer(correction, correction)
    try:
        info = check_positive_matrix(updated, step.size, matrix_rcond)
    except OptimizationFailure as error:
        return Update("skipped_" + error.status, phi, curvature, model_curvature, matrix_checked=True)
    updated = 0.5 * updated + 0.5 * updated.T
    residual = float(np.linalg.norm(updated @ step - gradient_change) / change_norm)
    if not math.isfinite(residual) or residual > 1e-8:
        return Update("skipped_secant_residual", phi, curvature, model_curvature,
                      residual, matrix_checked=True)
    return Update("accepted", phi, curvature, model_curvature, residual, updated, info, True)
