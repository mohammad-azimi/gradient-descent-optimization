from dataclasses import dataclass

import numpy as np


class OptimizationFailure(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


@dataclass(frozen=True)
class MatrixInfo:
    min_eigenvalue: float
    reciprocal_condition: float


def check_positive_matrix(matrix, size, rcond):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size):
        raise OptimizationFailure("invalid_matrix", "Expected an n by n matrix.")
    if not np.all(np.isfinite(matrix)):
        raise OptimizationFailure("nonfinite_matrix", "The matrix contains a non-finite entry.")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        raise OptimizationFailure("singular_matrix", "The matrix is zero.")
    scaled = matrix / scale
    if np.max(np.abs(scaled - scaled.T)) > 1e-12:
        raise OptimizationFailure("invalid_matrix", "The matrix is not symmetric to working accuracy.")
    try:
        eigenvalues = np.linalg.eigvalsh(scaled)
    except np.linalg.LinAlgError as error:
        raise OptimizationFailure("numerical_failure", "Matrix eigenvalue calculation failed.") from error
    magnitudes = np.abs(eigenvalues)
    relative_min = float(np.min(magnitudes) / np.max(magnitudes))
    if relative_min <= rcond:
        raise OptimizationFailure("singular_matrix", "The matrix is singular or nearly singular.")
    if eigenvalues[0] <= 0.0:
        raise OptimizationFailure("invalid_search_direction", "The matrix is not positive definite.")
    return MatrixInfo(float(eigenvalues[0]) * scale, relative_min)


def newton_direction(matrix, gradient, approach):
    if approach not in ("inverse", "solve"):
        raise ValueError("Choose inverse or solve for the Newton direction.")
    try:
        if approach == "inverse":
            return -np.linalg.inv(matrix) @ gradient
        return np.linalg.solve(matrix, -gradient)
    except np.linalg.LinAlgError as error:
        raise OptimizationFailure("singular_matrix", "The direction calculation failed.") from error
