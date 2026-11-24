from dataclasses import asdict
import math

import numpy as np


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def convergence_rows(run, reference, local_radius):
    point_ref = np.asarray(reference["point"])
    floor = 64 * np.finfo(float).eps * max(1.0, np.linalg.norm(point_ref))
    rows, errors = [], []
    for index, item in enumerate(run.history):
        point = np.asarray(item.point)
        error = float(np.linalg.norm(point - point_ref))
        previous_error = errors[-1] if errors else None
        ratio = quadratic_ratio = order = None
        valid_pair = previous_error is not None and min(error, previous_error) > floor
        if valid_pair:
            ratio = error / previous_error
            quadratic_ratio = ratio / previous_error
            if len(errors) >= 2 and floor < error < previous_error < errors[-2] <= local_radius:
                previous_log_ratio = math.log(previous_error) - math.log(errors[-2])
                if abs(previous_log_ratio) > 1e-3:
                    order = (math.log(error) - math.log(previous_error)) / previous_log_ratio
        errors.append(error)
        rows.append({
            "iteration": index,
            "x": finite(point[0]),
            "y": finite(point[1]),
            "objective": finite(item.value),
            "objective_gap": finite(item.value - reference["value"]),
            "gradient_x": finite(item.gradient[0]) if item.gradient else None,
            "gradient_y": finite(item.gradient[1]) if item.gradient else None,
            "gradient_norm": finite(item.gradient_norm),
            "point_error": finite(error),
            "error_ratio": finite(ratio),
            "quadratic_ratio": finite(quadratic_ratio),
            "local_order": finite(order),
            "error_reliable": int(error > floor),
            "alpha": finite(item.alpha),
            "step_norm": float(np.linalg.norm(point - run.history[index-1].point)) if index else None,
            "direction_x": finite(item.direction[0]) if item.direction else None,
            "direction_y": finite(item.direction[1]) if item.direction else None,
            "objective_change": finite(item.objective_change),
            "step_location": item.line_location,
            "line_contractions": item.line_iterations,
            "line_width": finite(item.line_width),
            "backtracks": item.backtracks,
            "directional_derivative_at_new_point": finite(item.previous_line_derivative),
            "relative_direction_residual": finite(item.relative_direction_residual),
            "direction_matrix_min_eigenvalue": finite(item.matrix_min_eigenvalue),
            "direction_matrix_rcond": finite(item.matrix_rcond),
            "update_status": item.update_status,
            "phi": finite(item.phi),
            "secant_curvature": finite(item.curvature),
            "model_curvature": finite(item.model_curvature),
            "relative_secant_residual": finite(item.secant_residual),
            **asdict(item.counts),
        })
    return rows
