import numpy as np
from scipy.optimize import brentq, root
from scipy.special import lambertw


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
