from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LineResult:
    alpha: float
    value: float
    iterations: int
    width: float
    converged: bool


def golden_section(function, lower=0.0, upper=1.0, tolerance=1e-10, max_iterations=200):
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("The search interval must have finite ordered endpoints.")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError("The line tolerance must be positive and finite.")
    if max_iterations < 1:
        raise ValueError("The line iteration limit must be positive.")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lower), float(upper)
    c, d = b-ratio*(b-a), a+ratio*(b-a)
    fc, fd = function(c), function(d)
    count = 0
    while b-a > tolerance and count < max_iterations:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b-ratio*(b-a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a+ratio*(b-a)
            fd = function(d)
        count += 1
    alpha, value = (c, fc) if fc <= fd else (d, fd)
    return LineResult(alpha, value, count, b-a, b-a <= tolerance)
