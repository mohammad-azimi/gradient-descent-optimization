from dataclasses import dataclass

import numpy as np


def exp_remainder(z):
    if abs(z) < 0.01:
        term = z * z / 2.0
        total = term
        for n in range(3, 13):
            term *= z / n
            total += term
        return total
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.expm1(z) - z)


def sin_remainder(z):
    if abs(z) < 0.01:
        term = z ** 3 / 6.0
        total = term
        for n in range(5, 14, 2):
            term *= -z * z / (n * (n - 1))
            total += term
        return total
    return float(z - np.sin(z))


@dataclass(frozen=True)
class Objective:
    key: str
    name: str
    kind: str

    def value(self, point):
        x, y = np.asarray(point, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            nonlinear = np.exp(-x - y) if self.kind == "exp" else np.cos(x + y)
            return float(x*x + 2*y*y + nonlinear - x + 2*y)

    def gradient(self, point):
        x, y = np.asarray(point, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            v = np.exp(-x-y) if self.kind == "exp" else np.sin(x+y)
            return np.array([2*x - v - 1, 4*y - v + 2], dtype=float)

    def hessian(self, point):
        x, y = np.asarray(point, dtype=float)
        v = np.exp(-x-y) if self.kind == "exp" else -np.cos(x+y)
        return np.diag([2.0, 4.0]) + v * np.ones((2, 2))

    def change(self, point, displacement, gradient):
        x, y = np.asarray(point, dtype=float)
        dx, dy = np.asarray(displacement, dtype=float)
        ds = dx + dy
        quadratic = dx*dx + 2*dy*dy
        with np.errstate(over="ignore", invalid="ignore"):
            if self.kind == "exp":
                nonlinear = np.exp(-x-y) * exp_remainder(-ds)
            else:
                s = x+y
                nonlinear = -2*np.cos(s)*np.sin(ds/2)**2 + np.sin(s)*sin_remainder(ds)
            return float(np.dot(gradient, displacement) + quadratic + nonlinear)

    def grid(self, x, y):
        with np.errstate(over="ignore", invalid="ignore"):
            v = np.exp(-x-y) if self.kind == "exp" else np.cos(x+y)
            return x*x + 2*y*y + v - x + 2*y


OBJECTIVES = {
    "f1": Objective("f1", "Exponential objective", "exp"),
    "f2": Objective("f2", "Cosine objective", "cos"),
}
