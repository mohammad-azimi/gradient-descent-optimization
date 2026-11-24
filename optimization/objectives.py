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
    a: float = 1.0
    b: float = 2.0
    c: float = 1.0
    d: float = 1.0
    k: float = -1.0
    m: float = 2.0

    def value(self, point):
        return float(self.grid(*np.asarray(point, dtype=float)))

    def gradient(self, point):
        x, y = np.asarray(point, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            s = self.c*x + self.d*y
            v = np.exp(-s) if self.kind == "exp" else np.sin(s)
            return np.array([2*self.a*x - self.c*v + self.k, 2*self.b*y - self.d*v + self.m], dtype=float)

    def hessian(self, point):
        x, y = np.asarray(point, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            s = self.c*x + self.d*y
            v = np.exp(-s) if self.kind == "exp" else -np.cos(s)
            return np.diag([2*self.a, 2*self.b]) + v*np.outer([self.c, self.d], [self.c, self.d])

    def change(self, point, displacement, gradient):
        x, y = np.asarray(point, dtype=float)
        dx, dy = np.asarray(displacement, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            s = self.c*x + self.d*y
            ds = self.c*dx + self.d*dy
            quadratic = self.a*dx*dx + self.b*dy*dy
            if self.kind == "exp":
                nonlinear = np.exp(-s) * exp_remainder(-ds)
            else:
                nonlinear = -2*np.cos(s)*np.sin(ds/2)**2 + np.sin(s)*sin_remainder(ds)
            return float(np.dot(gradient, displacement) + quadratic + nonlinear)

    def grid(self, x, y):
        with np.errstate(over="ignore", invalid="ignore"):
            s = self.c*x + self.d*y
            v = np.exp(-s) if self.kind == "exp" else np.cos(s)
            return self.a*x*x + self.b*y*y + v + self.k*x + self.m*y


OBJECTIVES = {
    "f1": Objective("f1", "Exponential objective", "exp"),
    "f2": Objective("f2", "Cosine objective", "cos"),
}
