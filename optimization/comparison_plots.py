from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .minimizer import METHODS
from .objectives import OBJECTIVES
from .plots import save


FAMILY_COLORS = {
    "gradient": "#1B6CA8",
    "newton": "#158478",
    "bfgs": "#BF5E12",
    "broyden": "#813EA4",
}
PANELS = [
    ("Gradient descent", ("division", "golden")),
    ("Newton direction from a linear solve", ("newton_solve_fixed", "newton_solve_golden")),
    ("Quasi-Newton with unit steps", ("bfgs_fixed", "broyden_fixed")),
    ("Quasi-Newton with Golden Section", ("bfgs_golden", "broyden_golden")),
]


def plotted_methods(rows):
    methods = {row["method"] for row in rows}
    return [
        row for row in rows
        if not ("newton_inverse" in row["method"] and row["method"].replace("inverse", "solve") in methods)
    ]


def style(row):
    spec = METHODS[row["method"]]
    golden = spec.step.startswith("golden")
    return {
        "color": FAMILY_COLORS[spec.family],
        "linestyle": "--" if golden else "-",
        "marker": "o" if golden else "s",
        "ms": 2.9,
        "lw": 1.6,
        "label": spec.short,
    }


def coords(trace):
    return np.array([[r["x"], r["y"]] for r in trace], dtype=float)


def case_title(rows):
    first = rows[0]
    return f"{first['function'].upper()} from ({first['initial_x']:g}, {first['initial_y']:g})"


def plot_trajectories(rows, traces, reference, output):
    rows = plotted_methods(rows)
    methods = {row["method"] for row in rows}
    ref = np.array(reference["point"])
    anchor = coords(traces[rows[0]["case"]])[0]
    radius = max(float(np.linalg.norm(anchor - ref)), 1.0)
    cloud = np.concatenate([coords(traces[row["case"]]) for row in rows] + [ref[None, :]])
    normal_scale = np.linalg.norm(cloud - ref, axis=1) <= 1.7 * radius
    low, high = cloud[normal_scale].min(axis=0), cloud[normal_scale].max(axis=0)
    span = max(float(np.max(high - low)), 1.0)
    midpoint = (low + high) / 2
    xlim, ylim = [(center - 0.62 * span, center + 0.62 * span) for center in midpoint]
    xx, yy = np.meshgrid(np.linspace(*xlim, 220), np.linspace(*ylim, 220))
    objective = OBJECTIVES[rows[0]["function"]]
    z = objective.grid(xx, yy) - reference["value"]
    start_gap = objective.value(anchor) - reference["value"]
    level_top = max(1e-5, min(float(np.nanpercentile(z, 90)), max(1.0, 1.7 * start_gap)))
    levels = np.geomspace(max(1e-7, level_top * 0.00015), level_top, 13)
    panels = [(title, keys) for title, keys in PANELS if methods.intersection(keys)]
    inverse_keys = tuple(key for key in methods if key.startswith("newton_inverse"))
    if inverse_keys:
        panels.append(("Newton direction from an explicit inverse", tuple(sorted(inverse_keys))))
    nrows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.9 * nrows + 0.35), squeeze=False,
                             constrained_layout=True)
    fig.suptitle(f"Optimization trajectories for {case_title(rows)}", fontsize=15, fontweight="bold")
    for ax, (title, keys) in zip(axes.flat, panels):
        ax.contour(xx, yy, z, levels=levels, colors="#B5C1CF", linewidths=0.65)
        ax.scatter(*anchor, marker="D", s=28, color="#26364A", zorder=6)
        ax.scatter(*ref, marker="*", s=108, color="#122333", zorder=7)
        clipped = False
        for row in rows:
            if row["method"] not in keys:
                continue
            path = coords(traces[row["case"]])
            aesthetics = style(row)
            color = aesthetics["color"]
            visible = ((xlim[0] <= path[:, 0]) & (path[:, 0] <= xlim[1])
                       & (ylim[0] <= path[:, 1]) & (path[:, 1] <= ylim[1]))
            clipped |= not np.all(visible)
            drawn = np.where(visible[:, None], path, np.nan)
            ax.plot(*drawn.T, **aesthetics, zorder=3)
            for start, end in zip(drawn[:7], drawn[1:8]):
                if np.linalg.norm(end - start) > 0.05 * span:
                    ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
            departures = np.flatnonzero(visible[:-1] & ~visible[1:])
            if departures.size:
                index = departures[0]
                origin, target = path[index:index + 2]
                vector = target - origin
                limits = np.array([xlim, ylim])
                fraction = min(
                    (limits[i, int(vector[i] > 0)] - origin[i]) / vector[i]
                    for i in range(2) if vector[i] != 0.0
                )
                ax.annotate("", origin + 0.93 * fraction * vector, origin,
                            arrowprops={"arrowstyle": "->", "linestyle": ":", "color": color, "lw": 1.3})
            if visible[-1]:
                ax.scatter(*path[-1], marker="s", s=70, facecolors="none",
                           edgecolors=color, linewidths=1.5, zorder=8)
        if clipped:
            ax.text(0.03, 0.94, "Distant iterates omitted", transform=ax.transAxes, va="top",
                    color="#53483C", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        ax.set(title=title, xlabel="$x$", ylabel="$y$", xlim=xlim, ylim=ylim, aspect="equal")
        ax.legend(fontsize=8, loc="best")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.legend(
        handles=[Line2D([0], [0], color="#26364A", marker="D", lw=0, label="Start"),
                 Line2D([0], [0], color="#122333", marker="*", lw=0, ms=9, label="Reference minimum"),
                 Line2D([0], [0], color="#26364A", marker="s", mfc="none", lw=0, label="Final visible iterate")],
        loc="outside lower center", ncol=3, fontsize=8.5,
    )
    save(fig, output)


def plot_convergence(rows, traces, reference, tolerance, output):
    rows = plotted_methods(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.85), constrained_layout=True)
    fig.suptitle(f"Convergence for {case_title(rows)}", fontsize=15, fontweight="bold")
    gap_floor = 64 * np.finfo(float).eps * max(1.0, abs(reference["value"]))
    metrics = ("objective_gap", "gradient_norm", "point_error", "alpha")
    titles = ("Objective gap $f(x_k)-f^*$", "Gradient norm", "Distance to the reference minimizer", "Step size")
    for row in rows:
        trace = traces[row["case"]]
        indices = [entry["iteration"] for entry in trace]
        for index, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            values = np.array([entry[metric] for entry in trace], dtype=float)
            if index == 0:
                values[values <= gap_floor] = np.nan
            elif index == 1:
                values[values <= 0] = np.nan
            elif index == 2:
                values[[not entry["error_reliable"] for entry in trace]] = np.nan
            ax.plot(indices, values, **style(row))
            if row["status"] != "converged" and np.isfinite(values[-1]):
                ax.scatter(indices[-1], values[-1], color=style(row)["color"], marker="x", s=60)
    axes[0, 1].axhline(tolerance, color="#44546C", ls=":", lw=1.3)
    axes[0, 1].text(0.99, 0.04, f"Stop below {tolerance:.0e}", fontsize=8, ha="right",
                    transform=axes[0, 1].transAxes)
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set(title=title, xlabel="Iteration")
        if index < 3 and any(np.any(line.get_ydata() > 0) for line in ax.lines):
            ax.set_yscale("log")
        if index == 3:
            ax.set_ylim(-0.02, 1.08)
        ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(4, len(rows)), fontsize=8.4)
    save(fig, output)


def plot_costs(rows, output):
    groups = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    groups = {method: group for method, group in groups.items() if group}
    labels = []
    for method, group in groups.items():
        successes = sum(row["status"] == "converged" for row in group)
        labels.append(f"{METHODS[method].short}  ({successes}/{len(group)})")
    colors = [FAMILY_COLORS[METHODS[method].family] for method in groups]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    fig.suptitle("Median cost across converged cases", fontsize=15, fontweight="bold")
    for ax, field, title, format_value in zip(
        axes.flat,
        ("iterations", "function_evaluations", "hessian_evaluations", "median_ms"),
        ("Outer updates", "Function evaluations", "Analytical Hessian evaluations", "Solver time in milliseconds"),
        (lambda v: f"{v:g}", lambda v: f"{v:g}", lambda v: f"{v:g}", lambda v: f"{v:.3f}"),
    ):
        values = []
        for group in groups.values():
            observed = [row[field] for row in group if row["status"] == "converged"]
            values.append(float(np.median(observed)) if observed else float("nan"))
        lengths = [v if np.isfinite(v) else 0 for v in values]
        ax.barh(labels, lengths, color=colors, height=0.67, alpha=0.9)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlim(0.0, max(1e-6, max(lengths)) * 1.18)
        for position, value in enumerate(values):
            caption = format_value(value) if np.isfinite(value) else "—"
            ax.annotate(caption, (lengths[position], position), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8)
        ax.tick_params(axis="y", labelsize=8.2, length=0)
        ax.grid(axis="x", alpha=0.14)
    save(fig, output)


def plot_quasi_comparison(main, adaptive, output):
    groups = defaultdict(dict)
    for row in main + adaptive:
        if METHODS[row["method"]].family in ("bfgs", "broyden"):
            groups[(row["function"], row["start_id"])][row["method"], row["variant"]] = row
    if not groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 5.1), constrained_layout=True)
    fig.suptitle("BFGS and Broyden class iteration counts", fontsize=15, fontweight="bold")
    policies = [("bfgs_", "main", "BFGS", "#BF5E12"), ("broyden_", "main", "Constant φ", "#813EA4"),
                ("broyden_", "adaptive", "Adaptive φ", "#418AA3")]
    for ax, strategy in zip(axes, ("fixed", "golden")):
        locations = np.arange(len(groups))
        ticks = []
        for position, lookup in enumerate(groups.values()):
            item = next(iter(lookup.values()))
            ticks.append(f"{item['function'].upper()}\n({item['initial_x']:g}, {item['initial_y']:g})")
            for index, (prefix, variant, label, color) in enumerate(policies):
                row = lookup.get((prefix + strategy, variant))
                if not row:
                    continue
                x = position + 0.24 * (index - 1)
                success = row["status"] == "converged"
                ax.bar(x, row["iterations"], 0.21, color=color, alpha=0.87 if success else 0.23)
                if not success:
                    ax.scatter([x], [row["iterations"]], marker="x", color=color, zorder=3)
        ax.set(xticks=locations, xticklabels=ticks, ylabel="Accepted updates",
               title="Unit steps" if strategy == "fixed" else "Closed Golden Section")
        ax.grid(axis="y", alpha=0.18)
    fig.legend(handles=[Line2D([0], [0], color=c, lw=5, label=l) for _, _, l, c in policies]
               + [Line2D([0], [0], color="#4D5968", marker="x", lw=0, label="Nonconverged")],
               loc="outside lower center", ncol=4, fontsize=8.5)
    save(fig, output)


def plot_newton_model(output):
    objective = OBJECTIVES["f1"]
    point = np.array([0.0, 0.0])
    value, gradient, hessian = objective.value(point), objective.gradient(point), objective.hessian(point)
    direction = np.linalg.solve(hessian, -gradient)
    alpha = np.linspace(0.0, 1.6, 220)
    direct = [objective.value(point + a * direction) for a in alpha]
    model = value + alpha * (gradient @ direction) + 0.5 * alpha**2 * (direction @ hessian @ direction)
    fig, ax = plt.subplots(figsize=(7.4, 3.5), constrained_layout=True)
    ax.plot(alpha, direct, color="#158478", lw=2.3, label="Objective on the Newton line")
    ax.plot(alpha, model, color="#37557C", lw=1.8, ls="--", label="Local quadratic model")
    ax.axvline(1.0, color="#8B97A6", ls=":", lw=1)
    ax.set(title="Newton model for the exponential objective at the origin",
           xlabel="Step size α", ylabel="Objective or model value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.16)
    save(fig, output)


def make_plots(main, adaptive, traces, references, tolerance, destination):
    destination.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in main:
        groups[(row["function"], row["start_id"])].append(row)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 9}):
        for (key, start), rows in groups.items():
            plot_trajectories(rows, traces, references[key], destination / f"paths_{key}_{start}.png")
            plot_convergence(rows, traces, references[key], tolerance, destination / f"convergence_{key}_{start}.png")
        plot_costs(main, destination / "cost_comparison.png")
        plot_quasi_comparison(main, adaptive, destination / "quasi_newton_comparison.png")
        if {"f1", "f2"} == {row["function"] for row in main}:
            plot_newton_model(destination / "newton_quadratic_model.png")
