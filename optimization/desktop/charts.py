import math

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from ..minimizer import METHODS


INK = "#142338"
MUTED = "#5D7086"
BORDER = "#DCE5EE"
PLOT_GRID = "#E9EEF4"
COLORS = dict(zip(METHODS, ("#175AA5", "#397FB6", "#6D50AE", "#8A68C7", "#007663",
                          "#149F85", "#AF581F", "#C57F15", "#AC476D", "#CA5D91")))
METRICS = {
    "Objective evaluations": "function_evaluations",
    "Gradient evaluations": "gradient_evaluations",
    "Hessian evaluations": "hessian_evaluations",
    "Linear solves": "linear_solves",
    "Matrix inversions": "matrix_inversions",
    "Matrix checks": "matrix_checks",
    "Accepted updates": "iterations",
    "Skipped updates": "skipped_updates",
    "Median solver time (ms)": "median_ms",
}


def _new_figure(width=9, height=5):
    return Figure(figsize=(width, height), dpi=100, facecolor="#FFFFFF", layout="constrained")


def _style_axes(ax, axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis=axis, color=PLOT_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _line_style(run):
    spec = METHODS[run.result.method]
    dashes = ":" if run.adaptive else "--" if spec.step.startswith("golden") else "-"
    return {"color": COLORS[run.result.method], "linestyle": dashes, "linewidth": 1.9,
            "marker": "o", "markersize": 3.5, "label": run.short}


def _paths(runs):
    return [np.array([row.point for row in run.result.history], dtype=float) for run in runs]


def _limits(paths, runs, reference):
    start = paths[0][0]
    anchors = [start]
    if reference is not None:
        anchors.append(np.array(reference["point"]))
    else:
        anchors.extend(path[-1] for run, path in zip(runs, paths)
                       if run.result.status == "converged" and np.all(np.isfinite(path[-1])))
    cloud = np.vstack(anchors)
    if np.max(np.abs(cloud)) > 1e100:
        return None
    radius = max(0.5, float(np.max(np.linalg.norm(cloud - start, axis=1))))
    points = np.vstack(paths)
    nearby = points[np.max(np.abs(points - start), axis=1) <= 2.3 * radius]
    cloud = np.vstack([cloud, nearby])
    lower, upper = cloud.min(axis=0), cloud.max(axis=0)
    span = max(0.5, float(np.max(upper - lower)))
    return np.column_stack(((lower + upper) / 2 - 0.65 * span,
                            (lower + upper) / 2 + 0.65 * span))


def _levels(z, objective, start, reference):
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return None
    if reference is not None:
        floor = reference["value"]
        cap = max(0.005, 1.75 * (objective.value(start) - floor))
        top = min(float(np.percentile(finite, 85)) - floor, cap)
        if top > 0:
            levels = floor + np.geomspace(top * 0.0004, top, 13)
            if np.unique(levels).size >= 2:
                return np.unique(levels)
    low, high = np.percentile(finite, [8, 87])
    return np.unique(np.linspace(low, high, 13)) if high > low else None


def trajectories(session, selected=None):
    runs = session.runs if selected is None else tuple(run for run in session.runs if run.id == selected)
    if not runs:
        runs = session.runs
    fig = _new_figure()
    ax = fig.subplots()
    arrays = _paths(runs)
    box = _limits(arrays, runs, session.reference)
    if box is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Coordinate scale exceeds the plot range. Open Iterations to inspect the values.",
                ha="center", va="center", transform=ax.transAxes, color=INK, wrap=True)
        return fig
    _style_axes(ax)
    ax.set(xlabel="x", ylabel="y", xlim=box[0], ylim=box[1])
    ax.set_aspect("equal", adjustable="box")
    xx, yy = np.meshgrid(np.linspace(*box[0], 190), np.linspace(*box[1], 190))
    objective = session.experiment.objective
    z = objective.grid(xx, yy)
    levels = _levels(z, objective, arrays[0][0], session.reference)
    if levels is not None and len(levels) >= 2:
        ax.contour(xx, yy, np.ma.masked_invalid(z), levels=levels, colors="#B6C7D9", linewidths=0.75)
    else:
        ax.text(0.03, 0.03, "No varying finite contours in this window", transform=ax.transAxes,
                fontsize=9, color=MUTED)
    clipped = False
    for run, path in zip(runs, arrays):
        visible = np.all((box[:, 0] <= path) & (path <= box[:, 1]), axis=1)
        clipped |= not np.all(visible)
        view = np.where(visible[:, None], path, np.nan)
        options = _line_style(run)
        ax.plot(*view.T, **options, zorder=4)
        if visible[-1]:
            if run.result.status == "converged":
                ax.scatter(*path[-1], marker="s", facecolors="none", edgecolors=options["color"],
                           s=64, linewidths=1.4, zorder=7)
            else:
                ax.scatter(*path[-1], marker="x", color=options["color"], s=64, linewidths=1.4, zorder=7)
        span = box[0, 1] - box[0, 0]
        for origin, target in zip(view[:6], view[1:7]):
            if np.linalg.norm(target - origin) > 0.05 * span:
                ax.annotate("", target, origin, zorder=5, arrowprops={
                    "arrowstyle": "->", "color": options["color"], "lw": 1.2,
                })
        departures = np.flatnonzero(visible[:-1] & ~visible[1:] & np.all(np.isfinite(path[1:]), axis=1))
        if departures.size:
            origin, target = path[departures[0]:departures[0]+2]
            direction = target - origin
            fraction = min((box[i, int(direction[i] > 0)] - origin[i]) / direction[i]
                           for i in range(2) if direction[i] != 0)
            ax.annotate("", origin + 0.93 * fraction * direction, origin, arrowprops={
                "arrowstyle": "->", "linestyle": ":", "color": options["color"], "lw": 1.0,
            })
    start = arrays[0][0]
    ax.scatter(*start, marker="D", s=54, color="#17324F", edgecolors="white", linewidths=0.8, zorder=9)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#17324F", marker="D", lw=0))
    labels.append("Start")
    if session.reference is not None:
        ax.scatter(*session.reference["point"], marker="*", s=150, color="#0B7455",
                   edgecolors="white", linewidths=0.7, zorder=10)
        handles.append(Line2D([0], [0], color="#0B7455", marker="*", ms=10, lw=0))
        labels.append("Reference minimum")
    if clipped:
        ax.text(0.025, 0.965, "Distant iterates are in the Iterations tab.", va="top",
                transform=ax.transAxes, fontsize=9, color=MUTED,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    fig.legend(handles, labels, ncols=min(4, len(labels)), loc="outside lower center",
               frameon=False, fontsize=8.5, labelcolor=INK)
    return fig


def convergence(session):
    fig = _new_figure()
    axes = fig.subplots(2, 2).flatten()
    third_metric = ("point_error", "Distance to reference", True) if session.reference else (
        "step_norm", "Step length", True,
    )
    specs = [("objective", "Objective value", False), ("gradient_norm", "Gradient norm", True),
             third_metric, ("alpha", "Step size α", False)]
    last_k = max(run.result.iterations for run in session.runs)
    for index, (ax, (metric, title, log_scale)) in enumerate(zip(axes, specs)):
        _style_axes(ax)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")
        if index >= 2:
            ax.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.set_xlim(-0.2, max(1, last_k) + 0.2)
        if log_scale:
            ax.set_yscale("log")
        for run in session.runs:
            xs = [item["iteration"] for item in run.trace]
            ys = np.array([item[metric] for item in run.trace], dtype=float)
            representable = np.isfinite(ys)
            if log_scale:
                representable &= ys > 0
            ax.plot(xs, np.where(representable, ys, np.nan), **_line_style(run))
            if run.result.status != "converged" and representable[-1]:
                ax.scatter(xs[-1], ys[-1], marker="x", color=COLORS[run.result.method], s=45, zorder=6)
        if metric == "gradient_norm":
            tolerance = session.experiment.settings.tolerance
            ax.axhline(tolerance, color=MUTED, linewidth=1, linestyle=":")
            ax.text(0.985, 0.045, f"Stop below {tolerance:g}", ha="right",
                    transform=ax.transAxes, color=MUTED, fontsize=8)
        if metric == "objective":
            values = [r["objective"] for run in session.runs for r in run.trace if r["objective"] is not None]
            if values and max(map(abs, values)) > 1e4:
                if min(values) > 0:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("symlog", linthresh=1)
        if metric == "alpha":
            ax.set_ylim(-0.04, 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=min(4, len(labels)),
               frameon=False, fontsize=8.4, labelcolor=INK)
    return fig


def work(session, metric="Objective evaluations"):
    fig = _new_figure(height=4.6)
    ax = fig.subplots()
    _style_axes(ax, axis="x")
    field = METRICS[metric]
    summaries = [item.summary() for item in session.runs]
    values = [row[field] for row in summaries]
    for index, (run, value) in enumerate(zip(session.runs, values)):
        converged = run.result.status == "converged"
        color = COLORS[run.result.method]
        ax.barh(index, value, color=color, height=0.63, alpha=1 if converged else 0.44)
        value_text = f"{value:.3f}" if field == "median_ms" else f"{value:d}"
        ax.annotate(value_text, (value, index), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(values)), [
        item.short + ("  ·  Stopped" if item.result.status != "converged" else "")
        for item in session.runs
    ])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_xlim(0, max(1, max(values, default=0)) * 1.23)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    if field != "median_ms":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    return fig
