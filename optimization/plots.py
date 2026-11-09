from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .objectives import OBJECTIVES
from .solvers import METHOD_NAMES


COLORS = {"division": "#176A9E", "golden": "#C15C17"}


def points(result):
    return np.array([[h["x"], h["y"]] for h in result.history])


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contour_path(ax, result, reference, common_cloud=None):
    objective = OBJECTIVES[result.objective]
    path = points(result)
    ref = np.array(reference["point"])
    cloud = np.vstack([path, ref]) if common_cloud is None else common_cloud
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    span = max(float(np.max(high-low)), 1.0)
    center = (low+high)/2
    limits = [center[0]-0.68*span, center[0]+0.68*span,
              center[1]-0.68*span, center[1]+0.68*span]
    xx, yy = np.meshgrid(np.linspace(*limits[:2], 360), np.linspace(*limits[2:], 360))
    z = objective.grid(xx, yy)
    largest = float(np.percentile(z-reference["value"], 95))
    levels = reference["value"]+np.geomspace(max(largest*3e-4, 1e-4), largest, 16)
    lines = ax.contour(xx, yy, z, levels=levels, colors="#AAB5BC", linewidths=0.75)
    ax.clabel(lines, lines.levels[::4], inline=True, fontsize=7, fmt="%.2f")
    color = COLORS[result.method]
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, ms=3.2, lw=1.7, label="All iterates", zorder=3)
    ax.scatter(*path[0], marker="s", s=55, color="#222222", label="Initial point", zorder=6)
    ax.scatter(*path[-1], facecolors="none", edgecolors=color, s=120, lw=2,
               label="Final iterate", zorder=7)
    ax.scatter(*ref, marker="*", color="#222222", s=75, label="Reference minimum", zorder=8)
    for k in range(min(len(path)-1, 8)):
        start, end = path[k], path[k+1]
        if np.linalg.norm(end-start) > span*0.015:
            ax.annotate("", xy=end, xytext=start,
                        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color})
    labelled = []
    for k in range(min(len(path), 4)):
        if all(np.linalg.norm(path[k]-p) > span*0.08 for p in labelled):
            ax.annotate(str(k), path[k], xytext=(6, 5), textcoords="offset points", fontsize=9)
            labelled.append(path[k])
    ax.set(xlim=limits[:2], ylim=limits[2:], xlabel="$x$", ylabel="$y$")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13)
    ax.set_title(f"{result.objective.upper()}  {METHOD_NAMES[result.method]}\n"
                 f"{result.iterations} updates, $\\|g\\|$ = {result.history[-1]['gradient_norm']:.2e}", fontsize=11)


def make_plots(runs, references, destination, tolerance):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlepad": 10, "legend.frameon": False})
    groups = {key: [r for r in runs if r.objective == key] for key in {r.objective for r in runs}}
    for result in runs:
        fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
        contour_path(ax, result, references[result.objective])
        ax.legend(loc="upper left", fontsize=9)
        save(fig, destination/f"contour_{result.objective}_{result.method}.png")
    for key in sorted(groups):
        items = groups[key]
        fig, axs = plt.subplots(1, len(items), figsize=(4.6*len(items), 4.9),
                                constrained_layout=True, squeeze=False)
        common = np.vstack([points(r) for r in items]+[np.array(references[key]["point"]).reshape(1, 2)])
        for ax, result in zip(axs[0], items):
            contour_path(ax, result, references[key], common)
            ax.legend(loc="upper left", fontsize=8)
        save(fig, destination/f"paths_{key}.png")
        fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
        for result in items:
            color = COLORS[result.method]
            label = METHOD_NAMES[result.method]
            k = np.arange(len(result.history))
            values = np.array([h["value"] for h in result.history])
            norms = np.array([h["gradient_norm"] for h in result.history])
            error = np.linalg.norm(points(result)-references[key]["point"], axis=1)
            gap = np.abs(values-references[key]["value"])
            axs[0, 0].plot(k, values, "o-", color=color, ms=3, label=label)
            axs[0, 1].semilogy(k, np.maximum(norms, 1e-17), "o-", color=color, ms=3, label=label)
            axs[1, 0].semilogy(k, np.maximum(error, 1e-17), "o-", color=color, ms=3, label=label)
            axs[1, 1].semilogy(k, np.maximum(gap, 1e-16), "o-", color=color, ms=3, label=label)
        axs[0, 1].axhline(tolerance, color="#333333", ls="--", lw=1, label="Stopping tolerance")
        axs[1, 1].axhspan(1e-17, 3e-15, color="#E5E5E5", label="Rounding scale")
        titles = ["Objective value", "Gradient norm", "Distance to reference minimum", "Absolute objective error"]
        for ax, title in zip(axs.flat, titles):
            ax.set(xlabel="Iteration $k$", title=title)
            ax.grid(alpha=0.2)
        axs[0, 0].legend(fontsize=8)
        axs[0, 1].legend(fontsize=8)
        fig.suptitle(f"{key.upper()}  Convergence from (2, 2)" if all(r.settings['initial_point'] == (2.0, 2.0) for r in items)
                     else f"{key.upper()}  Convergence", fontsize=14)
        save(fig, destination/f"convergence_{key}.png")
        objective = OBJECTIVES[key]
        path = np.vstack([points(r) for r in items])
        low, high = path.min(axis=0)-0.5, path.max(axis=0)+0.5
        xx, yy = np.meshgrid(np.linspace(low[0], high[0], 85), np.linspace(low[1], high[1], 85))
        fig = plt.figure(figsize=(8.2, 5.8))
        ax = fig.add_axes([0.02, 0.08, 0.82, 0.8], projection="3d")
        ax.plot_surface(xx, yy, objective.grid(xx, yy), cmap="Blues", alpha=0.35, linewidth=0)
        for result in items:
            p = points(result)
            ax.plot(p[:, 0], p[:, 1], objective.grid(p[:, 0], p[:, 1]), "o-",
                    color=COLORS[result.method], ms=3, lw=2, label=METHOD_NAMES[result.method])
        ax.set(xlabel="$x$", ylabel="$y$", title=f"{key.upper()}  Surface and iteration paths")
        fig.text(0.92, 0.52, "$f(x,y)$", rotation=90, ha="center", va="center")
        ax.legend(fontsize=8)
        save(fig, destination/f"surface_{key}.png")
    keys = sorted(groups)
    fig, axs = plt.subplots(1, len(keys), figsize=(6*len(keys), 4.4), constrained_layout=True, squeeze=False)
    for ax, key in zip(axs[0], keys):
        for result in groups[key]:
            rows = result.history[1:]
            y = [max(abs(h["gradient_cosine_previous"] or 0), 1e-12) for h in rows]
            ax.semilogy(range(1, len(rows)+1), y, "o-", color=COLORS[result.method], ms=3,
                        label=METHOD_NAMES[result.method])
        ax.set(title=f"{key.upper()}  Successive gradient directions", xlabel="Accepted step", ylabel="$|\\cos\\theta_k|$")
        ax.set_ylim(1e-12, 2)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, destination/"orthogonality.png")
    fig, axs = plt.subplots(1, len(keys), figsize=(6*len(keys), 4.4), constrained_layout=True, squeeze=False)
    for ax, key in zip(axs[0], keys):
        for result in groups[key]:
            ax.semilogy([h["function_evaluations"] for h in result.history],
                        [h["gradient_norm"] for h in result.history], "o-", ms=3,
                        color=COLORS[result.method], label=METHOD_NAMES[result.method])
        ax.axhline(tolerance, color="#333333", ls="--", lw=1)
        ax.set(title=f"{key.upper()}  Objective evaluation cost", xlabel="Cumulative function evaluations", ylabel="$\\|\\nabla f\\|$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, destination/"evaluation_cost.png")
    fig, axs = plt.subplots(1, len(keys), figsize=(5.9*len(keys), 5.1), constrained_layout=True, squeeze=False)
    for ax, key in zip(axs[0], keys):
        objective = OBJECTIVES[key]
        p = np.array([1.2, 0.2])
        g = objective.gradient(p)
        g /= np.linalg.norm(g)
        tangent = np.array([-g[1], g[0]])
        xx, yy = np.meshgrid(np.linspace(0.3, 2.1, 280), np.linspace(-0.7, 1.1, 280))
        z = objective.grid(xx, yy)
        level = objective.value(p)
        ax.contour(xx, yy, z, levels=[level], colors="#444444", linewidths=1.8)
        ax.contour(xx, yy, z, levels=[level*0.6, level*1.4, level*1.8], colors="#BAC1C7", linewidths=0.8)
        ax.plot([p[0]-.48*tangent[0], p[0]+.48*tangent[0]],
                [p[1]-.48*tangent[1], p[1]+.48*tangent[1]], color="#2C7E69", ls="--")
        for v, label, color in [(g, "$\\nabla f$", "#B04444"), (-g, "$-\\nabla f$", "#176A9E")]:
            end = p+0.55*v
            ax.annotate("", xy=end, xytext=p, arrowprops={"arrowstyle": "->", "lw": 2.2, "color": color})
            ax.text(*(end+np.array([0.04, 0.02])), label, color=color, fontsize=11)
        ax.scatter(*p, color="#222222", s=30)
        ax.text(p[0]+.08, p[1]-.15, "(1.2, 0.2)", fontsize=8)
        ax.text(0.05, 0.05, "$\\nabla f^T t = 0$\nArrow lengths scaled for display", transform=ax.transAxes, fontsize=9)
        ax.set(title=f"{key.upper()}  Gradient normal to a level curve", xlabel="$x$", ylabel="$y$", aspect="equal")
        ax.grid(alpha=0.15)
    save(fig, destination/"level_curve_geometry.png")
