from concurrent.futures import CancelledError
from dataclasses import replace
import math
from queue import Empty, SimpleQueue
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import charts, theme
from .session import (
    COEFFICIENTS, DEFAULT_SETTINGS, METHOD_PRESETS, Experiment, execute, export_session,
)
from ..minimizer import METHODS
from ..objectives import OBJECTIVES


OBJECTIVE_LABELS = {
    "F1 · Exponential": "f1",
    "F2 · Cosine": "f2",
}
FORMULAS = {
    "f1": "ax² + by² + exp(-cx - dy) + kx + my",
    "f2": "ax² + by² + cos(cx + dy) + kx + my",
}
ADVANCED_FIELDS = {
    "initial_step": ("Initial trial step", float),
    "reduction_factor": ("Reduction factor", float),
    "armijo_epsilon": ("Armijo coefficient", float),
    "max_backtracks": ("Maximum reductions", int),
    "line_lower": ("Lower step bound", float),
    "line_upper": ("Upper step bound", float),
    "line_tolerance": ("Line width tolerance", float),
    "max_line_iterations": ("Line search limit", int),
    "matrix_rcond": ("Matrix condition threshold", float),
    "curvature_tolerance": ("Curvature threshold", float),
    "broyden_phi": ("Broyden parameter phi", float),
    "repeats": ("Timing repeats", int),
    "local_radius": ("Local error radius", float),
}
SUMMARY_COLUMNS = [
    ("method", "Method", 180, "w"), ("status", "Status", 145, "w"),
    ("iterations", "k", 40, "e"), ("value", "Final f(x)", 102, "e"),
    ("gradient", "||g||", 84, "e"), ("nf", "Nf", 57, "e"),
    ("nh", "Nh", 47, "e"), ("time", "ms", 63, "e"),
]


def number(value, digits=8):
    return f"{value:.{digits}g}" if value is not None and math.isfinite(value) else "-"


def parse(value, cast, label):
    try:
        result = cast(value.strip().replace("−", "-").replace("٫", "."))
    except ValueError as error:
        kind = "an integer" if cast is int else "a number"
        raise ValueError(f"Enter {kind} for {label}.") from error
    if not math.isfinite(result):
        raise ValueError(f"Enter a finite value for {label}.")
    return result


def status_name(status):
    return status.replace("_", " ").capitalize().replace("Nonfinite", "Non-finite")


def coordinate_text(point):
    return f"({number(point[0])}, {number(point[1])})"


def make_table(parent, columns, height=10):
    frame = ttk.Frame(parent, style="White.TFrame")
    tree = ttk.Treeview(frame, columns=[col[0] for col in columns], show="headings", height=height,
                        selectmode="browse")
    vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    tree.tag_configure("stripe", background="#F4F8FC")
    tree.tag_configure("stopped", foreground=theme.WARNING)
    configure_columns(tree, columns)
    return frame, tree


def configure_columns(tree, columns):
    tree.configure(columns=[col[0] for col in columns])
    for key, title, width, anchor in columns:
        tree.heading(key, text=title, anchor=anchor)
        tree.column(key, width=width, minwidth=width-10, anchor=anchor, stretch=width >= 62)


class OptimizationLab:
    def __init__(self, root, *, autorun=True):
        self.root = root
        theme.install(root)
        width = min(root.winfo_screenwidth() - 50, 1420)
        height = min(root.winfo_screenheight() - 75, 860)
        root.geometry(f"{max(930, width)}x{max(660, height)}")
        root.minsize(930, 660)
        root.title("Optimization Lab")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.bind("<Control-Return>", self.start)
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", self._leave_fullscreen)

        self.event_queue = SimpleQueue()
        self.cancel_event = None
        self.session = None
        self.plot_cache = {}
        self.run_labels = {}

        self.objective_var = tk.StringVar(root, value=next(iter(OBJECTIVE_LABELS)))
        self.formula_var = tk.StringVar(root, value=FORMULAS["f1"])
        self.coefficients = {key: tk.StringVar(root) for key in COEFFICIENTS}
        self.start_vars = [tk.StringVar(root, value="0"), tk.StringVar(root, value="0")]
        self.tolerance_var = tk.StringVar(root)
        self.max_iterations_var = tk.StringVar(root)
        self.method_preset = tk.StringVar(root, value="Compare all methods")
        self.selected_methods = METHOD_PRESETS["Compare all methods"]
        self.method_count = tk.StringVar(root)
        self.advanced = {}
        self.compare_adaptive = False

        self.status_text = tk.StringVar(root, value="Ready")
        self.run_context = tk.StringVar(root, value="Choose inputs and run an experiment.")
        self.run_detail = tk.StringVar(root)
        self.iteration_detail = tk.StringVar(root)
        self.selected_a = tk.StringVar(root)
        self.selected_b = tk.StringVar(root, value="None")
        self.plot_focus = tk.StringVar(root, value="All methods")
        self.cost_metric = tk.StringVar(root, value=next(iter(charts.METRICS)))

        self._header()
        self._sidebar()
        self._workspace()
        self._footer()
        self._reset()
        root.after(60, self._poll)
        if autorun:
            root.after(140, self.start)

    def _header(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(22, 13, 22, 13))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(side="left")
        ttk.Label(brand, text="Optimization Lab", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(brand, text="Gradient descent · Newton · Quasi-Newton",
                  style="BrandNote.TLabel").pack(anchor="w", pady=(2, 0))
        self.export_button = ttk.Button(header, text="Export session", command=self._export,
                                        style="Header.TButton", state="disabled")
        self.export_button.pack(side="right")

    def _sidebar(self):
        panel = ttk.Frame(self.root, style="Side.TFrame", width=314)
        panel.grid(row=1, column=0, sticky="ns")
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        canvas = tk.Canvas(panel, bg=theme.NAVY, highlightthickness=0, width=292)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        form = ttk.Frame(canvas, style="Side.TFrame", padding=(16, 15, 12, 12))
        form.columnconfigure(0, weight=1)
        item = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        self._form(canvas, form)
        actions = ttk.Frame(panel, style="Side.TFrame", padding=(16, 9, 16, 13))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(actions, text="Run experiment", style="Primary.TButton", command=self.start)
        self.run_button.pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(actions, text="Stop", style="Side.TButton", command=self._cancel,
                                         state="disabled")
        self.cancel_button.pack(side="right", padx=(7, 0))

    def _form(self, canvas, form):
        title = ttk.Frame(form, style="Side.TFrame")
        title.pack(fill="x", pady=(0, 13))
        ttk.Label(title, text="Experiment", style="SideTitle.TLabel").pack(side="left")
        ttk.Button(title, text="Reset", style="Preset.TButton", command=self._reset).pack(side="right")

        ttk.Label(form, text="Objective", style="SideLabel.TLabel").pack(anchor="w")
        objective = ttk.Combobox(form, values=list(OBJECTIVE_LABELS), textvariable=self.objective_var,
                                 state="readonly", width=17)
        objective.pack(fill="x", pady=(4, 3))
        objective.bind("<<ComboboxSelected>>", self._objective_changed)
        ttk.Label(form, textvariable=self.formula_var, style="SideMuted.TLabel", wraplength=258).pack(
            anchor="w", pady=(2, 10))
        coeffs = ttk.Frame(form, style="Side.TFrame")
        coeffs.pack(fill="x", pady=(0, 10))
        for i, key in enumerate(COEFFICIENTS):
            coeffs.columnconfigure(i % 3, weight=1, uniform="coef")
            self._field(coeffs, key, self.coefficients[key], i // 3, i % 3, side=True, width=5)

        ttk.Label(form, text="Starting point", style="SideLabel.TLabel").pack(anchor="w", pady=(1, 4))
        starts = ttk.Frame(form, style="Side.TFrame")
        starts.pack(fill="x")
        for index, label in enumerate(("x0", "y0")):
            starts.columnconfigure(index, weight=1)
            self._field(starts, label, self.start_vars[index], 0, index, side=True, width=8)
        presets = ttk.Frame(form, style="Side.TFrame")
        presets.pack(fill="x", pady=(3, 10))
        for i, pair in enumerate(((0, 0), (0, 10), (10, 0), (2, 2))):
            presets.columnconfigure(i, weight=1)
            button = ttk.Button(presets, text=f"{pair[0]}, {pair[1]}", style="Preset.TButton",
                                command=lambda point=pair: self._set_start(point))
            button.grid(row=0, column=i, sticky="ew", padx=(0, 4))

        stopping = ttk.Frame(form, style="Side.TFrame")
        stopping.pack(fill="x", pady=(0, 8))
        for i in range(2):
            stopping.columnconfigure(i, weight=1)
        self._field(stopping, "Gradient tolerance", self.tolerance_var, 0, 0, side=True, width=8)
        self._field(stopping, "Iteration limit", self.max_iterations_var, 0, 1, side=True, width=8)

        ttk.Label(form, text="Methods", style="SideLabel.TLabel").pack(anchor="w", pady=(2, 4))
        combo = ttk.Combobox(form, values=(*METHOD_PRESETS, "Custom selection"), width=17,
                             textvariable=self.method_preset, state="readonly")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", self._preset_changed)
        choices = ttk.Frame(form, style="Side.TFrame")
        choices.pack(fill="x", pady=(5, 6))
        ttk.Label(choices, textvariable=self.method_count, style="SideMuted.TLabel").pack(side="left")
        ttk.Button(choices, text="Choose", style="Preset.TButton", command=self._choose_methods).pack(
            side="right")
        ttk.Button(form, text="Advanced settings", style="Side.TButton",
                   command=self._advanced_settings).pack(fill="x", pady=(3, 0))

        def bind_scroll(widget):
            widget.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"),
                        add="+")
            widget.bind("<Button-4>", lambda event: canvas.yview_scroll(-1, "units"), add="+")
            widget.bind("<Button-5>", lambda event: canvas.yview_scroll(1, "units"), add="+")
            for child in widget.winfo_children():
                bind_scroll(child)

        bind_scroll(canvas)

    @staticmethod
    def _field(parent, label, var, row, column, *, side=False, width=12):
        frame = ttk.Frame(parent, style="Side.TFrame" if side else "TFrame")
        frame.grid(row=row, column=column, sticky="ew", padx=(0, 6), pady=(0, 5))
        ttk.Label(frame, text=label, style="SideMuted.TLabel" if side else "Muted.TLabel").pack(anchor="w")
        ttk.Entry(frame, textvariable=var, width=width).pack(fill="x", pady=(2, 0))

    def _workspace(self):
        work = ttk.Frame(self.root, padding=(18, 12, 18, 12))
        work.grid(row=1, column=1, sticky="nsew")
        ttk.Label(work, textvariable=self.run_context, style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
        self.tabs = ttk.Notebook(work)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.enable_traversal()
        self.overview = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.iterations = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.trajectory = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.convergence = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.work = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        for frame, title in (
                (self.overview, "Overview"), (self.iterations, "Iterations"),
                (self.trajectory, "Trajectories"), (self.convergence, "Convergence"),
                (self.work, "Work")):
            self.tabs.add(frame, text=title)
        self._overview_tab()
        self._iteration_tab()
        self._chart_tabs()
        self.tabs.bind("<<NotebookTabChanged>>", self._draw_visible_chart)

    def _overview_tab(self):
        cards = ttk.Frame(self.overview)
        cards.pack(fill="x", pady=(0, 11))
        labels = (("Converged", "Statuses from the solver"), ("Fewest updates", "Among converged methods"),
                  ("Reference minimum", "Independent stationarity check"))
        self.summaries = []
        for i, (title, hint) in enumerate(labels):
            cards.columnconfigure(i, weight=1, uniform="card")
            border = tk.Frame(cards, background=theme.PAPER, highlightbackground=theme.BORDER,
                              highlightthickness=1)
            border.grid(row=0, column=i, padx=(0, 9) if i < 2 else 0, sticky="nsew")
            box = ttk.Frame(border, style="White.TFrame", padding=(12, 10))
            box.pack(fill="both")
            large = tk.StringVar(self.root, value="-")
            note = tk.StringVar(self.root, value=hint)
            ttk.Label(box, text=title, style="Kicker.TLabel").pack(anchor="w")
            ttk.Label(box, textvariable=large, style="Stat.TLabel").pack(anchor="w", pady=(4, 1))
            ttk.Label(box, textvariable=note, style="CardHint.TLabel", wraplength=205).pack(anchor="w")
            self.summaries.append((large, note))

        row = ttk.Frame(self.overview)
        row.pack(fill="x", pady=(3, 7))
        ttk.Label(row, text="Method comparison", style="PageTitle.TLabel").pack(side="left")
        ttk.Label(row, text="Nf function calls · Nh Hessian calls", style="Muted.TLabel").pack(side="right")
        frame, self.summary_table = make_table(self.overview, SUMMARY_COLUMNS)
        frame.pack(fill="both", expand=True)
        self.summary_table.bind("<<TreeviewSelect>>", self._selected_run)
        self._detail_strip(self.overview, self.run_detail)

    def _iteration_tab(self):
        controls = ttk.Frame(self.iterations)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="A").pack(side="left", padx=(0, 5))
        self.combo_a = ttk.Combobox(controls, textvariable=self.selected_a, state="readonly", width=18)
        self.combo_a.pack(side="left", fill="x", expand=True)
        ttk.Label(controls, text="B").pack(side="left", padx=(10, 5))
        self.combo_b = ttk.Combobox(controls, textvariable=self.selected_b, state="readonly", width=18)
        self.combo_b.pack(side="left", fill="x", expand=True)
        for combo in (self.combo_a, self.combo_b):
            combo.bind("<<ComboboxSelected>>", self._populate_iterations)
        columns = [("k", "k", 47, "e")]
        frame, self.iter_table = make_table(self.iterations, columns)
        frame.pack(fill="both", expand=True)
        self.iter_table.bind("<<TreeviewSelect>>", self._inspect_iteration)
        self._detail_strip(self.iterations, self.iteration_detail)

    def _detail_strip(self, parent, variable):
        label = ttk.Label(parent, textvariable=variable, style="Muted.TLabel", justify="left",
                          padding=(4, 7, 4, 0))
        label.pack(fill="x")
        parent.bind("<Configure>", lambda event: label.configure(wraplength=max(240, event.width-12)),
                    add="+")

    def _chart_tabs(self):
        self.chart_hosts = {}
        for name, tab in (
                ("trajectories", self.trajectory), ("convergence", self.convergence), ("work", self.work)):
            bar = ttk.Frame(tab)
            bar.pack(fill="x", pady=(0, 8))
            title = {"trajectories": "Search paths", "convergence": "Progress by iteration",
                     "work": "Computational work"}[name]
            ttk.Label(bar, text=title, style="PageTitle.TLabel").pack(side="left")
            if name == "trajectories":
                self.focus_box = ttk.Combobox(bar, textvariable=self.plot_focus, state="readonly", width=20)
                self.focus_box.pack(side="right")
                self.focus_box.bind("<<ComboboxSelected>>", self._refresh_chart)
            if name == "work":
                metric = ttk.Combobox(bar, values=list(charts.METRICS), textvariable=self.cost_metric,
                                      state="readonly", width=23)
                metric.pack(side="right")
                metric.bind("<<ComboboxSelected>>", self._refresh_chart)
            holder = ttk.Frame(tab, style="White.TFrame")
            holder.pack(fill="both", expand=True)
            ttk.Label(holder, text="Run an experiment to see the chart.",
                      style="White.TLabel").pack(expand=True)
            self.chart_hosts[name] = holder

    def _footer(self):
        footer = ttk.Frame(self.root, style="Footer.TFrame", padding=(14, 6))
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(footer, textvariable=self.status_text, style="Footer.TLabel").pack(side="left", fill="x")
        self.progress = ttk.Progressbar(footer, length=125, mode="determinate")
        self.progress.pack(side="right")
        ttk.Label(footer, text="Ctrl+Enter run   ·   F11 full screen",
                  style="Footer.TLabel").pack(side="right", padx=14)

    def _set_start(self, point):
        for var, value in zip(self.start_vars, point):
            var.set(number(value))

    def _objective_changed(self, _event=None):
        self.formula_var.set(FORMULAS[OBJECTIVE_LABELS[self.objective_var.get()]])

    def _reset(self):
        self.objective_var.set(next(iter(OBJECTIVE_LABELS)))
        self._objective_changed()
        for name, var in self.coefficients.items():
            var.set(number(getattr(OBJECTIVES["f1"], name)))
        self._set_start(DEFAULT_SETTINGS.initial_point)
        self.tolerance_var.set(number(DEFAULT_SETTINGS.tolerance))
        self.max_iterations_var.set(str(DEFAULT_SETTINGS.max_iterations))
        self.method_preset.set("Compare all methods")
        self._preset_changed()
        self.advanced = {name: getattr(DEFAULT_SETTINGS, name) for name in ADVANCED_FIELDS
                         if hasattr(DEFAULT_SETTINGS, name)}
        self.advanced.update(repeats=1, local_radius=0.01)
        self.compare_adaptive = False

    def _preset_changed(self, _event=None):
        preset = self.method_preset.get()
        if preset in METHOD_PRESETS:
            self.selected_methods = METHOD_PRESETS[preset]
            self.method_count.set(f"{len(self.selected_methods)} selected")
        else:
            self._choose_methods()

    def _dialog(self, title):
        window = tk.Toplevel(self.root, background=theme.BASE)
        window.title(title)
        window.transient(self.root)
        window.resizable(False, False)
        window.bind("<Escape>", lambda event: window.destroy())
        window.grab_set()
        content = ttk.Frame(window, padding=17)
        content.pack(fill="both", expand=True)
        return window, content

    def _choose_methods(self):
        window, panel = self._dialog("Choose optimization methods")
        ttk.Label(panel, text="Methods for this experiment", style="PageTitle.TLabel").pack(anchor="w")
        variables = {key: tk.BooleanVar(window, value=key in self.selected_methods) for key in METHODS}
        for key, spec in METHODS.items():
            ttk.Checkbutton(panel, text=spec.label, variable=variables[key]).pack(anchor="w", pady=(3, 0))

        def save():
            chosen = tuple(key for key, value in variables.items() if value.get())
            if not chosen:
                messagebox.showerror("Choose methods", "Select at least one optimization method.", parent=window)
                return
            self.selected_methods = chosen
            preset = next((name for name, keys in METHOD_PRESETS.items() if keys == chosen),
                          "Custom selection")
            self.method_preset.set(preset)
            self.method_count.set(f"{len(chosen)} selected")
            window.destroy()

        self._dialog_actions(panel, window, save)

    def _advanced_settings(self):
        window, panel = self._dialog("Advanced settings")
        ttk.Label(panel, text="Solver and timing settings", style="PageTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        values = {name: tk.StringVar(window, value=number(value)) for name, value in self.advanced.items()}
        groups = (
            ("Step division", ("initial_step", "reduction_factor", "armijo_epsilon", "max_backtracks")),
            ("Golden Section", ("line_lower", "line_upper", "line_tolerance", "max_line_iterations")),
            ("Matrix and Broyden", ("matrix_rcond", "curvature_tolerance", "broyden_phi")),
            ("Timing and analysis", ("repeats", "local_radius")),
        )
        for index, (title, names) in enumerate(groups):
            group = ttk.LabelFrame(panel, text=title, padding=(10, 6, 8, 6))
            group.grid(row=1+index//2, column=index%2, sticky="nsew", padx=4, pady=5)
            group.columnconfigure(0, weight=1)
            for row, name in enumerate(names):
                self._field(group, ADVANCED_FIELDS[name][0], values[name], row, 0, width=18)
        adaptive = tk.BooleanVar(window, value=self.compare_adaptive)
        ttk.Checkbutton(panel, text="Also run adaptive phi for selected Broyden methods",
                        variable=adaptive).grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 2))
        ttk.Label(panel, text="Step bounds apply to Newton and quasi-Newton Golden Section.",
                  style="Muted.TLabel").grid(row=4, column=0, columnspan=2, sticky="w")
        footer = ttk.Frame(panel)
        footer.grid(row=5, column=0, columnspan=2, sticky="ew")

        def save():
            try:
                parsed = {key: parse(values[key].get(), cast, label)
                          for key, (label, cast) in ADVANCED_FIELDS.items()}
                solver = {key: value for key, value in parsed.items() if hasattr(DEFAULT_SETTINGS, key)}
                replace(DEFAULT_SETTINGS, **solver).validate()
                if parsed["repeats"] < 1 or parsed["local_radius"] <= 0:
                    raise ValueError("Timing repeats and local error radius must be positive.")
            except (ValueError, OverflowError) as error:
                messagebox.showerror("Invalid settings", str(error), parent=window)
                return
            self.advanced = parsed
            self.compare_adaptive = adaptive.get()
            window.destroy()

        self._dialog_actions(footer, window, save)

    @staticmethod
    def _dialog_actions(panel, window, action):
        actions = ttk.Frame(panel)
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="Apply", command=action).pack(side="right")
        ttk.Button(actions, text="Cancel", command=window.destroy).pack(side="right", padx=(0, 7))

    def _experiment(self):
        key = OBJECTIVE_LABELS[self.objective_var.get()]
        objective = replace(OBJECTIVES[key], **{
            name: parse(var.get(), float, f"coefficient {name}") for name, var in self.coefficients.items()
        })
        solver = {key: value for key, value in self.advanced.items() if hasattr(DEFAULT_SETTINGS, key)}
        settings = replace(DEFAULT_SETTINGS, **solver,
                           initial_point=tuple(parse(var.get(), float, name)
                                               for var, name in zip(self.start_vars, ("x0", "y0"))),
                           tolerance=parse(self.tolerance_var.get(), float, "tolerance"),
                           max_iterations=parse(self.max_iterations_var.get(), int, "outer iteration limit"))
        request = Experiment(objective, settings, self.selected_methods, self.advanced["repeats"],
                             self.compare_adaptive, self.advanced["local_radius"])
        request.validate()
        return request

    def start(self, _event=None):
        if self.cancel_event is not None:
            return
        try:
            experiment = self._experiment()
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Check the experiment inputs", str(error), parent=self.root)
            return
        stop = Event()
        self.cancel_event = stop
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.export_button.configure(state="disabled")
        self.progress.configure(maximum=1, value=0)
        self.status_text.set("Starting experiment")
        queue = self.event_queue

        def worker():
            try:
                session = execute(experiment, lambda *progress: queue.put(("progress", progress)), stop.is_set)
                queue.put(("finished", session))
            except CancelledError:
                queue.put(("cancelled", None))
            except Exception as error:
                queue.put(("error", str(error)))

        Thread(target=worker, name="optimization-experiment", daemon=True).start()

    def _poll(self):
        while True:
            try:
                kind, value = self.event_queue.get_nowait()
            except Empty:
                break
            if kind == "progress":
                index, total, label = value
                self.progress.configure(maximum=total, value=index)
                if not self.cancel_event.is_set():
                    self.status_text.set(f"Running {index+1}/{total} · {label}")
                continue
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.cancel_event = None
            if kind == "finished":
                self._present(value)
            elif kind == "cancelled":
                self.status_text.set("Run stopped")
            else:
                self.status_text.set("Experiment error")
                messagebox.showerror("Experiment error", value, parent=self.root)
            self.export_button.configure(state="normal" if self.session else "disabled")
            self.progress.configure(maximum=1, value=1 if kind == "finished" else 0)
        self.root.after(60, self._poll)

    def _cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self.status_text.set("Stopping after the active solver call")

    def _present(self, session):
        self.session = session
        experiment = session.experiment
        settings = experiment.settings
        coeffs = "Assignment coefficients" if experiment.objective == OBJECTIVES[experiment.objective.key] else (
            "Custom coefficients"
        )
        self.run_context.set(
            f"{experiment.objective.name}  ·  Start {coordinate_text(settings.initial_point)}"
            f"  ·  Tolerance {settings.tolerance:g}  ·  {coeffs}"
        )
        success = [run for run in session.runs if run.result.status == "converged"]
        count = len(session.runs)
        self.status_text.set(f"Finished · {len(success)} of {count} converged")
        self.summaries[0][0].set(f"{len(success)} / {count}")
        self.summaries[0][1].set(f"{count-len(success)} stopped")
        best = min(success, key=lambda run: run.result.iterations) if success else None
        self.summaries[1][0].set(str(best.result.iterations) if best else "-")
        self.summaries[1][1].set(best.short if best else "No converged methods")
        self.summaries[2][0].set(number(session.reference["value"]) if session.reference else "-")
        self.summaries[2][1].set(coordinate_text(session.reference["point"]) if session.reference
                                 else "No verified global reference for these coefficients")
        self.summary_table.delete(*self.summary_table.get_children())
        for i, run in enumerate(session.runs):
            row = run.summary()
            tags = ("stripe",) if i % 2 else ()
            if run.result.status != "converged":
                tags += ("stopped",)
            self.summary_table.insert("", "end", iid=run.id, tags=tags, values=(
                run.short, status_name(row["status"]), row["iterations"], number(row["objective"]),
                number(row["gradient_norm"], 4), row["function_evaluations"], row["hessian_evaluations"],
                f"{row['median_ms']:.3f}",
            ))
        self.summary_table.selection_set(session.runs[0].id)
        self._selected_run()
        self.run_labels = {run.short: run for run in session.runs}
        labels = list(self.run_labels)
        self.combo_a.configure(values=labels)
        self.combo_b.configure(values=("None", *labels))
        self.selected_a.set(labels[0])
        self.selected_b.set(labels[1] if len(labels) > 1 else "None")
        self._populate_iterations()
        self.plot_focus.set("All methods")
        self.focus_box.configure(values=("All methods", *labels))
        self._clear_plots()
        self._draw_visible_chart()

    def _selected_run(self, _event=None):
        selected = self.summary_table.selection()
        if not selected or self.session is None:
            return
        run = next((item for item in self.session.runs if item.id == selected[0]), None)
        if run is None:
            return
        outcome = run.result
        last = outcome.history[-1]
        counts = outcome.counts
        detail = f"  ·  {outcome.detail}" if outcome.detail else ""
        self.run_detail.set(
            f"{run.short}  ·  Last point {coordinate_text(last.point)}  ·  "
            f"{counts.gradient_evaluations} gradient calls  ·  {counts.linear_solves} solves  ·  "
            f"{counts.skipped_updates} skipped updates{detail}"
        )

    def _selected_runs(self):
        return (self.run_labels.get(self.selected_a.get()), self.run_labels.get(self.selected_b.get()))

    def _populate_iterations(self, _event=None):
        a, b = self._selected_runs()
        if not a:
            return
        columns = [("k", "k", 45, "e")]
        for letter in ("A", "B") if b else ("A",):
            for field, width in (("x", 87), ("y", 87), ("f(x)", 101), ("Step", 62)):
                columns.append((f"{letter}{field}", f"{letter} · {field}", width, "e"))
        self.iter_table.delete(*self.iter_table.get_children())
        configure_columns(self.iter_table, columns)
        runs = (a, b) if b else (a,)
        length = max(len(run.trace) for run in runs)
        for index in range(length):
            row = [index]
            for run in runs:
                item = run.trace[index] if index < len(run.trace) else {}
                row.extend(number(item.get(key), 7) for key in ("x", "y", "objective", "alpha"))
            self.iter_table.insert("", "end", iid=str(index), values=row, tags=("stripe",) if index % 2 else ())
        self.iter_table.selection_set("0")
        self._inspect_iteration()

    def _inspect_iteration(self, _event=None):
        selected = self.iter_table.selection()
        if not selected:
            return
        index = int(selected[0])
        lines = []
        for letter, run in zip(("A", "B"), self._selected_runs()):
            if run is None:
                continue
            if index >= len(run.trace):
                lines.append(f"{letter} · Finished at k={run.result.iterations}")
                continue
            item = run.trace[index]
            line = f"{letter} · ||g|| {number(item['gradient_norm'], 5)}"
            if item["point_error"] is not None:
                line += f"  ·  Reference error {number(item['point_error'], 5)}"
            if item["direction_x"] is not None:
                line += f"  ·  p {coordinate_text((item['direction_x'], item['direction_y']))}"
            if item["update_status"]:
                line += f"  ·  {status_name(item['update_status'])}  ·  phi {number(item['phi'], 5)}"
            lines.append(f"{line}  ·  Nf {item['function_evaluations']}")
        self.iteration_detail.set("\n".join(lines))

    def _current_chart(self):
        return {str(self.trajectory): "trajectories", str(self.convergence): "convergence",
                str(self.work): "work"}.get(self.tabs.select())

    def _clear_plots(self):
        for name in tuple(self.plot_cache):
            self._remove_plot(name)

    def _remove_plot(self, name):
        _, figure = self.plot_cache.pop(name)
        figure.clear()
        for child in self.chart_hosts[name].winfo_children():
            child.destroy()

    def _refresh_chart(self, _event=None):
        name = self._current_chart()
        if name in self.plot_cache:
            self._remove_plot(name)
        self._draw_visible_chart()

    def _draw_visible_chart(self, _event=None):
        name = self._current_chart()
        if not self.session or name is None or name in self.plot_cache:
            return
        holder = self.chart_hosts[name]
        for child in holder.winfo_children():
            child.destroy()
        figure = self._figure(name)
        canvas = FigureCanvasTkAgg(figure, master=holder)
        toolbar = NavigationToolbar2Tk(canvas, holder, pack_toolbar=False)
        toolbar.configure(background=theme.PAPER)
        toolbar.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()
        self.plot_cache[name] = canvas, figure

    def _figure(self, name):
        if name == "trajectories":
            selected = self.run_labels.get(self.plot_focus.get())
            return charts.trajectories(self.session, selected.id if selected else None)
        if name == "convergence":
            return charts.convergence(self.session)
        return charts.work(self.session, self.cost_metric.get())

    def _export(self):
        if self.session is None:
            return
        directory = filedialog.askdirectory(title="Choose a folder for the exported session", parent=self.root)
        if not directory:
            return
        self.status_text.set("Exporting CSV, settings and charts")
        self.root.update_idletasks()
        try:
            figures = {"trajectories": charts.trajectories(self.session), "convergence": charts.convergence(self.session),
                       "work": charts.work(self.session, self.cost_metric.get())}
            output = export_session(self.session, directory, figures)
        except (OSError, ValueError) as error:
            self.status_text.set("Export error")
            messagebox.showerror("Unable to export the session", str(error), parent=self.root)
            return
        self.status_text.set(f"Exported · {output.name}")
        messagebox.showinfo("Session exported", str(output), parent=self.root)

    def _toggle_fullscreen(self, _event=None):
        current = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current)

    def _leave_fullscreen(self, _event=None):
        self.root.attributes("-fullscreen", False)

    def _close(self):
        self._cancel()
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(f"Cannot open the Tk desktop window: {error}") from error
    OptimizationLab(root)
    root.mainloop()
