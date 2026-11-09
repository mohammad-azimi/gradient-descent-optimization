# Gradient descent for two unconstrained objectives

Practical project for Methods of Optimization and Decision Making at Peter the Great St Petersburg Polytechnic University. Group 1 uses the first data row of both assignment tables.

The project implements gradient descent with step division and gradient descent with a custom golden section line search. Both run on both objectives. SciPy is used only for independent reference and validation calculations.

## Windows CMD

Use Python 3.11 or newer. Extract this folder, open CMD inside it, then run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_experiments.py
python validate.py
python build_report.py
```

All work runs on the CPU. `run_experiments.py` saves figures without requiring an interactive window. Open PNG files from `results\figures` in VS Code or your image viewer. The included Word report describes the included result snapshot. After changing the settings, rerun both experiments and the report builder to keep them consistent.

Example commands:

```bat
python run_experiments.py --tolerance 1e-8 --output results_tight
python run_experiments.py --x0 0 0 --initial-step 0.5 --output results_origin
python run_experiments.py --function f1 --method golden --line-tolerance 1e-7 --no-sweeps --output results_f1
python run_experiments.py --help
```

The full report builder requires all four runs and their sweeps. Use the default command to recreate the complete report dataset. `validate.py` loads the settings from the chosen results folder, checks the algorithms and verifies the saved sweep statuses.

## Objectives

Both tables give `a=1, b=2, c=1, d=1, k=-1, m=2`.

- `f1(x,y) = x^2 + 2*y^2 + exp(-x-y) - x + 2*y`
- `f2(x,y) = x^2 + 2*y^2 + cos(x+y) - x + 2*y`

Both objectives have a unique global minimum. The strong convexity constants used for validation are 2 for f1 and `2-sqrt(2)` for f2. f1 has no global Lipschitz constant for its gradient, but it has a finite bound on each compact initial sublevel set. f2 has the global upper Hessian bound `4+sqrt(2)`.

## Lecture interpretation

- Slides 18 and 19 give the step division acceptance condition `f(x-alpha*g)-f(x) <= -epsilon*alpha*||g||^2`. The code reduces alpha only when this condition fails. The initial trial step resets to alpha0 at every outer iteration.
- Slide 18 prints the acceptance inequality and the reduction instruction together without a clear branch. Slide 19 states the acceptance condition more clearly. Reducing when the condition already holds would keep shrinking an acceptable step.
- Slide 24 combines the one dimensional minimization instruction with the previous backtracking scheme. Slide 25 gives a separate complete minimization algorithm without the extra backtracking stage. This implementation follows slide 25 for Method 2.
- Slides 24, 25 and 28 specify `0 < alpha < 1`. Golden section contracts an initial bracket with endpoints 0 and 1 and evaluates strictly interior points. It does not expand this interval. alpha0 belongs only to Method 1 in this implementation. lambda and epsilon do not enter Method 2.
- The stopping condition is the lecture's strict inequality `||gradient|| < delta`.
- No numerical initial point or step constants are prescribed in the files. The common starting point (2,2) and the defaults in `config.json` are experiment choices.
- The assignment DOCX contains the two formulas and parameter tables. The request to record and display all points and explain geometry is also based on the teacher's oral instructions supplied by the student.

The interval matters. An exact interior line minimum satisfies `g_next^T g_current = 0`. A boundary infimum on the open interval need not satisfy this equality. `boundary_example.csv` demonstrates this for f2. For f1 the global lower Hessian bound of 2 puts the unrestricted ray minimum at alpha <= 1/2, inside the lecture interval.

## Code map

| File | Purpose |
| --- | --- |
| `optimization/objectives.py` | Objective functions, analytical gradients, Hessians, and stable objective increments |
| `optimization/line_search.py` | Golden section implementation with cached interior values |
| `optimization/solvers.py` | Both descent methods, counters, statuses and iteration history |
| `optimization/reference.py` | Independent scalar reference equations, Lambert W and a stationary-system root check |
| `optimization/plots.py` | Contours, paths, surfaces, convergence and geometry figures |
| `run_experiments.py` | CLI, main runs, repeated timing and sensitivity experiments |
| `validate.py` | Independent mathematical and numerical checks |
| `build_report.py` | Word report generated from saved numerical outputs |

To add an objective, supply `key`, `value(point)`, `gradient(point)`, and `change(point, displacement, gradient)` to the solver. `change` returns the algebraically equivalent increment `f(point+displacement)-f(point)` in a numerically stable representation. Add `hessian` and `grid` for validation and plotting, and define an independent reference before comparing errors. The provided CLI exposes f1 and f2.

## Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `initial_point` | `[2,2]` | Shared starting point |
| `initial_step` | `0.8` | Method 1 initial trial alpha, reset each iteration |
| `reduction_factor` | `0.5` | Method 1 lambda |
| `armijo_epsilon` | `0.1` | Method 1 sufficient decrease constant |
| `tolerance` | `1e-6` | Gradient norm stopping threshold delta |
| `max_iterations` | `10000` | Outer update limit |
| `line_tolerance` | `1e-10` | Absolute width tolerance in alpha for Golden Section |
| `max_line_iterations` | `200` | Golden Section contraction limit per step |
| `max_backtracks` | `100` | Allowed reductions per Method 1 step |

The lecture restricts alpha0, lambda and epsilon to (0,1). The CLI validates these restrictions. The line search bracket is fixed by the lecture and is not replaced by an expanding bracket.

## Numerical accounting

`iterations` counts accepted updates. Row 0 is the initial point, so each successful run has `iterations+1` saved points. Every row stores the coordinates, objective, gradient components and norm, incoming alpha, step norm, actual objective increment, inner search work, successive gradient dot product, cosine, angle and cumulative counters. Incoming quantities are zero on row 0. Undefined angles appear as empty CSV cells.

One scalar objective oracle evaluation means one direct objective value or one algebraically equivalent stable objective increment. Every rejected trial and every newly sampled Golden Section point counts. Two initial Golden Section interior values are evaluated, then exactly one new value per contraction. Cached values are reused. The first objective value and a final direct verification are counted, giving:

- Step Division: `Nf = 2 + sum(1 + reductions_at_each_accepted_step)` for successful runs.
- Golden Section: `Nf = 2 + sum(2 + golden_contractions_at_each_step)` for successful runs.
- Both methods: `Ng = accepted_updates + 1`.

The initial and final direct evaluations both count even if the algorithm takes zero updates. The scalar stable increment uses the already available current gradient and does not secretly compute another gradient. Small exponential and sine remainders use Taylor series to avoid cancellation. These are evaluations of the same objective, with no change to the acceptance inequality. The last trace value is replaced by the final directly evaluated objective. References, plot grids, derivative diagnostics, validation runs, timing repetitions and sweeps have separate scopes and are excluded from each main run's counters. Hessians are never used by either solver.

Elapsed time uses `perf_counter` inside the solver and includes history and counter bookkeeping. It excludes imports, I/O, plotting and independent references. Each main run has its own measured time. For a less noisy comparison, the CSV also gives the median of 31 additional executions after one warm-up. All raw timing samples and environment versions are saved. Timings depend on the computer and do not predict the Windows timings exactly.

## Outputs

- `results/summary.csv` and `results/experiment.json`: all four main results and references.
- `results/histories/*.csv`: every saved iterate.
- `results/benchmark_times.csv`: raw repeated timings.
- `results/tolerance_sweep.csv`: three gradient tolerances, both functions and methods.
- `results/step_sweep.csv`: four alpha0 values for Method 1.
- `results/line_accuracy_sweep.csv`: three Golden Section tolerances.
- `results/initial_point_sweep.csv`: four common initial points.
- `results/boundary_example.*`: a line minimum near the upper interval boundary.
- `results/validation.json`: independent validation results.
- `results/figures/*.png`: 13 scientific figures.
- `report/Gradient_Descent_Group1_Report.docx`: English report.
- `presentation.txt`: short English presentation and suggested figures.

The report images showing mathematical notation are included in `report/assets/equations`. `report_equations.py` can regenerate them using Matplotlib. Matrices in the Word report use editable native equations. The report builder does not require Node.js or an equation editor. The teacher's files are preserved in `references`.
