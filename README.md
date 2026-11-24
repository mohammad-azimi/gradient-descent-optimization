# Unconstrained Multidimensional Optimization

Practical project for Methods of Optimization and Decision Making at Peter the Great St. Petersburg Polytechnic University. Group 1 uses the first parameter row of both assignment objectives. The comparison covers an exponential and a cosine objective, each with a unique global minimizer.

## Desktop app

Open **Optimization Lab** to compare methods from a visual interface. It starts with F1 at the origin and compares all ten configurations. The form accepts the coefficients of F1 or F2, a new starting point, the gradient tolerance and the iteration limit. Select a method preset or choose individual methods. **Advanced settings** exposes the line search, matrix and Broyden parameters, optional adaptive Broyden runs and timing repeats.

From the project directory in Windows CMD, launch the window using the existing environment.

```cmd
.venv\Scripts\python.exe run_gui.py
```

You can also open `launch_optimizer_gui.cmd` from the project folder. The desktop app uses Tkinter and the existing project dependencies. On a Python installation without Tk support, install the optional Tcl/Tk component for that Python interpreter.

| Tab | What it shows |
| --- | --- |
| Overview | Termination, final objective and gradient norm, iteration count, evaluation counts and solver time |
| Iterations | Every point starting from `X_0` with two selected runs aligned by iteration. Select a row to inspect the gradient, search direction and update details. |
| Trajectories | Search paths over objective contours. The method selector isolates one path and the toolbar supports zoom, pan and saving the current plot. |
| Convergence | Objective value, gradient norm, reference error or step length, and step size |
| Work | A comparison of selected operation counts or median solver time |

For the live demonstration, select **BFGS pair**, click the **0, 10** starting point, and run. The fixed-step run stops after three accepted updates on F1. Golden Section converges in five. Open **Iterations** to compare the two paths. Press **F11** for full screen and **Ctrl+Enter** to run.

With custom coefficients, a reference minimum is shown when the app can verify strong convexity and solve the stationarity equation. Each run shows the actual termination status. **Export session** writes the input settings, full iteration histories, summary, timing samples and three PNG charts into a new subdirectory of a folder you choose.

## Method configurations

| CLI method | Search direction | Step rule |
| --- | --- | --- |
| `division` | Negative gradient | Original sufficient decrease and step division |
| `golden` | Negative gradient | Original Golden Section on the open interval (0, 1) |
| `newton_inverse_fixed` | Explicit inverse of the analytical Hessian | Unit |
| `newton_inverse_golden` | Explicit inverse of the analytical Hessian | Closed Golden Section |
| `newton_solve_fixed` | Linear solve with the analytical Hessian | Unit |
| `newton_solve_golden` | Linear solve with the analytical Hessian | Closed Golden Section |
| `bfgs_fixed` | Linear solve with a BFGS Hessian approximation | Unit |
| `bfgs_golden` | Linear solve with a BFGS Hessian approximation | Closed Golden Section |
| `broyden_fixed` | Linear solve with a Broyden Hessian approximation, φ = 0.5 | Unit |
| `broyden_golden` | Linear solve with a Broyden Hessian approximation, φ = 0.5 | Closed Golden Section |

Every main run stops on a gradient norm strictly below 1e-6 or records a termination status and the last valid iterate. Closed Golden Section compares both endpoints on [1e-8, 1] by default. The bracket bounds, width tolerance, outer tolerance and iteration limits are independent. The constant-step configurations always use α = 1.

The Broyden formula uses Hessian notation. Parameter zero gives BFGS and parameter one gives DFP. The proposed value 0.5 is a restricted Broyden-class member. Its rank-one correction is orthogonal to the secant displacement, which preserves the secant equation. The optional sensitivity comparison bounds an adaptive parameter to [0.1, 0.9] using the relative mismatch between the observed gradient change and the model prediction. The use of the Broyden class follows the reconstructed lecture explanation and Nocedal and Wright, [Numerical Optimization, second edition](https://doi.org/10.1007/978-0-387-40065-5), §6.3.

## Windows CMD

Python 3.11 or newer is required. Run these commands from the repository root for a new setup.

```cmd
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the existing virtual environment when dependencies are already installed. The tests compare both Newton directions, check the Hessian BFGS and DFP endpoints, verify the closed line interval, and match the original gradient descent snapshots.

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe validate_comparison.py --check-only
```

Run the complete comparison in a separate directory, validate the exported files, and rebuild the updated Word report.

```cmd
set "GD_RUN=%TEMP%\gradient-comparison-%RANDOM%-%RANDOM%"
.venv\Scripts\python.exe run_comparison.py --output "%GD_RUN%"
.venv\Scripts\python.exe validate_comparison.py --results "%GD_RUN%"
.venv\Scripts\python.exe build_report.py --comparison "%GD_RUN%" --output "%GD_RUN%\Gradient_Descent_Group1_Report.docx"
start "" "%GD_RUN%\Gradient_Descent_Group1_Report.docx"
```

For a quick demonstration, select an objective, method and starting point to print the full sequence of iterates. The command below writes its own output in the Windows temporary directory.

```cmd
.venv\Scripts\python.exe run_comparison.py --function f1 --method newton_solve_fixed --x0 0 10 --show-iterates --no-plots --benchmark-repeats 3 --output "%TEMP%\gd-newton-demo"
```

Use `run_comparison.py --help` for the interval bounds, tolerances, benchmark repetitions and `--no-adaptive`. The original `run_experiments.py` still accepts `division` and `golden` and writes the original output schema.

## Published comparison

The 60 primary cases combine ten methods, two objectives and the three starting points (0, 0), (0, 10) and (10, 0). Fifty-eight converged. Both failed runs used unit-step quasi-Newton on F1 from (0, 10). The first trial moved to (1, −32), where the exponential raises the objective from about 220 to 2.90e13. Each run retained its nonfinite-objective status. Both Golden Section configurations converged in 5 updates from the same point.

| Example | Updates | Function calls | Hessian calls |
| --- | ---: | ---: | ---: |
| F1 from (10, 0), Newton solve with unit steps | 4 | 5 | 4 |
| F1 from (10, 0), Newton solve with Golden Section | 3 | 160 | 3 |
| F2 from (0, 10), original step division | 35 | 62 | 0 |
| F2 from (0, 10), BFGS with Golden Section | 7 | 372 | 0 |

The two Newton direction calculations gave the same iteration counts for every matched case. All four Newton configurations converged in all six cases. The adaptive parameter was tested in 12 additional runs. One converged matched case improved from 7 to 6 updates, and one worsened from 17 to 41. All six adaptive Golden Section counts matched their constant-parameter counterparts.

Recorded timings are medians from 31 repetitions per configuration after a warm-up. Costs include solver bookkeeping and numerical safeguards. Timing differences in two dimensions are sensitive to the machine and operating system.

| File or directory | Contents |
| --- | --- |
| [comparison_results/summary.csv](comparison_results/summary.csv) | All 60 main cases with work counters, error and final status |
| [comparison_results/adaptive_comparison.csv](comparison_results/adaptive_comparison.csv) | Matched constant and adaptive parameter cases |
| [comparison_results/histories](comparison_results/histories) | Every accepted point, incoming direction, step, error and update diagnostics |
| [comparison_results/convergence_rates.csv](comparison_results/convergence_rates.csv) | Local order observations above the error reliability floor |
| [comparison_results/figures](comparison_results/figures) | Trajectories, convergence, cost and the local Newton model |
| [comparison_results/validation.json](comparison_results/validation.json) | Export validation and independent line-step checks |
| [report/Gradient_Descent_Group1_Report.docx](report/Gradient_Descent_Group1_Report.docx) | Comparative report in the original Polytech format |
| [presentation_comparison.md](presentation_comparison.md) | English speaking notes, demonstration commands and questions |
| `results/` and `demo_*/` | Original baseline snapshots and demonstrations |

## Reuse from Python

The public `minimize` function accepts any finite, nonempty initial vector. Provide an objective with a `key`, `value(point)` and `gradient(point)`. Newton also needs `hessian(point)`. If supplied, `change(point, displacement, gradient)` should return an accurate scalar objective increment. The examples and CLI plots use two-dimensional objectives. Dense Newton and Hessian-form quasi-Newton require O(n²) matrix storage and dense linear solves, so a large neural-network experiment would need another strategy.

```python
from optimization import OBJECTIVES, OptimizationSettings, minimize

settings = OptimizationSettings(initial_point=(0, 10), tolerance=1e-6)
run = minimize(OBJECTIVES["f1"], "bfgs_golden", settings)
print(run.status, run.history[-1].point, run.counts)
```

## Original gradient descent workflow

Use Python 3.11 or newer. Extract this folder, open CMD inside it, then run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_experiments.py
python validate.py
python build_report.py --output "report\Gradient_Descent_Baseline_Report.docx"
```

All work runs on the CPU. `run_experiments.py` saves figures without requiring an interactive window. Open PNG files from `results\figures` in VS Code or your image viewer. The baseline-only report reads the original results snapshot. The comparative report also uses the new comparison results.

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
