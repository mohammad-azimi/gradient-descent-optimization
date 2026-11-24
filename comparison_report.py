from collections import Counter
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from optimization import METHODS
from optimization.exports import read_csv
from polytech_theme import restyle_report
from report_equations import EQUATIONS, render_equations


TITLE = "Comparative Analysis of First Order Second Order and Quasi Newton Methods"
MORE_EQUATIONS = {
    "taylor": r"f(x_k+p)\approx f(x_k)+g_k^Tp+\frac{1}{2}p^TH_kp",
    "newton": r"g_k+H_kp_k=0\quad\Longrightarrow\quad H_kp_k=-g_k,\quad p_k=-H_k^{-1}g_k",
    "newton_descent": r"g_k^T p_k=-g_k^T H_k^{-1}g_k<0\quad(g_k\ne 0)",
    "closed_line": r"\alpha_k\approx\arg\min_{\alpha\in[\varepsilon_\alpha,\,1]}f(x_k+\alpha p_k)",
    "secant": r"s_k=x_{k+1}-x_k,\quad y_k=g_{k+1}-g_k,\quad B_{k+1}s_k=y_k",
    "bfgs": r"B_{k+1}^{\mathrm{BFGS}}=B_k-\frac{(B_ks_k)(B_ks_k)^T}{s_k^TB_ks_k}+\frac{y_ky_k^T}{y_k^Ts_k}",
    "broyden": r"B_{k+1}=B_{k+1}^{\mathrm{BFGS}}+\phi_k(s_k^TB_ks_k)w_kw_k^T",
    "w": r"w_k=\frac{y_k}{y_k^Ts_k}-\frac{B_ks_k}{s_k^TB_ks_k},\quad w_k^Ts_k=1-1=0",
    "curvature": r"y_k^Ts_k=\int_0^1 s_k^TH(x_k+t s_k)s_k\,dt\geq\mu\Vert s_k\Vert^2",
    "adaptive": r"\phi_k=0.1+0.8\,\frac{\Vert y_k-B_ks_k\Vert}{\Vert y_k\Vert+\Vert B_ks_k\Vert}",
    "order": r"e_k=\Vert x_k-x^*\Vert,\quad \widehat{p}_k=\frac{\log(e_k/e_{k-1})}{\log(e_{k-1}/e_{k-2})}",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def new_page(report, heading):
    report.heading(heading).paragraph_format.page_break_before = True


def report_body(report, baseline, results, data, main, adaptive, validation):
    refs = data["reference_solutions"]
    settings = data["settings"]
    figures = results / "figures"
    recorded = {(r["function"], r["start_id"], r["method"]): r for r in main}
    status = Counter(r["status"] for r in main)
    pairs = read_csv(results / "adaptive_comparison.csv")
    wins = sum(p["adaptive_status"] == p["constant_status"] == "converged"
               and int(p["adaptive_iterations"]) < int(p["constant_iterations"]) for p in pairs)
    losses = sum(p["adaptive_status"] == p["constant_status"] == "converged"
                 and int(p["adaptive_iterations"]) > int(p["constant_iterations"]) for p in pairs)
    times = {
        method: [r for r in main if r["method"] == method and r["status"] == "converged"]
        for method in METHODS
    }
    median_times = {method: median(float(r["median_ms"]) for r in rows)
                    for method, rows in times.items() if rows}

    report.heading("Objective", 2)
    report.p(
        f"We compare ten configurations of gradient descent, Newton and quasi-Newton methods on two "
        f"strongly convex objectives. The experiment uses three initial points and one strict gradient "
        f"tolerance. {status['converged']} of {len(main)} runs converged. The four Newton configurations "
        f"converged in all assigned cases. The two nonconverged runs used unit-step quasi-Newton updates "
        f"on the exponential objective starting at (0, 10)."
    )
    report.p(
        f"We compare the Hessian BFGS update with a Broyden class update at φ = {settings['broyden_phi']:g}, "
        f"and test an adaptive rule based on secant mismatch separately. The study reports the complete trajectories, "
        f"evaluation counts, repeated timings and error relative to independently verified minimizers."
    )
    report.p("Saint Petersburg")

    report.heading("1 Problem Formulation")
    report.p(
        "We minimize two smooth functions of two real variables. Both coefficient sets use the first row "
        "of the assignment for practical subgroup 1 [1]. The original baseline objectives and derivatives "
        "are used throughout the comparison."
    )
    report.table(
        ["Parameter row", "a", "b", "c", "d", "k", "m"],
        [["First in both tables", 1, 2, 1, 1, "−1", 2]],
        [1.85, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77],
    )
    report.eq("f1")
    report.eq("f2")
    report.heading("Analytical derivatives", 2)
    report.eq("grad1")
    report.eq("hess1")
    report.eq("grad2")
    report.eq("hess2")

    new_page(report, "2 Convexity and Reference Solutions")
    report.p(
        "A uniformly positive lower Hessian bound is stronger than positive definiteness at each point. "
        "For the exponential objective, the Hessian is diag(2, 4) plus a positive semidefinite rank-one term. "
        "The cosine objective lies between the two symmetric matrices shown below."
    )
    report.eq("bound1")
    report.eq("bound2")
    report.eq("constants")
    report.p(
        "The lower matrix for F2 has eigenvalues 2 ± √2. The upper matrix has eigenvalues 4 ± √2, "
        "so the gradient of F2 is globally Lipschitz continuous. The gradient of F1 has a finite "
        "Lipschitz bound on each compact sublevel set and its bounded neighbourhood. Its Hessian can "
        "grow arbitrarily large across the full domain. Strong convexity and coercivity give one "
        "global minimizer per objective."
    )
    report.eq("coercive")
    report.heading("Independent stationarity checks", 2)
    report.eq("ref1")
    report.eq("scalar1")
    report.p(
        "On the exponential problem, the scalar equation is strictly increasing for positive t. "
        "We compute its root by Brent's method and check it using the principal Lambert W function. "
        "On the cosine problem, the derivative of s − 0.75 sin(s) is at least 0.25, so zero is its only root. "
        "A separate two-dimensional root calculation checks both references."
    )
    report.eq("ref2")
    report.table(
        ["Objective", "Reference x", "Reference y", "Minimum value"],
        [[f.upper(), f"{v['point'][0]:.10f}", f"{v['point'][1]:.10f}", f"{v['value']:.12f}"]
         for f, v in refs.items()], [1.1, 1.7, 1.7, 2.2],
    )
    report.eq("error")
    report.p("The error bounds follow from strong convexity and apply to each finite computed point.")

    new_page(report, "3 First Order Baselines")
    report.eq("update")
    report.p(
        "Step Division resets the trial step to 0.8 and halves it until the sufficient decrease condition "
        "is satisfied, with coefficient 0.1. The condition and the reduction rule follow the original "
        "first-order lecture [2, slides 18–19]."
    )
    report.eq("armijo")
    report.p(
        "The first-order Golden Section method follows the complete algorithm on slide 25. "
        "The bracket begins at zero and one and uses interior points only. A one-sided boundary infimum "
        "can be approached when the unrestricted line minimizer lies outside this original interval."
    )
    report.eq("golden")
    report.heading("4 Newton Directions and Step Selection")
    report.eq("taylor")
    report.p(
        "The local second-order Taylor model has gradient gₖ + Hₖp with respect to p. "
        "Setting it to zero gives the Newton system. Positive definiteness makes the model strictly "
        "convex in p and gives a descent direction at each nonstationary point of either objective."
    )
    report.eq("newton")
    report.eq("newton_descent")
    report.p(
        "Direction A forms the dense inverse of Hₖ and multiplies it by −gₖ. Direction B solves the linear "
        "system for the single right-hand side. Both target the same direction. A direct solve avoids "
        "forming an entire inverse and the rounding associated with multiplying it by the gradient."
    )
    report.p(
        "The unit-step configurations take αₖ = 1 at every update. The second strategy minimizes over "
        f"the closed interval [{settings['line_lower']:.0e}, {settings['line_upper']:g}] in the main experiment. "
        f"The Golden Section width tolerance is {settings['line_tolerance']:.0e}; both endpoints are compared "
        "with the best interior candidate. This admits the exact full Newton step when it is the line minimum."
    )
    report.eq("closed_line")

    new_page(report, "5 Quasi Newton Updates")
    report.p(
        "The approximate Hessian B starts as the identity. The solver computes Bₖpₖ = −gₖ. "
        "The change in gradient over the actual displacement records the action of an average Hessian "
        "along that displacement and supplies the secant condition."
    )
    report.eq("secant")
    report.eq("bfgs")
    report.eq("broyden")
    report.eq("w")
    report.p(
        "Multiplying the BFGS formula by sₖ cancels Bₖsₖ and leaves yₖ. The extra Broyden term also "
        "vanishes when multiplied by sₖ. Both corrections are symmetric. For positive curvature, "
        "BFGS and DFP preserve positive definiteness. The Hessian Broyden update is their convex "
        "combination for 0 ≤ φ ≤ 1 [3, §6.3, equations (6.32)–(6.33)]."
    )
    report.eq("curvature")
    report.p(
        f"Here φ = 0 gives BFGS, φ = 1 gives DFP, and φ = {settings['broyden_phi']:g} gives the tested "
        f"intermediate member. We use the Hessian notation and retain both assigned step strategies "
        "for each update family [4]."
    )
    report.heading("Numerical checks in the solver", 2)
    report.p(
        "The solver checks curvature, symmetry, positive definiteness, a reciprocal condition "
        "threshold and the relative secant residual. The curvature tests use relative inner-product "
        "thresholds plus a minimum positive denominator. A rejected correction keeps the previous "
        "matrix and records the reason. Newton terminates on an unusable Hessian. The unit-step "
        "configurations retain α = 1 even when finite trial values rise."
    )
    report.eq("adaptive")
    report.p(
        "A separate sensitivity run increases φ when the secant mismatch is large relative to "
        "the magnitude of both gradient-change vectors. The triangle inequality bounds this "
        "choice between 0.1 and 0.9 on valid updates."
    )

    new_page(report, "6 Software and Experimental Design")
    report.p(
        "The shared minimize interface selects a direction and a step rule through a method registry. "
        "It accepts a vector-valued point and an objective with value and gradient callables. Newton "
        "also receives a Hessian callable. The assignment functions provide a stable objective "
        "increment for line searches. Other objectives can rely on direct value differences."
    )
    report.table(["Component", "Responsibility"], [
        ["objectives.py and reference.py", "Analytical derivatives and independent reference minima"],
        ["solvers.py and line_search.py", "Original descent methods and shared scalar search"],
        ["minimizer.py and newton.py", "Unified interface, Newton directions and histories"],
        ["quasi_newton.py", "Hessian BFGS and restricted Broyden updates"],
        ["run_comparison.py and validate_comparison.py", "Repeat runs, export metrics and verify saved data"],
        ["build_report.py", "Rebuild the report from the recorded experiments"],
    ], [3.0, 3.75])
    report.p(
        "The main experiment crosses ten configurations, two objectives and the starting points "
        "(0, 0), (0, 10) and (10, 0). Both first-order algorithms retain their original settings. "
        "All configurations apply the strict gradient stopping condition and an outer cap of "
        f"{settings['max_iterations']:,} iterations. The inner search cap is {settings['max_line_iterations']}."
    )
    report.eq("stop")
    report.p(
        f"The comparison uses δ = {settings['tolerance']:.0e}. The outer rule, the closed interval's lower "
        "bound and the scalar width tolerance are independent settings. The exported trace includes "
        "each accepted xₖ, the incoming direction and step, the gradient, objective value, reference error "
        "and diagnostics. A failure row reports the last finite iterate and the operations attempted."
    )
    report.heading("Evaluation and timing conventions", 2)
    report.p(
        "Each new objective sample or stable objective increment counts once. The legacy solvers "
        "accumulate increments and verify the final objective directly. The new solvers evaluate the "
        "objective at each accepted point and count both endpoint trials for closed Golden Section. "
        "A failed attempt also counts any evaluations it performs. Linear solves, explicit inversions, "
        "Hessian evaluations and spectral checks have separate counters."
    )
    report.p(
        f"Each case has one warm-up and {data['benchmark_repeats']} timed repetitions. Solver time includes "
        "diagnostics and history creation. Reference construction, plots, export and the adaptation of "
        f"legacy result records occur outside the timer. The saved runs used Python {data['environment']['python']} "
        f"and NumPy {data['environment']['numpy']} on {data['environment']['platform'].split('-')[0]}."
    )
    report.p(
        "Stable line increments use algebraically equivalent expressions with Taylor remainders "
        "for small exponential and trigonometric differences. This reduces cancellation near a "
        "minimum. The accompanying CSV files retain more digits than the tables for numerical checking."
    )

    new_page(report, "7 Numerical Results")
    report.p(
        "The table gives accepted updates. An asterisk denotes a run ending on a nonfinite trial "
        "objective. The complete summary CSV also records final values, residuals and terminal causes."
    )
    iteration_table = []
    for method, spec in METHODS.items():
        cells = [spec.short]
        for function in ("f1", "f2"):
            for start in ("s1", "s2", "s3"):
                row = recorded[function, start, method]
                cells.append(row["iterations"] + ("" if row["status"] == "converged" else "*"))
        iteration_table.append(cells)
    report.table(["Method", "F1\n(0,0)", "F1\n(0,10)", "F1\n(10,0)",
                  "F2\n(0,0)", "F2\n(0,10)", "F2\n(10,0)"],
                 iteration_table, [1.65, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
    report.p("The median work below uses converged cases. The success column gives the denominator.")
    cost_table = []
    for method, completed in times.items():
        counts = [median(int(r[k]) for r in completed)
                  for k in ("iterations", "function_evaluations", "gradient_evaluations",
                            "hessian_evaluations", "matrix_inversions", "linear_solves")]
        cost_table.append([METHODS[method].short, f"{len(completed)}/6",
                           *(f"{value:g}" for value in counts),
                           f"{median(float(r['median_ms']) for r in completed):.3f}"])
    report.table(
        ["Method", "Success", "k", "Nf", "Ng", "Nh", "Inv", "Solve", "ms"],
        cost_table, [1.40, 0.74, 0.54, 0.61, 0.60, 0.60, 0.58, 0.69, 0.84],
    )
    winners = [r for r in main if r["status"] == "converged"]
    max_error = max(float(r["point_error"]) for r in winners)
    report.p(
        f"Across converged cases, the largest distance to a reference solution was {max_error:.2e}. "
        f"The two Newton direction calculations agreed on termination and update count in every "
        f"matched pair. The maximum distance between corresponding Newton iterates was "
        f"{validation['max_newton_trajectory_distance']:.2e}."
    )

    new_page(report, "8 Trajectories and Step Geometry")
    report.figure(
        figures / "paths_f1_s2.png",
        "Figure 1. Four method groups on the exponential objective from (0, 10). "
        "The square outline marks a final iterate inside the plotted window.",
        width=6.55,
    )
    report.p(
        "The gradient is normal to a regular level curve. Newton and quasi-Newton directions "
        "generally turn away from that normal because they use curvature. At an exact interior "
        "line minimum, the new gradient is orthogonal to the previous direction. This implies "
        "orthogonal successive gradients for steepest descent. A boundary minimum may have "
        "a nonzero directional derivative, so the same test does not apply there."
    )
    report.p(
        "The two unit-step quasi-Newton paths leave the visible range on F1. The saved histories "
        "show the full excursion and its final status. The four displayed groups use the same "
        "contour axes. The equivalent explicit-inverse Newton trajectories are available in the CSV files."
    )

    new_page(report, "9 Convergence Rates")
    report.eq("order")
    report.p(
        "Linear convergence has an eventual contraction bounded by a constant below one. "
        "For superlinear convergence the ratio eₖ₊₁/eₖ tends to zero. Quadratic convergence "
        "bounds eₖ₊₁ by a constant times eₖ². We record ratios above the reliability floor "
        "and estimate order from decreasing triples inside the local error radius "
        f"{data['local_error_radius']:g}."
    )
    report.figure(
        figures / "convergence_f1_s1.png",
        "Figure 2. Objective gap, gradient norm, point error and step size for F1 from the origin. "
        "The logarithmic curves omit roundoff-scale gaps and unreliable point errors.",
        width=6.55,
    )
    trace = read_csv(results / recorded["f1", "s1", "newton_solve_fixed"]["history_csv"])
    q = trace[-2:]
    bfgs = read_csv(results / recorded["f1", "s1", "bfgs_golden"]["history_csv"])
    report.p(
        f"For unit-step Newton on F1 from (0, 0), the last two quadratic ratios were "
        f"{float(q[0]['quadratic_ratio']):.3f} and {float(q[1]['quadratic_ratio']):.3f}. "
        f"For BFGS Golden Section on the same case, the last two consecutive-error ratios were "
        f"{float(bfgs[-2]['error_ratio']):.3f} and {float(bfgs[-1]['error_ratio']):.3f}. "
        "These short sequences are consistent with fast local convergence and give limited evidence "
        "about asymptotic order."
    )
    report.p(
        "For F2 at the origin, x + y remains zero on the first Newton line and the objective along "
        "that line is quadratic. Newton reaches the reference in one step in this case, so an empirical "
        "order is undefined. The recorded ratios remain descriptive where the error has reached "
        "floating-point resolution."
    )

    new_page(report, "10 Efficiency and Stability")
    report.figure(
        figures / "cost_comparison.png",
        "Figure 3. Median accepted updates, objective calls, Hessian calls and execution time. "
        "Labels include the number of converged cases.",
        width=6.55,
    )
    unit = recorded["f1", "s3", "newton_solve_fixed"]
    golden = recorded["f1", "s3", "newton_solve_golden"]
    failure = read_csv(results / recorded["f1", "s2", "bfgs_fixed"]["history_csv"])
    report.p(
        f"For F1 from (10, 0), Newton with the linear solve used {unit['iterations']} updates and "
        f"{unit['function_evaluations']} objective calls with unit steps, compared with "
        f"{golden['iterations']} updates and {golden['function_evaluations']} calls with Golden Section. "
        "The extra scalar search can outweigh an improvement in outer updates."
    )
    report.p(
        f"For F1 from (0, 10), the first quasi-Newton unit step raised the objective from "
        f"{float(failure[0]['objective']):.2f} to {float(failure[1]['objective']):.2e}. "
        "The first correction was rejected by the conditioning check. Both fixed-step updates later "
        "encountered a nonfinite objective. The Golden Section variants converged from the same point."
    )
    report.p(
        "A dense Newton or quasi-Newton direction uses O(n²) storage and a dense system solve costs "
        "O(n³). An explicit inverse has the same asymptotic order and computes additional columns. "
        "A BFGS or Broyden rank update costs O(n²) before matrix safeguards and eliminates analytic "
        "Hessian evaluations. This implementation still solves dense systems and checks their spectra. "
        "The extra safeguards are reflected in the measured solver times."
    )

    new_page(report, "11 Adaptive Parameter Comparison")
    report.figure(
        figures / "quasi_newton_comparison.png",
        "Figure 4. Matched BFGS, constant Broyden and adaptive Broyden trials. "
        "An × identifies a nonconverged run.",
        width=6.55,
    )
    changed = [
        f"{p['function'].upper()} from ({float(p['initial_x']):g}, {float(p['initial_y']):g}) "
        f"with {'unit steps' if p['method'].endswith('fixed') else 'Golden Section'} changed "
        f"from {p['constant_iterations']} to {p['adaptive_iterations']} updates."
        for p in pairs if p["constant_status"] == p["adaptive_status"] == "converged"
        and p["constant_iterations"] != p["adaptive_iterations"]
    ]
    report.p(
        f"Under the adaptive rule, iteration count fell in {wins} and rose in {losses} of the "
        "matched converged pairs. " + " ".join(changed) +
        " The constant rule remains the proposed method in the main table."
    )
    report.heading("12 Conclusions and Limitations")
    fastest = min(median_times, key=median_times.get)
    report.p(
        f"{METHODS[fastest].label} gave the lowest median solver time across the main "
        "configuration groups in this recording. Small timing differences deserve replication. "
        "Closed Golden Section stabilized both quasi-Newton "
        "methods on the difficult exponential start. The main comparison measures the trade-off "
        "between iterations, function calls, analytical derivatives and failure risk in two dimensions."
    )
    report.p(
        "Exact Newton is locally quadratic under a Lipschitz Hessian near a nonsingular minimum. "
        "BFGS superlinear convergence requires further assumptions on the iterates and step rule [3]. "
        "These fixed and bounded line steps are evaluated empirically. Strong convexity guarantees "
        "uniqueness of the minimizer and positive secant curvature in exact arithmetic. It does not "
        "give a general convergence guarantee for arbitrary fixed steps."
    )

    new_page(report, "Appendix Original Gradient Descent Experiments")
    old = read_csv(baseline / "summary.csv")
    report.p(
        "The original comparison starts at (2, 2). Its main results and trajectories serve as "
        "regression checks alongside the three demonstration starting points."
    )
    report.table(
        ["Objective", "Method", "Updates", "Nf", "Ng", "Final value"],
        [[r["function"].upper(), r["method"], r["iterations"], r["function_evaluations"],
          r["gradient_evaluations"], f"{float(r['minimum_value']):.10f}"] for r in old],
        [0.9, 2.25, 0.83, 0.65, 0.65, 1.42],
    )
    report.p(
        "The original runs and sensitivity data include tolerance, trial-step, line-accuracy and "
        "initial-condition changes. The supplied boundary example has a negative directional "
        "derivative at α = 1 and illustrates why the original open bracket may not yield perpendicular "
        "successive steepest-descent directions. The previous figures and benchmark CSVs remain "
        "available in the results folder."
    )
    report.heading("Verification", 2)
    report.p(
        f"Validation checked {validation['cases_checked']} saved histories, "
        f"{validation['timed_samples_checked']:,} raw timing samples and "
        f"{validation['newton_pairs_checked']} Newton pairs. Independent scalar roots agreed with "
        f"the selected closed line step to within {validation['max_closed_line_alpha_error']:.2e}. "
        f"The largest relative secant residual across accepted updates was "
        f"{validation['max_relative_secant_residual']:.2e}. The additional automated tests cover "
        "derivatives, unit steps, endpoints, operation counts, curvature rejections and legacy regression."
    )
    report.heading("References", 2)
    report.p("[1] Uncond.docx. Course assignment, first data row of both parameter tables.")
    report.p("[2] Unconditional Minimization Methods. Lecture 5, slides 18–19 and 21–28. 2026.")
    report.p(
        "[3] J. Nocedal and S. J. Wright. Numerical Optimization. 2nd ed., Springer, 2006. "
        "Chapters 3 and 6, especially Section 6.3. DOI 10.1007/978-0-387-40065-5."
    )
    report.p(
        "[4] Course lecture on second-order and quasi-Newton methods. Student transcript and "
        "subsequent mathematical reconstruction, November 2026."
    )


def build_comparison(report_type, baseline, results, output):
    data = load_json(results / "experiment.json")
    validation = load_json(results / "validation.json")
    main = read_csv(results / "summary.csv")
    adaptive = read_csv(results / "adaptive_summary.csv")
    if (not validation["passed"] or len(main) != 60 or len(adaptive) != 12
            or validation["cases_checked"] != 72
            or data["initial_points"] != [[0, 0], [0, 10], [10, 0]]):
        raise ValueError("Build the report from a validated complete comparison at the three assigned starting points.")
    with TemporaryDirectory() as directory:
        assets = Path(directory)
        report = report_type(assets)
        report.manifest = render_equations(assets, EQUATIONS | MORE_EQUATIONS)
        report_body(report, baseline, results, data, main, adaptive, validation)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.doc.save(output)
        restyle_report(output, cover_overrides={
            "document_title": TITLE,
            "report_title": "Comparative Analysis of\nOptimization Methods",
            "cover_lines": ["First Order", "Newton and", "Quasi Newton Methods"],
            "header_title": "Unconstrained Optimization Methods",
            "city_spacing": 40,
        }, separate_abstract=False)
