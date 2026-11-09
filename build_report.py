import argparse
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from report_equations import render_equations
from polytech_theme import restyle_report


ROOT = Path(__file__).resolve().parent


def read_csv(file):
    with file.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_method(name):
    return "SD" if name == "Step Division" else "GSS"


def label(row):
    return row["function"].upper()+" "+short_method(row["method"])


def number(value, digits=3):
    return f"{float(value):.{digits}e}"


def math_run(text):
    run = OxmlElement("m:r")
    node = OxmlElement("m:t")
    node.text = text
    run.append(node)
    return run


def math_subscript(base, subscript):
    node = OxmlElement("m:sSub")
    for tag, text in [("m:e", base), ("m:sub", subscript)]:
        child = OxmlElement(tag)
        child.append(math_run(text))
        node.append(child)
    return node


def math_matrix(rows):
    delimiter = OxmlElement("m:d")
    properties = OxmlElement("m:dPr")
    for tag, value in [("m:begChr", "("), ("m:endChr", ")")]:
        child = OxmlElement(tag)
        child.set(qn("m:val"), value)
        properties.append(child)
    delimiter.append(properties)
    expression = OxmlElement("m:e")
    matrix = OxmlElement("m:m")
    for row in rows:
        mr = OxmlElement("m:mr")
        for text in row:
            cell = OxmlElement("m:e")
            cell.append(math_run(text))
            mr.append(cell)
        matrix.append(mr)
    expression.append(matrix)
    delimiter.append(expression)
    return delimiter


class Report:
    def __init__(self, equations):
        self.doc = Document()
        for node in self.doc.styles.element.xpath(".//w:pBdr"):
            node.getparent().remove(node)
        self.equations = equations
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.top_margin, section.bottom_margin = Inches(0.7), Inches(0.7)
        section.left_margin, section.right_margin = Inches(0.85), Inches(0.85)
        section.header_distance, section.footer_distance = Inches(0.3), Inches(0.3)
        for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "Caption"]:
            style = self.doc.styles[name]
            style.font.name = "Calibri"
            style.font.color.rgb = RGBColor(0, 0, 0)
        normal = self.doc.styles["Normal"]
        normal.font.size = Pt(11)
        normal.paragraph_format.space_after = Pt(7)
        normal.paragraph_format.line_spacing = 1.1
        for name, size in [("Title", 25), ("Heading 1", 17), ("Heading 2", 12), ("Caption", 9)]:
            self.doc.styles[name].font.size = Pt(size)
        self.doc.styles["Heading 1"].paragraph_format.space_after = Pt(10)
        self.doc.styles["Heading 2"].paragraph_format.space_before = Pt(9)
        self.doc.styles["Caption"].paragraph_format.space_after = Pt(10)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
        self.doc.core_properties.title = "Gradient Descent for Two Unconstrained Objectives"
        self.doc.core_properties.subject = "Methods of Optimization and Decision Making Practical Group 1"
        self.doc.core_properties.author = "Group 1"

    def p(self, text, style=None):
        return self.doc.add_paragraph(text, style=style)

    def heading(self, text, level=1, page=False):
        if page:
            self.doc.add_page_break()
        return self.doc.add_heading(text, level)

    def eq(self, name):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(8)
        if name in ["hess1", "hess2", "bound2"]:
            equation = OxmlElement("m:oMath")
            if name == "hess1":
                equation.append(math_subscript("H", "1"))
                equation.append(math_run(" = "))
                equation.append(math_matrix([["2+t", "t"], ["t", "4+t"]]))
                equation.append(math_run(",  "))
                sub = OxmlElement("m:sSup")
                for tag, text in [("m:e", "e"), ("m:sup", "−x−y")]:
                    node = OxmlElement(tag)
                    node.append(math_run(text))
                    sub.append(node)
                equation.append(math_run("t = "))
                equation.append(sub)
                equation.append(math_run(" > 0"))
            elif name == "hess2":
                equation.append(math_subscript("H", "2"))
                equation.append(math_run(" = "))
                equation.append(math_matrix([["2−q", "−q"], ["−q", "4−q"]]))
                equation.append(math_run(",  q = cos(x+y)"))
            else:
                equation.append(math_matrix([["1", "−1"], ["−1", "3"]]))
                equation.append(math_run(" ≼ "))
                equation.append(math_subscript("H", "2"))
                equation.append(math_run(" ≼ "))
                equation.append(math_matrix([["3", "1"], ["1", "5"]]))
            p._p.append(equation)
        else:
            p.add_run().add_picture(str(self.equations/name)+".png", width=Inches(self.manifest[name]["width_inches"]))

    def figure(self, file, caption, width=6.7):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(3)
        p.add_run().add_picture(str(file), width=Inches(width))
        p = self.p(caption, "Caption")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def table(self, headers, rows, widths, font=9.5):
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for cell, text in zip(table.rows[0].cells, headers):
            cell.text = str(text)
        repeat = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            cells = table.add_row().cells
            for cell, text in zip(cells, row):
                cell.text = str(text)
        for i, row in enumerate(table.rows):
            no_split = OxmlElement("w:cantSplit")
            row._tr.get_or_add_trPr().append(no_split)
            for j, cell in enumerate(row.cells):
                cell.width = Inches(widths[j])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                properties = cell._tc.get_or_add_tcPr()
                margins = OxmlElement("w:tcMar")
                for side, size in [("top", 70), ("bottom", 70), ("left", 80), ("right", 80)]:
                    node = OxmlElement("w:"+side)
                    node.set(qn("w:w"), str(size))
                    node.set(qn("w:type"), "dxa")
                    margins.append(node)
                properties.append(margins)
                borders = OxmlElement("w:tcBorders")
                for side in ["top", "left", "bottom", "right"]:
                    node = OxmlElement("w:"+side)
                    node.set(qn("w:val"), "single")
                    node.set(qn("w:sz"), "4")
                    node.set(qn("w:color"), "D9D9D9")
                    borders.append(node)
                properties.append(borders)
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "E8EEF3" if i == 0 else ("F7F9FA" if i % 2 == 0 else "FFFFFF"))
                properties.append(shading)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.05
                    for run in p.runs:
                        run.font.size = Pt(font)
                        run.font.bold = i == 0
        self.p("").paragraph_format.space_after = Pt(1)
        return table


def main():
    parser = argparse.ArgumentParser(description="Generate the English report from saved results")
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    parser.add_argument("--output", type=Path, default=ROOT/"report"/"Gradient_Descent_Group1_Report.docx")
    args = parser.parse_args()
    result_dir = args.results.resolve()
    metadata = json.loads((result_dir/"experiment.json").read_text(encoding="utf-8"))
    summary = metadata["summary"]
    if len(summary) != 4 or any(row["status"] != "converged" for row in summary):
        raise ValueError("The full report requires four converged runs from the full experiment command.")
    settings, refs = metadata["settings"], metadata["references"]
    validation = json.loads((result_dir/"validation.json").read_text(encoding="utf-8"))
    if not validation["passed"]:
        raise ValueError("Run and pass validate.py before creating the report.")
    csv_summary = read_csv(result_dir/"summary.csv")
    for row, saved in zip(summary, csv_summary):
        if row["function"] != saved["function"] or row["iterations"] != int(saved["iterations"]):
            raise ValueError("Summary files disagree. Rerun the full experiment command.")
    figures = result_dir/"figures"
    assets = ROOT/"report"/"assets"/"equations"
    report = Report(assets)
    report.manifest = render_equations(assets)
    p = report.p("PETER THE GREAT ST PETERSBURG\nPOLYTECHNIC UNIVERSITY")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(35)
    p = report.p("Gradient Descent for Two\nUnconstrained Objectives", "Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(20)
    p = report.p("Methods of Optimization and Decision Making\nPractical work\nGroup 1", "Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(30)
    report.heading("Objective", 2)
    report.p("We implemented two first order gradient descent methods for the first parameter row of both assignment functions. The study compares numerical accuracy and computational cost and explains the geometry of the generated paths. Both methods reached the gradient norm stopping condition for both functions.")
    a, b, c, d = summary
    report.p(f"For the exponential function, step division required {a['iterations']} accepted updates and the golden section version required {b['iterations']}. For the cosine function, they required {c['iterations']} and {d['iterations']} updates. The more accurate line minimization did not give a universal improvement in iteration count or execution cost.")
    report.p("The package contains the complete iteration histories, independent checks, comparison experiments and the scientific figures used in this report.")
    p = report.p("Saint Petersburg\n2026")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    report.heading("1 Problem Formulation", page=True)
    report.p("The assignment defines two functions with six coefficients [1]. Group 1 uses the first data row of both tables. We apply both methods to both resulting functions, giving four main experiments.")
    report.table(["Parameter", "a", "b", "c", "d", "k", "m"], [["First data row", 1, 2, 1, 1, -1, 2]], [1.6, .85, .85, .85, .85, .85, .85])
    report.eq("f1")
    report.eq("f2")
    report.heading("Analytical derivatives", 2)
    report.eq("grad1")
    report.eq("hess1")
    report.eq("grad2")
    report.eq("hess2")
    report.p("The gradient determines the search direction. The Hessian is used only for mathematical analysis and independent checks. Neither descent method uses second derivatives.")

    report.heading("2 Mathematical Background", page=True)
    report.heading("Convexity and existence of a minimum", 2)
    report.eq("bound1")
    report.p("The first Hessian is positive definite everywhere. Its lower eigenvalue bound is 2. For the second function, the cosine factor belongs to the interval from minus one to one, which gives the matrix bounds below.")
    report.eq("bound2")
    report.eq("constants")
    report.p("The lower matrix has eigenvalues 2 minus square root of 2 and 2 plus square root of 2. The upper matrix has largest eigenvalue 4 plus square root of 2. Both objectives are therefore strongly convex. Completing the squares gives the following lower bounds.")
    report.eq("coercive")
    report.p("The bounds tend to infinity as the point tends to infinity. Thus each continuous objective attains a minimum, and strong convexity makes that minimum unique and global.")
    report.heading("Convergence conditions", 2)
    report.p("The cosine objective has a globally Lipschitz gradient with constant 4 plus square root of 2. The exponential objective does not have a global upper Hessian bound because its exponential term is unbounded. Its initial sublevel set is compact, however, and the Hessian is bounded there. A compact neighbourhood of that set supplies the local Lipschitz bound needed for the step division argument in lecture slides 21 and 22 [2].")
    report.p("With sufficient decrease and a valid reduction factor, standard gradient descent convergence applies on these level sets. The ideal line minimization also yields descent. With an approximate line search, finite inner accuracy and rounding limit how closely the ideal properties can be reproduced. We use the gradient norm to decide actual numerical convergence.")
    report.eq("error")
    report.p("These strong convexity bounds also connect the final gradient residual to point error and objective error. They provide an independent quantitative check on the computed solutions.")

    report.heading("Independent Reference Solutions", page=True)
    report.heading("Exponential objective", 2)
    report.p("Setting the two partial derivatives to zero and introducing the positive scalar t reduces the stationary equations to one variable.")
    report.eq("ref1")
    report.eq("scalar1")
    report.p("The scalar equation is strictly increasing for positive t. We solved it using SciPy Brent root finding on the bracket from 0.1 to 1 and checked it with the principal Lambert W value. This calculation does not use either gradient descent routine.")
    report.heading("Cosine objective", 2)
    report.eq("ref2")
    report.p("The derivative of s minus three quarters of sin s is at least one quarter, so s equals zero is the only root. The resulting minimum value is exactly 0.25. We also solved each two variable stationarity system independently with SciPy root finding from the origin.")
    report.table(["Function", "Reference x", "Reference y", "Reference minimum"],
                 [[key.upper(), f"{refs[key]['point'][0]:.12f}", f"{refs[key]['point'][1]:.12f}", f"{refs[key]['value']:.12f}"] for key in ["f1", "f2"]],
                 [1, 1.8, 1.8, 2.1])
    report.p("The uniqueness proofs establish global optimality. The independent numerical procedures locate and verify the stationary points. Their residuals and mutual differences are recorded in validation.json.")

    report.heading("3 Gradient Descent Methods", page=True)
    report.eq("update")
    report.eq("stop")
    report.heading("Step division", 2)
    report.p("At each outer iteration we compute the gradient and reset the trial step to alpha0. We accept the first trial satisfying the lecture's sufficient decrease condition [2, slide 19].")
    report.eq("armijo")
    report.p("If the condition fails, we multiply the trial step by lambda and test again. The accepted point becomes the next iterate. This is an Armijo backtracking rule with the lecture's epsilon coefficient. We keep epsilon distinct from the gradient tolerance delta.")
    report.heading("One dimensional minimization", 2)
    report.eq("golden")
    report.p("The interval is fixed by lecture slide 25. Golden section search begins with bracket endpoints zero and one and evaluates strictly interior points. It does not expand the bracket. The two interior positions use the following ratio.")
    report.eq("ratio")
    report.p("Each contraction discards one side, reuses one previous value and evaluates one new interior point. The inner search stops when the absolute bracket width is below its own tolerance. We return the better of the two retained interior points. Strong convexity makes every nonstationary line function strictly convex.")
    report.heading("Interpretation of the lecture slides", 2)
    report.p("Slide 18 prints the sufficient decrease inequality next to the step reduction instruction without a clear branch. We reduce only when the acceptance condition fails, consistent with slide 19. Slide 24 combines a line minimization instruction with the preceding backtracking template. For Method 2 we use the complete prose algorithm on slide 25, which selects the line minimum without an extra backtracking stage. This choice preserves the distinction between the two requested methods.")

    report.heading("4 Implementation Details", page=True)
    report.p("The objective module supplies analytical derivatives. The line search module implements Golden Section. The solver module performs updates, counts work and saves the trace. Separate modules generate figures and calculate reference solutions. NumPy and Matplotlib support the implementation. SciPy is restricted to validation and references.")
    report.table(["Setting", "Main experiment value"], [
        ["Initial point", str(tuple(settings["initial_point"]))],
        ["Initial trial step alpha0", settings["initial_step"]],
        ["Reduction factor lambda", settings["reduction_factor"]],
        ["Sufficient decrease epsilon", settings["armijo_epsilon"]],
        ["Gradient tolerance delta", number(settings["tolerance"])],
        ["Golden Section interval", "0 < alpha < 1"],
        ["Golden Section width tolerance", number(settings["line_tolerance"])],
        ["Maximum outer updates", settings["max_iterations"]],
    ], [3.7, 3.0], font=10)
    report.p("The assignment and lecture do not prescribe a numerical starting point or these numerical constants. We selected one shared initial point for a controlled comparison. alpha0, lambda and epsilon affect Method 1. They do not enter the Method 2 line minimization.")
    report.heading("Iteration records and evaluation counts", 2)
    report.p("Row zero stores the initial point. Each accepted update adds a row containing coordinates, objective value, gradient components and norm, incoming step, actual step length, line search work and cumulative counters. Thus a run with k updates stores k plus one points. Dot products, angles and cosines between successive gradients are also saved.")
    report.p("A function evaluation counts one scalar objective value or one algebraically equivalent objective increment. All rejected trial steps and every new Golden Section sample count. Reused cached values do not count again. The initial value and a final direct objective verification also count. Each initial or accepted point requires one gradient evaluation. Plot grids, references and other runs are outside each run's counters.")
    report.p("Timing uses perf_counter inside the solver, including history and counter bookkeeping. It excludes imports, plotting, disk output and references. We report the median of repeated executions after one warm-up and preserve every timing sample. These are CPU measurements from the recorded environment and are dependent on the computer.")

    report.heading("Stable Evaluation and Numerical Checks", page=True)
    report.p("Near a minimum, subtracting two almost equal objective values can hide a small decrease. We evaluate the same objective along a line through an algebraically equivalent increment. Let h be the displacement, r the sum of its components, g the current analytical gradient and s the sum of the current coordinates.")
    report.eq("change1")
    report.eq("change2")
    report.p("The exponential remainder and r minus sin r are evaluated by short Taylor series for small arguments. This avoids cancellation in the nonlinear remainders. The calculation reuses the current gradient and does not evaluate another gradient. Every scalar increment evaluation counts once as an objective oracle call. The final objective value is evaluated directly, so the reported final value does not depend on accumulated increments.")
    report.heading("Validation evidence", 2)
    tests = validation["checks"]
    report.table(["Independent check", "Observed result"], [
        ["Gradient against central differences", number(tests[0]["max_gradient_relative_error"])],
        ["Hessian against differentiated gradient", number(tests[0]["max_hessian_relative_error"])],
        ["Stable increment against direct subtraction", number(tests[0]["max_increment_relative_error"])],
        ["Golden step against line derivative root", number(tests[3]["max_alpha_difference_from_brent"])],
        ["Main runs and mathematical error bounds", "All four passed"],
        ["Saved sensitivity run statuses", str(tests[5]["runs_checked"])+" passed"],
    ], [4.3, 2.4], font=10)
    report.p("The first three entries are maximum relative discrepancies across the sampled points. The line step entry is the maximum absolute alpha difference over interior main Golden Section steps. A separate quadratic problem verifies the counters with a known number of calls. We also check every main update, every accepted sufficient decrease condition and the interior or boundary status of every main line minimum.")

    report.heading("5 Numerical Experiments", page=True)
    report.p("SD denotes Step Division. GSS denotes gradient descent with Golden Section line minimization. All four main runs use the same initial point and gradient tolerance. Every run reached the strict gradient norm stopping condition.")
    report.table(["Run", "Final x", "Final y", "Final objective", "Gradient norm", "Point error"], [
        [label(row), f"{row['final_x']:.9f}", f"{row['final_y']:.9f}", f"{row['minimum_value']:.12f}", number(row["gradient_norm"]), number(row["point_error"])] for row in summary
    ], [.8, 1.15, 1.15, 1.4, 1.1, 1.1], font=9)
    report.table(["Run", "Updates", "Function calls", "Gradient calls", "Median ms", "Status"], [
        [label(row), row["iterations"], row["function_evaluations"], row["gradient_evaluations"], f"{row['median_time_ms']:.3f}", row["status"]] for row in summary
    ], [.8, .75, 1.1, 1.1, 1.1, 1.85], font=9.5)
    report.p(f"Each median uses {summary[0]['timing_repeats']} executions after a warm-up. The single-run times, timing ranges and raw samples are included in the CSV outputs. Small timing differences should be interpreted with the recorded environment and Python overhead in mind.")
    report.p(f"For F1, GSS used {b['iterations']} updates compared with {a['iterations']} for SD, but required {b['function_evaluations']} function calls compared with {a['function_evaluations']}. It reduced gradient calls from {a['gradient_evaluations']} to {b['gradient_evaluations']}. This can matter when gradients are expensive, although these two-dimensional gradients are inexpensive.")
    report.p(f"For F2, SD required {c['iterations']} updates and {c['function_evaluations']} function calls, while GSS required {d['iterations']} and {d['function_evaluations']}. More accurate line minimization improved the immediate line objective but did not produce fewer outer iterations for this starting point and these settings.")
    report.p("The final objective errors are consistent with the gradient residuals and the strong convexity bounds. Objective values alone can look identical after rounding, even when coordinate errors and gradient norms differ.")

    report.heading("6 Geometrical Interpretation", page=True)
    report.heading("Descent and level curves", 2)
    report.eq("descent")
    report.p("For a differentiable objective, this negative directional derivative guarantees a decrease for a sufficiently small positive step. Among directions of unit Euclidean length, the normalized negative gradient gives the steepest local decrease.")
    report.eq("normal")
    report.p("A level curve keeps the objective constant. Its tangent therefore has zero inner product with the gradient. The gradient is normal to the curve at regular points. This property is distinct from orthogonality between two successive search directions.")
    report.figure(figures/"level_curve_geometry.png", "Figure 1. The gradient is normal to the level curve at the marked regular point. The negative gradient points towards lower values. Arrow lengths are scaled for display.")
    report.p("All planar path figures use equal horizontal and vertical scales, preserving angles. Markers show every saved iterate, although points close to the solution overlap at the full path scale. Their complete coordinates remain available in the history CSV files.")

    for key, title, figure_number in [("f1", "Exponential Function Paths", 2), ("f2", "Cosine Function Paths", 3)]:
        report.heading(title, page=True)
        report.figure(figures/f"paths_{key}.png", f"Figure {figure_number}. All iteration points for {key.upper()}. The square marks the common initial point, the circle marks the final iterate and the star marks the independent reference solution. Arrowheads show the search direction.")
        selected = []
        for method, method_label in [("division", "SD"), ("golden", "GSS")]:
            history = read_csv(result_dir/"histories"/f"{key}_{method}.csv")
            for row in history[:3]:
                selected.append([method_label, row["iteration"], f"{float(row['x']):.7f}", f"{float(row['y']):.7f}",
                                 f"{float(row['alpha_from_previous']):.7f}", f"{float(row['value']):.8f}"])
        report.table(["Method", "k", "x", "y", "Incoming alpha", "Objective"], selected, [.7, .4, 1.2, 1.2, 1.45, 1.75], font=9.5)
        report.p("The table lists the initial point and first two updates as examples. The initial row has zero incoming step. The full history includes all later points and both gradient components.")
        if key == "f1":
            report.p("Step division initially overshoots across the valley and then corrects with smaller accepted steps. The first line minimization puts the iterate near the bottom of its search line, after which the direction changes towards the common minimum.")
        else:
            report.p("The contours become elongated around the minimum. The two rules follow different paths through this valley. Successive exact line minima give perpendicular directions, but this geometric property alone does not guarantee fewer iterations than sufficient decrease backtracking.")

    for key, fig_num in [("f1", 4), ("f2", 5)]:
        report.heading("7 Convergence" if key == "f1" else "Convergence of the Cosine Function", page=True)
        report.figure(figures/f"convergence_{key}.png", f"Figure {fig_num}. Objective value, gradient norm, distance to the independent minimum and absolute objective error for {key.upper()}. The dashed line is the stopping threshold. The shaded band indicates the scale where objective rounding becomes relevant.")
        if key == "f1":
            report.p("Both sequences reduce the objective. The line minimization run reaches its terminal gradient residual in fewer outer iterations. The objective gap approaches floating point resolution, so the gradient norm and coordinate error provide a more informative final accuracy check.")
        else:
            report.p("The gradient norm need not decrease at every accepted step, although the objective decreases. The line minimization sequence has an approximately linear trend on the logarithmic residual plot near the solution. The sufficient decrease sequence reaches the selected stopping threshold earlier in this experiment.")
        report.p("The plots show all saved outer iterates. They do not count each trial inside a line search as a separate outer iteration. Computation cost must therefore also be compared using function and gradient evaluation counts.")

    report.heading("8 Orthogonality of Successive Directions", page=True)
    report.p("At an exact interior minimum of the one dimensional line problem, its derivative vanishes. Since each search direction equals the negative gradient, the following relation holds [2, slides 27 and 28].")
    report.eq("orthogonal")
    report.eq("cosine")
    report.figure(figures/"orthogonality.png", "Figure 6. Absolute cosine of the angle between successive gradients. Zero represents orthogonality. Golden Section gives values close to zero in the main runs. Step division does not impose this condition.")
    report.p(f"The maximum absolute cosine in the main GSS runs was {b['max_abs_successive_gradient_cosine']:.3e} for F1 and {d['max_abs_successive_gradient_cosine']:.3e} for F2. The residual includes inner search error and floating point effects. When the next gradient is very small, its normalized angle is especially sensitive to rounding.")
    boundary = json.loads((result_dir/"boundary_example.json").read_text(encoding="utf-8"))
    boundary_check = tests[4]
    report.p(f"The interval restriction is essential. A separate F2 run started at ({boundary['initial_x']:.8f}, {boundary['initial_y']:.8f}). Its first step was {boundary_check['first_alpha']:.12f} and its first successive-gradient cosine was {boundary_check['first_cosine']:.8f}. The line derivative at alpha equal to one was {boundary_check['derivative_at_alpha_one']:.3e}, so the unrestricted minimum lay beyond the lecture interval. On the open interval, this step approximates a boundary infimum. Orthogonality is not required there.")

    report.heading("9 Sensitivity Experiments", page=True)
    report.heading("Gradient tolerance", 2)
    report.p("We repeated each function and method for three stopping tolerances, keeping the other settings fixed. Every saved run reached its selected gradient threshold.")
    tol_rows = read_csv(result_dir/"tolerance_sweep.csv")
    report.table(["Run", "Delta", "Updates", "Function calls", "Gradient norm"], [
        [label(r), number(r["tolerance"], 0), r["iterations"], r["function_evaluations"], number(r["gradient_norm"])] for r in tol_rows
    ], [1.2, 1.1, .9, 1.3, 2.2], font=9.5)
    report.heading("Initial trial step for step division", 2)
    step_rows = read_csv(result_dir/"step_sweep.csv")
    report.table(["Function", "Alpha0", "Updates", "Function calls"], [
        [r["function"].upper(), r["initial_step"], r["iterations"], r["function_evaluations"]] for r in step_rows
    ], [1.2, 1.2, 1.3, 3.0], font=9.5)
    report.p("A smaller initial trial step can avoid rejected trials while requiring more outer iterations. A larger trial can decrease iteration count yet increase function evaluations through reductions. These results depend on the objective and the starting point.")

    report.heading("Line Accuracy and Starting Point", page=True)
    report.heading("Golden Section bracket tolerance", 2)
    line_rows = read_csv(result_dir/"line_accuracy_sweep.csv")
    report.table(["Function", "Line tolerance", "Updates", "Function calls", "Max absolute cosine"], [
        [r["function"].upper(), number(r["line_tolerance"], 0), r["iterations"], r["function_evaluations"], number(r["max_abs_successive_gradient_cosine"])] for r in line_rows
    ], [.8, 1.25, .8, 1.25, 2.6], font=9.5)
    report.p("A tighter line tolerance substantially increases scalar objective calls. In these experiments it does not change the outer iteration count. The orthogonality residual generally improves, but eventually reaches a rounding floor and need not improve monotonically with a smaller bracket width. Bracket width and step error are different quantities in floating point arithmetic.")
    report.heading("Common alternative starting points", 2)
    starts = read_csv(result_dir/"initial_point_sweep.csv")
    pairs = {}
    for r in starts:
        k = (r["function"], r["initial_x"], r["initial_y"])
        pairs.setdefault(k, {})[short_method(r["method"])] = r
    rows = []
    for (key, x, y), pair in pairs.items():
        rows.append([key.upper(), f"({float(x):g}, {float(y):g})", pair["SD"]["iterations"], pair["GSS"]["iterations"], pair["GSS"]["near_upper_boundary_steps"]])
    report.table(["Function", "Initial point", "SD updates", "GSS updates", "GSS near boundary steps"], rows,
                 [.8, 1.3, 1.2, 1.2, 2.2], font=9.5)
    report.p("Every starting point converged. A near boundary step means alpha exceeded one minus 1e-7. Some alternative F2 starts generate such steps, even though all main line minima are interior. The ranking of methods by outer iteration count also changes with the initial point.")

    report.heading("10 Discussion and Conclusion", page=True)
    report.figure(figures/"evaluation_cost.png", "Figure 7. Gradient residual against cumulative function evaluations. The Golden Section method spends many scalar evaluations inside each outer iteration.")
    report.p("The methods reach the same unique minima through different step rules. For the exponential objective, line minimization reduces gradient calls but increases function calls and measured execution time. For the cosine objective and the selected main settings, step division uses fewer updates and fewer evaluations of both types. These small examples support a comparison under stated settings, rather than a universal ranking.")
    report.p("The geometric claims require separate conditions. The gradient is normal to a regular level curve for any differentiable objective. Successive gradient directions become orthogonal only after an exact interior line minimum. Approximate searches and boundary infima need separate treatment, which the main and additional experiments demonstrate.")
    report.p("Both objectives are strongly convex, so the computed stationary points are their unique global minima. Independent references, derivative checks, error bounds and complete iteration records support the results. Step division offers low scalar evaluation cost here. More accurate line minimization gives a useful geometric interpretation but can spend work beyond what is needed for the outer stopping tolerance.")
    report.heading("Reproduction", 2)
    report.p("Create the Python environment and install requirements.txt. Then run run_experiments.py, validate.py and build_report.py in that order. The README gives the full Windows CMD commands. The recorded execution used Python "+metadata["environment"]["python"].split()[0]+", NumPy "+metadata["environment"]["numpy"]+", Matplotlib "+metadata["environment"]["matplotlib"]+" and SciPy "+metadata["environment"]["scipy"]+". The package runs on a CPU without a GPU dependency.")
    report.heading("References", 2)
    report.p("[1] Uncond.docx. Course assignment supplied by the instructor. Questions 1 and 2, first data row of both parameter tables.")
    report.p("[2] Lecture_5_Uncond_1_2026.pptx. Unconditional Minimization Methods, Lecture 5. Supplied course slides. Slides 18 and 19, 21 and 22, and 24 through 28.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.doc.save(args.output)
    restyle_report(args.output)
    presentation = (
        "I implemented two first order gradient descent methods for both functions in the first row of our assignment. "
        "Both methods use the negative analytical gradient. The first method reduces the step until the sufficient decrease condition holds. "
        "The second method uses my own Golden Section search on the lecture interval between zero and one. "
        "I saved every iterate and counted every function and gradient evaluation. "
        f"For the exponential function, the methods needed {a['iterations']} and {b['iterations']} updates. "
        f"For the cosine function, they needed {c['iterations']} and {d['iterations']}. "
        "Both functions are strongly convex and have a unique global minimum. I checked the results against independent reference solutions. "
        "The contour figures show the paths and search directions. The gradient is perpendicular to a level curve. "
        "An exact interior line minimum also makes successive gradient directions perpendicular. With approximate or boundary line searches, that equality need not hold exactly. "
        "The results show why iteration counts must be compared together with evaluation counts and final accuracy.\n\n"
        "SHOW: paths_f1.png, paths_f2.png, level_curve_geometry.png, orthogonality.png, evaluation_cost.png, then results/summary.csv.\n"
    )
    (ROOT/"presentation.txt").write_text(presentation, encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
