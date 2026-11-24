# Practical Presentation Script

## Speaking notes

**Purpose and objectives.** I extended our gradient descent project to compare first-order, second-order and quasi-Newton methods. I kept the exponential and cosine objectives from the first parameter row of the assignment. Their Hessians have positive global lower bounds, so each objective has a unique global minimizer. The reference minima are about 0.02226823 for F1 and exactly 0.25 for F2.

**Directions and steps.** The first-order methods move in the direction of the negative gradient. I kept Step Division and the original Golden Section procedure. Newton starts from the second-order Taylor model. Differentiating the model in the displacement gives the system $H_k p_k=-g_k$. I computed the direction by forming the inverse or by solving this system, and tested each direction with unit steps and Golden Section. For the new line searches I compare both endpoints of the closed interval, which allows an exact step of one. Every method uses the same stopping threshold, a gradient norm below ten to the minus six.

**Quasi-Newton methods.** BFGS starts with an identity matrix and builds a Hessian approximation from the change in the gradient after each step. Its update makes the new matrix act on the displacement in the same way as the observed gradient change. I used the same framework with the restricted Broyden parameter set to one half. The extra correction has zero action on the displacement, so it preserves the secant equation. The implementation also checks curvature, positive definiteness and conditioning.

**Experimental results.** I ran ten configurations on both objectives from (0, 0), (0, 10) and (10, 0). Fifty-eight of the sixty primary runs converged. All four Newton configurations converged from all the tested starting points. For F1 from (10, 0), the Newton linear solve used four unit steps and five objective calls. With Golden Section it used three steps and one hundred sixty calls. On F1 from (0, 10), both unit-step quasi-Newton methods took an excessively large first step and later stopped on a nonfinite objective. With Golden Section both reached the tolerance in five steps.

**Trajectory and accuracy.** The contour charts show the initial point, accepted points, movement directions and reference minimum. I also saved the error of each iterate. On F1 from the origin, the last two quadratic error ratios for unit-step Newton are about 0.217 and 0.214. This is consistent with fast local convergence. On F2 from the origin, the Newton line is quadratic and Newton finishes in one step. There are too few points to estimate an order in that case.

**Comparison.** The explicit-inverse and direct-solve Newton paths agree closely. The direct solve avoids forming a full inverse. BFGS and the Broyden variant evaluate zero analytical Hessians, while the fixed steps can overshoot. I also compared an adaptive parameter with the constant value. It helped in one unit-step case and hurt in another. I would report both results and keep the constant value as the main experimental variant.

## What to show on screen

1. Run the first demonstration and read the $X_k$ sequence aloud for a few steps.
2. Open `comparison_results/figures/paths_f1_s2.png`. Follow the arrow that leaves the window in the unit-step quasi-Newton panel.
3. Open `comparison_results/figures/convergence_f1_s1.png` and explain what the stopping line marks.
4. Open `comparison_results/summary.csv` and point to both failure rows and the function, gradient and Hessian counters.

```cmd
.venv\Scripts\python.exe run_comparison.py --function f1 --method newton_solve_fixed --x0 0 10 --show-iterates --no-plots --benchmark-repeats 3 --output "%TEMP%\gd-newton-demo"
.venv\Scripts\python.exe run_comparison.py --function f1 --method bfgs_golden --x0 0 10 --show-iterates --no-plots --benchmark-repeats 3 --output "%TEMP%\gd-bfgs-demo"
start "" "comparison_results\figures\paths_f1_s2.png"
start "" "comparison_results\figures\convergence_f1_s1.png"
start "" "comparison_results\summary.csv"
```

## Questions the professor may ask

**1. What is the difference between first-order and second-order methods?**

Gradient descent uses the gradient to choose a direction. Newton also uses the Hessian to account for the local curvature of the objective.

**2. How did you derive the Newton direction?**

I differentiated the second-order Taylor model with respect to the displacement. Setting that derivative to zero gives $g_k+H_kp_k=0$.

**3. Why does solving a system give the same direction?**

If the Hessian is invertible, multiplying $H_kp_k=-g_k$ by its inverse gives $p_k=-H_k^{-1}g_k$. Both calculations therefore target the same vector.

**4. Why use a direct solve in practice?**

The solve computes one vector for one right-hand side. Forming the inverse computes the whole matrix first. The solve avoids that extra work and the following matrix-vector multiplication.

**5. What is the difference between Newton and quasi-Newton?**

Newton evaluates the analytical Hessian at each update. Quasi-Newton updates an approximation using the differences between successive points and gradients.

**6. What is the secant condition?**

It says $B_{k+1}s_k=y_k$. The new approximate Hessian applied to the displacement gives the observed change in gradient.

**7. How does BFGS approximate the Hessian?**

It starts with the identity and adds two rank-one corrections. The first removes the previous curvature term along the step. The second inserts the observed curvature. Positive curvature preserves positive definiteness in exact arithmetic.

**8. Why choose a nonzero Broyden parameter?**

One half gives a reproducible intermediate member of the restricted Broyden class. The correction vanishes along the secant displacement and the update preserves positive definiteness under positive curvature. The choice lets us test what changes relative to BFGS.

**9. How did you determine the step size?**

For the new methods I either use one at every update or minimize along the direction by Golden Section on a closed interval from ten to the minus eight to one. I evaluate the two endpoints as candidates.

**10. What do linear, superlinear and quadratic convergence mean?**

Linear means the error eventually shrinks by a factor bounded below one. For superlinear convergence the consecutive error ratio tends to zero. Quadratic convergence bounds the next error by a constant times the square of the current error.

**11. Which method performed best?**

All four Newton configurations converged on all six cases. The two unit-step Newton configurations were the fastest by the median solver times in this two-dimensional experiment. Newton with Golden Section saved one iteration on F1 from (10, 0), while spending 160 objective calls compared with five.

**12. How did you validate the results?**

I checked derivatives by finite differences, used independent stationarity solutions for the references, compared the two Newton directions and tested both endpoints of the line search. The export validator also recalculates the gradients, line minima and evaluation counts.

**13. Why did some methods require more iterations?**

The direction and step length control how far an iterate moves toward the minimizer. Gradient descent can zigzag when curvature is different across directions. Newton accounts for the local curvature directly. A quasi-Newton method has to build that information from its steps.

**14. How can you reuse the framework?**

I can supply a new objective value and gradient, give the solver a starting vector, and select a method and settings. Newton additionally needs the Hessian. The histories and work counters are returned through the same interface.
