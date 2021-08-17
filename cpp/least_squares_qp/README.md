# Least Squares Quadratic Optimization Library

Least squares quadratic optimization library for Robotics.

This library contains a solver using a stack-of-tasks<sup>1,2</sup> approach for
finding solutions to a set of weighted least-squares problems with a
hierarchical structure. For every hierarchy, the solver finds a solution to the
following least squares optimization problem:

<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{x}{\mathrm{argmin}}&space;\sum_i&space;w_i&space;||&space;M_i&space;x&space;-&space;b_i&space;||^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\underset{x}{\mathrm{argmin}}&space;\sum_i&space;w_i&space;||&space;M_i&space;x&space;-&space;b_i&space;||^2" title="\underset{x}{\mathrm{argmin}} \sum_i w_i || M_i x - b_i ||^2" /></a>

subject to:

<a href="https://www.codecogs.com/eqnedit.php?latex=l_j&space;\leq&space;C_j&space;x&space;\leq&space;u_j&space;\quad&space;\forall&space;j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?l_j&space;\leq&space;C_j&space;x&space;\leq&space;u_j&space;\quad&space;\forall&space;j" title="l_j \leq C_j x \leq u_j \quad \forall j" /></a>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;w_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;w_i" title="w_i" /></a>,
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;M_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;M_i" title="M_i" /></a>,
and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;b_i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;b_i" title="b_i" /></a>
are the weight, coefficient matrix, and bias of the
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;i" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;i" title="i" /></a>-th
task, respectively; and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;C_j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;C_j" title="C_j" /></a>,
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;l_j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;l_j" title="l_j" /></a>,
and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u_j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;u_j" title="u_j" /></a>
are the coefficient matrix, lower bound, and upper bound for the
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;j" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;j" title="j" /></a>-th
constraint, respectively.

The tasks of each hierarchy are projected to the remaining hierarchies as a
nullspace projection constraint of the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=M_{\text{null}}&space;x&space;=&space;M_{\text{null}}&space;x_{\text{sol}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?M_{\text{null}}&space;x&space;=&space;M_{\text{null}}&space;x_{\text{sol}}" title="M_{\text{null}} x = M_{\text{null}} x_{\text{sol}}" /></a>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;M_{\text{null}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;M_{\text{null}}" title="M_{\text{null}}" /></a>
contains the stacked coefficient matrices of the tasks in the previous
hierarchy, and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_{\text{sol}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;x_{\text{sol}}" title="x_{\text{sol}}" /></a>
is the solution to the optimization problem of the previous hierarchy.

## References

1.  N. Mansard, O. Stasse, P. Evrard and A. Kheddar, "A versatile Generalized
    Inverted Kinematics implementation for collaborative working humanoid
    robots: The Stack Of Tasks," 2009 International Conference on Advanced
    Robotics, Munich, 2009, pp. 1-6.
2.  A. Rocchi, E. M. Hoffman, D. G. Caldwell, and N. G. Tsagarakis, “Opensot: a
    whole-body control library for the compliant humanoid robot coman,” in
    Robotics and Automation (ICRA), 2015 IEEE International Conference on. IEEE,
    2015, pp. 1093–1099.
