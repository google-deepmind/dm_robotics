# Least Squares Quadratic Optimization Library

<!--* freshness: { owner: 'josechenf' reviewed: '2020-09-03' } *-->

Least squares quadratic optimization library for Robotics.

This library contains a solver using a stack-of-tasks[^1][^2] approach for
finding solutions to a set of weighted least-squares problems with a
hierarchical structure. For every hierarchy, the solver finds a solution to the
following least squares optimization problem:

$$ \underset{x}{\mathrm{argmin}} \sum_i w_i || M_i x - b_i ||^2 $$

subject to:

$$l_j \leq C_j x \leq u_j \quad \forall j$$

where $$w_i$$, $$M_i$$, and $$b_i$$ are the weight, coefficient matrix, and bias
of the $$i$$-th task, respectively; and $$C_j$$, $$l_j$$, and $$u_j$$ are the
coefficient matrix, lower bound, and upper bound for the $$j$$-th constraint,
respectively.

The tasks of each hierarchy are projected to the remaining hierarchies as a
nullspace projection constraint of the form:

$$M_{\text{null}} x = M_{\text{null}} x_{\text{sol}}$$

where $$M_{\text{null}}$$ contains the stacked coefficient matrices of the tasks
in the previous hierarchy, and $$x_{\text{sol}}$$ is the solution to the
optimization problem of the previous hierarchy.

## References

[^1]: N. Mansard, O. Stasse, P. Evrard and A. Kheddar, "A versatile Generalized
    Inverted Kinematics implementation for collaborative working humanoid
    robots: The Stack Of Tasks," 2009 International Conference on Advanced
    Robotics, Munich, 2009, pp. 1-6.
[^2]: A. Rocchi, E. M. Hoffman, D. G. Caldwell, and N. G. Tsagarakis, “Opensot:
    a whole-body control library for the compliant humanoid robot coman,” in
    Robotics and Automation (ICRA), 2015 IEEE International Conference on.
    IEEE, 2015, pp. 1093–1099.
