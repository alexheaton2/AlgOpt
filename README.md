# AlgOpt
Solving constrained optimization problems where the constraints are an algebraic set, i.e. defined by polynomials.

To do:
- Implement an `AlgebraicSet` as a subtype of `Manifold` in `Manifolds.jl`.
- Implement whatever functions are required to run the algorithms in `Manopt.jl` to minimize an objective function constrained to the `AlgebraicSet` we created.
- The above tasks may require `exp!` or `retract` functions to be defined. We could use Newton methods like those in `NLsolve.jl` to approximate the `exp!` function.
- Found out we also need to implement `log!` on `AlgebraicSet`.
- Or we could use parameter homotopy methods to approximate the `exp!` function. Try both of these.
