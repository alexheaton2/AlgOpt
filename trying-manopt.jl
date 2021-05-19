using ManifoldsBase, LinearAlgebra, Test
import ManifoldsBase: check_manifold_point, check_tangent_vector, manifold_dimension, exp!

using LinearAlgebra, ForwardDiff

"""
    AlgebraicSet <: Manifold{ℝ}

Define an algebraic set. Construct by `AlgebraicSet(gs,d,N,tol)`
where `gs` is a list of polynomial functions whose zeros define the algebraic set
and `dim` is a nonnegative integer, the dimension of the algebraic set.
Then `N` is the ambient dimension, which is the number of variables in the polynomials.
Finally `tol` is the tolerance for checking if a point is in the algebraic set. If we
evaluate the polynomial functions at a point, and the norm of the resulting vector of nearly zero
entries is less than `tol`, then we judge the point to be on the algebraic set.
"""

struct AlgebraicSet <: Manifold{ManifoldsBase.ℝ}
    eqns
    varietydim::Int
    ambientdim::Int
    numeqns::Int
    residualtol::Float64
    f
    df
    f!
end

function AlgebraicSet(eqns,d::Int,N::Int,tol::Float64)
    k = length(eqns)
    if k==1
        f = x -> eqns[1](x)
        df = x -> ForwardDiff.gradient(eqns[1], x)
    else
        f = x -> [eqn(x) for eqn in eqns]
        df = x -> ForwardDiff.jacobian(f, x)
    end
    f! = (F,x) -> begin
        for i in 1:k
            F[i] = eqns[i](x)
        end
        for i in (k+1):N
            F[i] = 0.
        end
        return F
    end
    return AlgebraicSet(eqns,d,N,k,tol,f,df,f!)
end
AlgebraicSet(eqns,d::Int,N::Int) = AlgebraicSet(eqns,d,N,1e-8) # default tolerance

Base.show(io::IO, M::AlgebraicSet) = print(io, 
"An algebraic set of dimension $(M.varietydim) with ambient dimension $(
M.ambientdim) defined by the $(M.numeqns) polynomials $(M.eqns).")

g1(x) = (x[1]^4 + x[2]^4 - 1) * (x[1]^2 + x[2]^2 - 2) + x[1]^5 * x[2]
gs = [g1]
dim = 1
ambientdim = 2

M = AlgebraicSet(gs,dim,ambientdim)

p = [1.0; 0.0] # g1(p) = 0, so p is a point on the variety V(g1)
X = [1.0; 4.0] # check if this is a tangent vector, yes!

M.f(p), M.df(p)

M.df(p)'*X # dot product is zero since v is a tangent vector to p

function check_manifold_point(M::AlgebraicSet, p)
    # p is a point on the manifold
    (size(p)) == (M.ambientdim,) || return DomainError(size(p),"The size of $p is not $(M.ambientdim).")
    if norm( [eqn(p) for eqn in M.eqns] ) > M.residualtol
        return DomainError(p,
            "The norm of vector of evaluations of the equations at $p is not less than $(M.residualtol).")
    end
    return nothing
end

function check_tangent_vector(M::AlgebraicSet, p, X, check_base_point = true)
    # p is a point on the manifold, X is a tangent vector
    if check_base_point
        mpe = check_manifold_point(M, p)
        mpe === nothing || return mpe
    end
    size(X) != size(p) && return DomainError(size(X), "The size of $X is not $(size(p)).")
    if M.numeqns == 1
        if M.df(p)' * X > M.residualtol
            return DomainError( M.df(p)' * X, "The tangent $X is not orthogonal to $p.")
        end
    else
        if norm(M.df(p) * X) > M.residualtol
            return DomainError( norm(M.df(p) * X), "The tangent $X is not orthogonal to $p.")
        end
    end
    return nothing
end;

is_manifold_point(M, randn(2)) # should be false

@test_throws DomainError is_manifold_point(M, rand(3), true) # only on R^2, throws an error.

# The following two tests return true
[ is_manifold_point(M, p); is_tangent_vector(M,p,X) ]

manifold_dimension(M::AlgebraicSet) = M.varietydim

manifold_dimension(M)

using NLsolve

function exp!(M::AlgebraicSet, q, p, X)
    # mutates `q` to refer to the point on the manifold in tangent direction `X` from point `p`
    nX = norm(X)
    if nX == 0
        q .= p
    else
        #q .= cos(nX/M.radius)*p + M.radius*sin(nX/M.radius) .* (X./nX)
        initpt = p + X
        result = NLsolve.nlsolve(M.f!, initpt, autodiff=:forward)
        #println(result)
        q .= result.zero
    end
    return q
end

p = [1.0; 0.0] # g1(p) = 0, so p is a point on the variety V(g1)
X = [1.0; 4.0] # check if this is a tangent vector, yes!

X = normalize(X) / 100.

q = exp(M, p, X) # takes a moment because we're using NLsolve for the first time...

is_manifold_point(M,q)





using Manifolds, Manopt
#random_tangent(M, p, Val(:Gaussian)) throws an error, not sure why... anyway we don't need it.

# M is an `AlgebraicSet` defined above
initialpoint = [1.0,0.0] # let this be the initial point we start with. Try to find the closest point to `u`
u = [2.0,2.0] # find closest point on `M` to this `u`

F(M,y) = sum((y[i] - u[i])^2 for i in 1:2)
gradF(M,y) = ForwardDiff.gradient(y -> F(M,y), y)

# this throws an error. Asks for `log!` to be implemented on `AlgebraicSet`
#closestpoint = gradient_descent(M, F, gradF, initialpoint)

