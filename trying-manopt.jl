using ManifoldsBase, LinearAlgebra, Test
import ManifoldsBase: check_manifold_point, check_tangent_vector, manifold_dimension, exp!, log!, inner

using LinearAlgebra, ForwardDiff, HomotopyContinuation

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

# N=ambient dimension. I don't know the inherent advantage of writing it this way, but this is the way it is
# set up in ManifoldsBase, so I concurred.

struct AlgebraicSet{N} <: Manifold{ManifoldsBase.ℝ} where {N}
    eqns
    varietydim::Int
    numeqns::Int
    residualtol::Float64
    f
    df
    f!
    EDSystem
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
            F[i] = 0
        end
        return F
    end
    
    # Initialize the EDSystem
    HomotopyContinuation.@var varz[1:N]
    algeqnz = [eqn(varz) for eqn in eqns]
    HomotopyContinuation.@var u[1:N]
    HomotopyContinuation.@var λ[1:length(algeqnz)]
    Lagrange = Base.sum((varz-u).^2) + sum(λ.*algeqnz)
    ∇Lagrange = HomotopyContinuation.differentiate(Lagrange, vcat(varz,λ))
    EDSystem = HomotopyContinuation.System(∇Lagrange, variables=vcat(varz,λ), parameters=u)

    return AlgebraicSet{N}(eqns,d,k,tol,f,df,f!,EDSystem)
end
AlgebraicSet(eqns,d::Int,N::Int) = AlgebraicSet(eqns,d,N,1e-8) # default tolerance

Base.show(io::IO, M::AlgebraicSet{N}) where {N} = print(io,
"An algebraic set of dimension $(M.varietydim) with ambient dimension $(
N) defined by the $(M.numeqns) polynomials $(M.eqns).")






g1(x) = (x[1]^4 + x[2]^4 - 1) * (x[1]^2 + x[2]^2 - 2) + x[1]^5 * x[2]
gs = [g1]
dim = 1
ambientdim = 2

M = AlgebraicSet(gs,dim,ambientdim)





function check_manifold_point(M::AlgebraicSet{N}, p) where{N}
    # p is a point on the manifold
    (size(p)) == (N,) || return DomainError(size(p),"The size of $p is not $(M.ambientdim).")
    if norm( [eqn(p) for eqn in M.eqns] ) > M.residualtol
        return DomainError(p,
            "The norm of vector of evaluations of the equations at $p is not less than $(M.residualtol).")
    end
    return nothing
end

function check_tangent_vector(M::AlgebraicSet{N}, p, X, check_base_point = true) where {N}
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

manifold_dimension(M::AlgebraicSet) = M.varietydim





using HomotopyContinuation, LinearAlgebra

function exp!(M::AlgebraicSet{N}, q, p, X) where {N}
    # mutates `q` to refer to the point on the manifold in tangent direction `X` from point `p`
    check_tangent_vector(M,p,X); check_manifold_point(M,p);
    nX = norm(X)
    if norm(X) < 1e-12
        q .= p
    else
        u0 = p
        u1 = p+X
        # TODO: This is redundant. We could only do this once and carry l0 on during the calculations
        A = HomotopyContinuation.evaluate(HomotopyContinuation.differentiate(M.EDSystem.expressions, M.EDSystem.variables[N+1:end]), M.EDSystem.variables[1:N] => p)
        l0 = A\(-HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(HomotopyContinuation.evaluate(M.EDSystem.expressions, M.EDSystem.variables[N+1:end] => [0 for _ in N+1:length(M.EDSystem.variables)]), M.EDSystem.variables[1:N] => p),  M.EDSystem.parameters=>u0))
        # Solve the EDStep-system
        res = HomotopyContinuation.solve( M.EDSystem, vcat(p, l0); start_parameters = u0, target_parameters = u1)
        q.= (real_solutions(res)[1])[1:N]
    end
    return q
end

function log!(M::AlgebraicSet{N}, X, p, q) where {N}
    # project q back to the tangent space of p via orthogonal projection relative to T_q(M). 
    # log is supposed to invert exp. This is done by finding the solution of N_q(M)+q ∩ T_p(V)+p (for regular p,q)
    check_manifold_point(M,p); check_manifold_point(M,q);
    Jsp=M.df(p)
    Jp = Array{Float64,2}(undef, size(Jsp)[1], size(Jsp)!=(size(Jsp)[1],) ? size(Jsp)[2] : 1 )
    size(Jsp)!=(size(Jsp)[1],) ? Jp = Jsp : Jp[:,1] = Jsp
    Qp,_ = LinearAlgebra.qr(Jp)
    Np = Qp[:, 1:(N - M.varietydim)] # basis of p's normal space
    
    Jsq=M.df(q)
    Jq = Array{Float64,2}(undef, size(Jsq)[1], size(Jsq)!=(size(Jsq)[1],) ? size(Jsq)[2] : 1 )
    size(Jsq)!=(size(Jsq)[1],) ? Jq = Jsq : Jq[:,1] = Jsq
    Qq,_ = LinearAlgebra.qr(Jq)
    Tq = Qq[:, (N - M.varietydim + 1):end] # basis of q's tangent space
    
    @var ambientvarz[1:N]
    L = HomotopyContinuation.System(vcat(Np'*ambientvarz .- Np'*p, Tq'*ambientvarz .- Tq'*q))
    # The projected tangent vector is solution - basepoint
    res = HomotopyContinuation.solve(L)
    X .= HomotopyContinuation.real_solutions(res)[1] .- p
    return X
end

function inner(N::AlgebraicSet, p, X, Y)
    # Calculate the standard inner product in T_x(M)
    check_tangent_vector(M,p,X); check_tangent_vector(M,p,Y);
    return(X'*Y)
end


p = [1.0, 0.0]
X = [1.0, 4.0]

# We try to see whether log(exp) = exp(log) = id
q = exp(M, p, X)
X0 = log(M, p, q)
q1=exp(M,p,X0)
display(q1-q)
display(X0-X)



initialpoint = [1.0,0.0] # let this be the initial point we start with. Try to find the closest point to `u`
u = [2.0,2.0] # find closest point on `M` to this `u`

F(M,y) = sum((y[i] - u[i])^2 for i in 1:2)
gradF(M,y) = ForwardDiff.gradient(y -> F(M,y), y)

# this throws an error. Asks for `log!` to be implemented on `AlgebraicSet`
closestpoint = gradient_descent(M, F, gradF, initialpoint)


