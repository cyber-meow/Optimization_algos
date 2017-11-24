# Solves the primal problem of a soft-margin SVM classifier using
# the ACCPM algorithm.

# The main function.
# This implementation is deterministic for illustration purpose.
# In practice we may want to use random starting point.
function svmaccpm(X::AbstractMatrix,
                  Y::AbstractVector,
                  C::Real,
                  ϵ::Real,
                  droppingstrategy::Symbol=:None;
                  α::Real=0.01, β::Real=0.05)
  w = zeros(size(X, 2))
  z = ones(Y)*2
  x0 = [w; z]
  A, b = initialpolytope(X, Y, C, w, z)
  wsize = size(w, 1)
  f(x) = sum(x[1: wsize].^2)/2 + C*sum(x[wsize+1: end])
  ∇f(x) = [x[1: size(w, 1)]; C*ones(Y)]
  x, optdistances, numconstraints = accpm(
    f, ∇f, A, b, x0, ϵ, droppingstrategy; α=α, β=β)
  x[1: wsize], optdistances, numconstraints
end

# Computes the initial polyhedron for the SVM problem with some given
# initial point.
# Notes that in the original problem formulation the polyhedron assocaited
# with the constraint is not bounded, but once given some possible value
# of the objective we can easily give an upper bound to |w| and z.
function initialpolytope(X::AbstractMatrix,
                         Y::AbstractVector,
                         C::Real,
                         w::AbstractVector,
                         z::AbstractVector)
  objvalue = sum(w.^2)/2 + C*sum(z)
  wupper = √(2*objvalue)
  zupper = objvalue/C
  A = [-Y.*X (-eye(size(Y, 1)));
       zeros(X) (-eye(size(Y, 1)));
       -eye(size(w, 1)) zeros(X)';
       eye(size(w, 1)+size(z, 1));]
  b = [-1.*ones(Y); zeros(Y);
       fill(wupper, 2*size(w, 1)); fill(zupper, size(z))]
  A, b
end

# The general ACCPM algorithm to minimize `f` using the first-order
# derivative information.
# We suppose a linear constraint Ax ≤ b exists for the original problem.
function accpm(f, ∇f,
               A::AbstractMatrix,
               b::AbstractVector, 
               x0::AbstractVector,
               ϵ::Real,
               droppingstrategy::Symbol=:None;
               α::Real=0.01, β::Real=0.05)
  (all(A*x0.<b)
   || throw(DomainError("x0 must be strictly feasible.")))
  (droppingstrategy in [:None, :DropRedundant, :KeepConstant]
   || throw(ArgumentError(
        "droppingstrategy must be one of the three:"
        * ":None, :DropRedundant or :KeepConstant.")))
  x = copy(x0)
  ms = [size(A, 1)]
  u, l = f(x), -Inf
  optdistances::Vector{Float64} = []
  while true
    x, H = polytope_analcenter(A, b, x, 1e-6; α=α, β=β)
    ∇, fx = ∇f(x), f(x)
    u = min(u, fx)
    l = max(l, fx - ms[end]*√(∇'*(H\∇)))
    push!(optdistances, u - l)
    u - l < ϵ && break
    if droppingstrategy ≠ :None
      A, b = dropconstraint(A, b, x, H, droppingstrategy)
    end
    A, b = [A; ∇'], [b; ∇'*x]
    x = nextstartpoint(A, b, x)
    push!(ms, size(A, 1))
  end
  x, optdistances, ms
end

# In the ACCPM algorithm, at the end of each iteration, the vector `x`
# is found on the face of the polytope. However, to run the Newton method
# that gives the analytic center, a strictly feasible point must be given.
# This function supposes that we have Ax ≤ b but with eqaulity only
# on the last line and try to move `x` a little to get a new `x'` such
# that Ax' ≤ b (the last line to `A` is suppose to be not all zero).
function nextstartpoint(A::AbstractMatrix,
                        b::AbstractVector,
                        x::AbstractVector)
  a = A[end, :]
  upperbound = (b-A*x)[1: end-1]
  lowerbound = max.(0, -(A*a)[1: end-1])
  δ = minimum(upperbound./lowerbound)
  x - δ*a/2
end

# Drop the constraints according to different strategies.
#
# :DropRedundant:
#   Guarantees not to change the polytope in question.
#
# :KeepConstant:
#   Keeps a fixed number of constraints which are the most relevant.
#   Here I choose to keep 3n constraints where n is the variable dimension.
#   However, we shouldn't end up with something that doesn't satisfy the
#   initial constraints, so this implementation can have bugs in some
#   particular cases.
function dropconstraint(A::AbstractMatrix,
                        b::AbstractVector,
                        x::AbstractVector,
                        H::AbstractMatrix,
                        droppingstrategy::Symbol)
  m = size(A, 1)
  Hinv = inv(H)
  relevemeasure = (b-A*x)./[√((A*Hinv)[i, :]⋅A[i, :]) for i=1:m]
  if droppingstrategy == :DropRedundant
    tokeep = relevemeasure .< m
  end
  if droppingstrategy == :KeepConstant
    tokeep = sortperm(relevemeasure)[1: min(3*size(A, 2), end)]
  end
  A[tokeep, :], b[tokeep]
end
