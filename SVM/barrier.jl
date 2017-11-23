# Solves the SVM classification problem using the barrier method.

# The main function.
# This implementation is deterministic for illustration purpose.
# In practice we may want to use random starting point.
function svmbarrier(X::AbstractMatrix,
                    Y::AbstractVector,
                    C::Real,
                    ϵ::Real;
                    t0::Real=1., 
                    μ::Real=100.,
                    α::Real=0.01, β::Real=0.05)
  w = zeros(size(X, 2))
  z = ones(Y)*2
  x0 = [w; z]
  A = [-Y.*X (-eye(size(Y, 1))); zeros(X) (-eye(size(Y, 1)))]
  b = [-1.*ones(Y); zeros(Y)]
  wsize = size(w, 1)
  f(x) = sum(x[1: wsize].^2)/2 + C*sum(x[wsize+1: end])
  ∇f(x) = [x[1: size(w, 1)]; C*ones(Y)]
  Hf(x) = Diagonal([ones(wsize); zeros(Y)])
  x, αdual, numstepsarray = barriermethod(
    f, ∇f, Hf, A, b, x0; ϵ=ϵ, t0=t0, μ=μ, α=α, β=β)
  x[1: wsize], αdual[1: size(Y, 1)], numstepsarray
end

# Implements the barrier method given the objective function and its
# first and second derivatives.
# The problem is supposed to have a linear constraint Ax ≤ b.
function barriermethod(f, ∇f, Hf,
                       A::AbstractMatrix,
                       b::AbstractVector,
                       x0::AbstractVector;
                       ϵ::Real=1e-3,
                       t0::Real=1.,
                       μ::Real=100.,
                       α::Real=0.01, β::Real=0.05)
  (all(A*x0.<b)
   || throw(DomainError("x0 must be strictly feasible.")))
  m = float(size(Y, 1))
  x, t = x0, t0
  numstepsarray::Vector{Int} = []
  while true
    x, numsteps = centeringstep(f, ∇f, Hf, A, b, x, t; α=α, β=β)
    push!(numstepsarray, numsteps)
    m/t < ϵ && break
    t *= μ
  end
  αdual = 1./(t*(b-A*x))
  x, αdual, numstepsarray
end

# Carries out a single centering step of the barrier method.
# The problem is supposed to have a linear constraint Ax ≤ b.
function centeringstep(f, ∇f, Hf,
                       A::AbstractMatrix,
                       b::AbstractVector,
                       x0::AbstractVector,
                       t::Real;
                       ϵ::Real=1e-8,
                       α::Real=0.01, β::Real=0.05)
  x = copy(x0)
  numsteps = 0
  obj(x) = t*f(x)-sum(log.(b-A*x))
  while true
    d = 1./(b-A*x)
    ∇ = t*∇f(x)+A'*d
    H = t*Hf(x)+A'*Diagonal(d)^2*A
    Δxnt = -H\∇
    λ² = -∇⋅Δxnt
    λ²/2 < ϵ && break
    tbacktrace = 1
    xtest = x + tbacktrace*Δxnt
    while(any(A*xtest .≥ b) || obj(xtest) > obj(x)-tbacktrace*α*λ²)
      tbacktrace *= β
      xtest = x + tbacktrace*Δxnt
    end
    # numeric limit
    obj(xtest) == obj(x) && break
    x = xtest
    numsteps += 1
  end
  x, numsteps
end
