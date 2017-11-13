# Solve the SVM classification problem using the barrier method.

function svmcentering(
    X::AbstractMatrix, Y::AbstractVector, C::Real,
    w::AbstractVector, z::AbstractVector, t::Real;
    ϵ::Real=1e-10, α::Real=0.01, β::Real=0.05)
  A = [-Y.*X (-eye(size(Y, 1))); zeros(X) (-eye(size(Y, 1)))]
  b = [-1.*ones(Y); zeros(Y)]
  w, z = copy(w), copy(z)
  x = [w; z]
  f(w, z) = t*(norm(w)^2/2+C*sum(z))-sum(log.(Y.*(X*w)+z-1)+log.(z))
  fx(x) = f(x[1: size(w, 1)], x[size(w, 1)+1: end])
  numsteps = 0
  while true
    d = 1./(b-A*x)
    ∇ = t*[w; C*ones(Y)]+A'*d
    H = t*Diagonal([ones(w); zeros(Y)])+A'*Diagonal(d)^2*A
    Δxnt = -H\∇
    λ² = -∇⋅Δxnt
    λ²/2 < ϵ && break
    tbacktrace = 1
    xtest = x + tbacktrace*Δxnt
    while(any(A*xtest .≥ b) || fx(xtest) > fx(x)-tbacktrace*α*λ²)
      tbacktrace *= β
      xtest = x + tbacktrace*Δxnt
    end
    assert(fx(xtest) <= fx(x))
    # numeric limit
    fx(xtest) == fx(x) && break
    x = xtest
    w, z = x[1: size(w, 1)], x[size(w, 1)+1: end]
    numsteps += 1
  end
  w, z, numsteps
end

# This implementation is deterministic for illustration purpose.
# In practice we may want to use random starting points.
function svmbarrier(
    X::AbstractMatrix, Y::AbstractVector, C::Real, ϵ::Real;
    t0::Real=1., μ::Real=100., α::Real=0.01, β::Real=0.05)
  w = zeros(size(X, 2))
  z = ones(Y)*2
  m = float(2*size(Y, 1))
  t = t0
  numstepsarray::Vector{Int} = []
  while true
    w, z, numsteps = svmcentering(X, Y, C, w, z, t; α=α, β=β)
    push!(numstepsarray, numsteps)
    m/t < ϵ && break
    t *= μ
  end
  αdual = 1./(t*Y.*X*w+z-1)
  w, αdual, numstepsarray
end
