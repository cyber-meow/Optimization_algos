# Uses the Newton method to find the analytic center of a polytope.

function polytope_analcenter(A::AbstractMatrix,
                             b::AbstractVector,
                             x0::AbstractVector,
                             ϵ::Real;
                             maxiter::Integer=40,
                             α::Real=0.01, β::Real=0.05)
  all(A*x0.<b) || throw(
    DomainError("x0 must lie strictly inside the polytope"))
  x = copy(x0)
  H = zeros(size(x0, 1), size(x0, 1))
  polytope_log(x) = -sum(log.(b-A*x))
  λ² = Inf
  while true
    d = 1./(b-A*x)
    ∇ = A'*d
    H = A'*Diagonal(d)^2*A
    Δxnt = -H\∇
    -∇⋅Δxnt > λ² && break
    λ² = -∇⋅Δxnt
    λ²/2 < ϵ && break
    t = 1
    xtest = x + t*Δxnt
    while(any(A*xtest .≥ b)
          || polytope_log(xtest) > polytope_log(x)-t*α*λ²)
      t *= β
      xtest = x + t*Δxnt
    end
    # numeric limit
    polytope_log(xtest) == polytope_log(x) && break
    x = xtest
  end
  x, H
end
