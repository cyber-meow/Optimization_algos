# A SVM solver based on the dual coordinate descent method.

# The naif implementation.
function svm_dualcoordinatedescent(
    X::AbstractMatrix, Y::AbstractVector,
    C::Real, ϵ::Real; maxiter::Integer=10000)
  α = C/2*ones(Y)
  w = X'*(Y.*α)
  Q = Diagonal(Y)*X*X'*Diagonal(Y)
  primal(w) = norm(w)^2/2 + C*sum(max.(0, ones(Y)-Y.*(X*w)))
  dual(α) = -α⋅(Q*α)/2 + sum(α)
  wmin = copy(w)
  wminvalue = primal(w)
  dualitygaps = [wminvalue-dual(α)]
  n = 0
  while dualitygaps[end] > ϵ && n < maxiter
    for i = 1:length(Y)
      ∇ᵢ = Q[i,:]⋅α - 1
      αᵢ = α[i]
      α[i] = min(max(α[i]-∇ᵢ/Q[i, i], 0), C)
      w += (α[i]-αᵢ)*Y[i]*X[i,:]
    end
    wvalue = primal(w)
    if wvalue < wminvalue 
      wmin, wminvalue = copy(w), wvalue
    end
    push!(dualitygaps, wminvalue-dual(α))
    n += 1
  end
  wmin, α, dualitygaps
end

# Add random permutation of indices inside each outer iteration
# and a more efficient way to compute the gradient of each univariate function.
function svm_dualcoordinatedescentopt(
    X::AbstractMatrix, Y::AbstractVector, C::Integer, iter::Integer)
  α = C/2*ones(Y)
  w = X'*(Y.*α)
  Qdiag = sum(X.^2, 2)
  primal(w) = norm(w)^2/2 + C*sum(max.(0, ones(Y)-Y.*(X*w)))
  wmin = copy(w)
  wminvalue = primal(w)
  for k = 1:iter
    for i = randperm(length(Y))
      ∇ᵢ = Y[i]*w⋅X[i,:] - 1
      p∇ᵢ = α[i]==0?min(∇ᵢ, 0):α[i]==C?max(∇ᵢ, 0):∇ᵢ
      if p∇ᵢ ≠ 0
        αᵢ = α[i]
        α[i] = min(max(α[i]-∇ᵢ/Qdiag[i], 0), C)
        w += (α[i]-αᵢ)*Y[i]*X[i,:]
      end
    end
    wvalue = primal(w)
    if wvalue < wminvalue 
      wmin, wminvalue = copy(w), wvalue
    end
  end
  wmin, α
end
