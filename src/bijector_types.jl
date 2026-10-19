using LogExpFunctions: LogExpFunctions

# ---- is_monotonically_increasing / is_monotonically_decreasing ----

"""
    is_monotonically_increasing(f)

Returns `true` if `f` is monotonically increasing.
"""
is_monotonically_increasing(f) = false
is_monotonically_increasing(::typeof(identity)) = true

"""
    is_monotonically_decreasing(f)

Returns `true` if `f` is monotonically decreasing.
"""
is_monotonically_decreasing(f) = false
is_monotonically_decreasing(::typeof(identity)) = false

# ---- SimplexBijector ----

####################
# Simplex bijector #
####################
struct SimplexBijector <: AbstractBijector end

(b::SimplexBijector)(x) = _simplex_bijector(x, b)
inverse(::SimplexBijector) = InverseSimplexBijector()

function with_logabsdet_jacobian(b::SimplexBijector, x)
    return _simplex_bijector(x, b), _logabsdetjac_simplex(b, x)
end

struct InverseSimplexBijector <: AbstractBijector end
inverse(::InverseSimplexBijector) = SimplexBijector()

(::InverseSimplexBijector)(y) = _simplex_inv_bijector(y)

function with_logabsdet_jacobian(::InverseSimplexBijector, y)
    x = _simplex_inv_bijector(y)
    return x, -_logabsdetjac_simplex(SimplexBijector(), x)
end

function _simplex_bijector(x::AbstractArray, b::SimplexBijector)
    sz = size(x)
    K = size(x, 1)
    y = similar(x, Base.setindex(sz, K - 1, 1))
    _simplex_bijector!(y, x, b)
    return y
end

# Vector implementation.
function _simplex_bijector!(y, x::AbstractVector, ::SimplexBijector)
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(x)
    remaining = x[K]
    # Sum the remaining mass directly to avoid cancellation near the simplex boundary.
    for k in (K-1):-1:1
        y[k] = log(x[k]) - log(remaining) + log(T(K - k))
        remaining += x[k]
    end
    return y
end

# Matrix implementation.
function _simplex_bijector!(Y, X::AbstractMatrix, ::SimplexBijector)
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(X)
    for n in 1:size(X, 2)
        remaining = X[K, n]
        for k in (K-1):-1:1
            Y[k, n] = log(X[k, n]) - log(remaining) + log(T(K - k))
            remaining += X[k, n]
        end
    end

    return Y
end

function _simplex_inv_bijector(y)
    sz = size(y)
    K = sz[1] + 1
    x = similar(y, Base.setindex(sz, K, 1))
    _simplex_inv_bijector!(x, y)
    return x
end

function _simplex_inv_bijector!(x, y::AbstractVector)
    K = length(y) + 1
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(y)
    remaining = one(T)
    for k in 1:(K-1)
        t = y[k] - log(T(K - k))
        x[k] = remaining * LogExpFunctions.logistic(t)
        # Subtracting logistic(t) from one loses small positive tails.
        remaining *= LogExpFunctions.logistic(-t)
    end
    x[K] = remaining
    return x
end

function _simplex_inv_bijector!(X, Y::AbstractMatrix)
    K, N = size(Y, 1) + 1, size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(Y)
    for n in 1:size(X, 2)
        remaining = one(T)
        for k in 1:(K-1)
            t = Y[k, n] - log(T(K - k))
            X[k, n] = remaining * LogExpFunctions.logistic(t)
            remaining *= LogExpFunctions.logistic(-t)
        end
        X[K, n] = remaining
    end

    return X
end

_logabsdetjac_simplex(::SimplexBijector, x::AbstractVector) = -sum(log, x)

# Needed to avoid falling back to `with_logabsdet_jacobian` for matrix inputs.
function _logabsdetjac_simplex(b::SimplexBijector, x::AbstractMatrix{<:Number})
    return sum(Base.Fix1(_logabsdetjac_simplex, b), eachcol(x))
end
