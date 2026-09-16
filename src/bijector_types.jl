using LogExpFunctions: LogExpFunctions

# ---- Helpers ----

_eps(::Type{T}) where {T<:Number} = T(eps(T))
_eps(::Type{Real}) = eps(Float64)
_eps(::Type{T}) where {T<:Integer} = eps(Float64)

function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    return clamped_x
end

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
    ϵ = _eps(T)
    sum_tmp = zero(T)
    z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    y[1] = LogExpFunctions.logit(z) + log(T(K - 1))
    for k in 2:(K-1)
        sum_tmp += x[k-1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        y[k] = LogExpFunctions.logit(z) + log(T(K - k))
    end
    return y
end

# Matrix implementation.
function _simplex_bijector!(Y, X::AbstractMatrix, ::SimplexBijector)
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(X)
    ϵ = _eps(T)
    for n in 1:size(X, 2)
        sum_tmp = zero(T)
        z = X[1, n] * (one(T) - 2ϵ) + ϵ
        Y[1, n] = LogExpFunctions.logit(z) + log(T(K - 1))
        for k in 2:(K-1)
            sum_tmp += X[k-1, n]
            z = (X[k, n] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
            Y[k, n] = LogExpFunctions.logit(z) + log(T(K - k))
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
    ϵ = _eps(T)
    z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
    x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
    sum_tmp = zero(T)
    for k in 2:(K-1)
        z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
    end
    sum_tmp += x[K-1]
    x[K] = _clamp(one(T) - sum_tmp, 0, 1)
    return x
end

function _simplex_inv_bijector!(X, Y::AbstractMatrix)
    K, N = size(Y, 1) + 1, size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(Y)
    ϵ = _eps(T)
    for n in 1:size(X, 2)
        sum_tmp, z = zero(T), LogExpFunctions.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
        for k in 2:(K-1)
            z = LogExpFunctions.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k-1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
        end
        sum_tmp += X[K-1, n]
        X[K, n] = _clamp(one(T) - sum_tmp, 0, 1)
    end

    return X
end

function _logabsdetjac_simplex(b::SimplexBijector, x::AbstractVector{T}) where {T}
    ϵ = _eps(T)
    lp = zero(T)

    K = length(x)

    sum_tmp = zero(eltype(x))
    z = x[1]
    lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ))
    for k in 2:(K-1)
        sum_tmp += x[k-1]
        z = x[k] / max(one(T) - sum_tmp, ϵ)
        lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ)) + log(max(one(T) - sum_tmp, ϵ))
    end

    return -lp
end

# Needed to avoid falling back to `with_logabsdet_jacobian` for matrix inputs.
function _logabsdetjac_simplex(b::SimplexBijector, x::AbstractMatrix{<:Number})
    return sum(Base.Fix1(_logabsdetjac_simplex, b), eachcol(x))
end
