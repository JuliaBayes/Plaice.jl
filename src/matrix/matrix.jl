# Generic definitions for matrix distributions.

using LinearAlgebra: LinearAlgebra as LA

function _cholesky_factors(X, uplo)
    if X.uplo == uplo
        return X.factors
    else
        return X.factors'
    end
end

cholesky_upper(X::AbstractMatrix) = cholesky_upper(LA.cholesky(LA.Hermitian(X)))
cholesky_upper(X::LA.Cholesky) = X.U
cholesky_upper(X) = LA.UpperTriangular(_cholesky_factors(X, 'U'))

cholesky_lower(X::AbstractMatrix) = cholesky_lower(LA.cholesky(LA.Hermitian(X, :L)))
cholesky_lower(X::LA.Cholesky) = X.L
cholesky_lower(X) = LA.LowerTriangular(_cholesky_factors(X, 'L'))

# Somehow, ChangesOfVariables doesn't have a logjac implemented for `vec`, so we need to
# wrap it.
struct Vec{N} <: AbstractBijector
    size::NTuple{N,Int}
end
(::Vec)(x::AbstractArray) = vec(x)
inverse(v::Vec) = Reshape(v.size)

function with_logabsdet_jacobian(::Vec, x::AbstractArray{T,N}) where {T<:Number,N}
    return vec(x), zero(T)
end
function with_logabsdet_jacobian(::Vec, x::AbstractArray)
    return vec(x), false
end

struct Reshape{N} <: AbstractBijector
    size::NTuple{N,Int}
end
(r::Reshape)(x::AbstractArray) = reshape(x, r.size)
inverse(r::Reshape) = Vec(r.size)
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray{T,N}) where {T<:Number,N}
    return reshape(x, r.size), zero(T)
end
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray)
    return reshape(x, r.size), false
end
