using LinearAlgebra: LinearAlgebra as LA

function _get_cartesian_indices(n::Int, uplo::Char)
    if uplo == 'U'
        return [(i, j) for j in 1:n for i in 1:j]
    else
        return [(i, j) for j in 1:n for i in j:n]
    end
end

struct CholeskyVec <: AbstractBijector
    n::Int
    uplo::Char
end
function (c::CholeskyVec)(x::LA.Cholesky)
    factors = x.UL
    y = similar(x.factors, div(c.n * (c.n + 1), 2))
    idx = 1
    for j in 1:c.n
        rows = if c.uplo == 'U'
            1:j
        else
            j:c.n
        end
        for i in rows
            y[idx] = factors[i, j]
            idx += 1
        end
    end
    return y
end
function with_logabsdet_jacobian(c::CholeskyVec, x::LA.Cholesky{T}) where {T<:Number}
    return (c(x), zero(T))
end
logabsdet_jacobian(::CholeskyVec, ::LA.Cholesky{T}) where {T<:Number} = zero(T)

struct CholeskyUnVec <: AbstractBijector
    n::Int
    uplo::Char
end
inverse(c::CholeskyVec) = CholeskyUnVec(c.n, c.uplo)
inverse(c::CholeskyUnVec) = CholeskyVec(c.n, c.uplo)

function (c::CholeskyUnVec)(xvec::AbstractVector{T}) where {T<:Number}
    factors = similar(xvec, c.n, c.n)
    fill!(factors, zero(T))
    idx = 1
    for j in 1:c.n
        rows = if c.uplo == 'U'
            1:j
        else
            j:c.n
        end
        for i in rows
            factors[i, j] = xvec[idx]
            idx += 1
        end
    end
    return LA.Cholesky(factors, c.uplo, 0)
end
function with_logabsdet_jacobian(c::CholeskyUnVec, x::AbstractVector{T}) where {T<:Number}
    return (c(x), zero(T))
end
logabsdet_jacobian(::CholeskyUnVec, ::AbstractVector{T}) where {T<:Number} = zero(T)
