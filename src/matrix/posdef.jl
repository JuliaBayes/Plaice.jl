# Distributions over positive definite matrices.

import LinearAlgebra as LA
import IrrationalConstants: logtwo

struct PosDef <: AbstractBijector
    original_size::Int
end
function with_logabsdet_jacobian(p::PosDef, x::AbstractMatrix{T}) where {T<:Number}
    Base.require_one_based_indexing(x)
    LA.checksquare(x)
    d = p.original_size
    L = cholesky_lower(x)
    yvec = similar(L, div(d * (d + 1), 2))
    idx = 1
    z = zero(eltype(L))
    weight = d + 1
    for i in 1:d
        for j in 1:i
            if i == j
                logLij = log(L[i, j])
                yvec[idx] = logLij
                z -= weight * logLij
                weight -= 1
            else
                yvec[idx] = L[i, j]
            end
            idx += 1
        end
    end
    logjac = z - (d * oftype(z, logtwo))
    return yvec, logjac
end
inverse(p::PosDef) = InvPosDef(p.original_size)

function logabsdet_jacobian(p::PosDef, x::AbstractMatrix{T}) where {T<:Number}
    L = cholesky_lower(x)
    d = p.original_size
    z = zero(eltype(L))
    for i in 1:d
        z -= (d + 2 - i) * log(L[i, i])
    end
    return z - (d * oftype(z, logtwo))
end

struct InvPosDef <: AbstractBijector
    original_size::Int
end
function with_logabsdet_jacobian(ip::InvPosDef, yvec::AbstractVector{T}) where {T<:Number}
    d = ip.original_size
    X = similar(yvec, float(T), d, d)
    z = zero(eltype(X))
    weight = d + 1
    for j in 1:d
        for i in 1:(j-1)
            X[i, j] = 0
        end
        # Coordinates use lower-triangle row order; matrix writes follow columns.
        idx = div(j * (j + 1), 2)
        X[j, j] = exp(yvec[idx])
        z += weight * yvec[idx]
        weight -= 1
        for i in (j+1):d
            idx += i - 1
            X[i, j] = yvec[idx]
        end
    end
    logjac = z + (d * oftype(z, logtwo))
    return X * X', logjac
end
inverse(ip::InvPosDef) = PosDef(ip.original_size)

function logabsdet_jacobian(ip::InvPosDef, yvec::AbstractVector{T}) where {T<:Number}
    d = ip.original_size
    logjac = d * oftype(zero(float(T)), logtwo)
    for i in 1:d
        logjac += (d + 2 - i) * yvec[div(i * (i + 1), 2)]
    end
    return logjac
end
