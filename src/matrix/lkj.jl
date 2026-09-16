using LinearAlgebra: LinearAlgebra as LA
using LogExpFunctions: logcosh

"""
    Corr

A bijector to transform a correlation matrix to an unconstrained vector.

# Reference
https://mc-stan.org/docs/reference-manual/transforms.html#correlation-matrices
"""
struct Corr <: AbstractBijector end

struct InvCorr <: AbstractBijector end
inverse(::Corr) = InvCorr()
inverse(::InvCorr) = Corr()

"""
    CorrCholesky(uplo)

A bijector to transform a Cholesky factor of a correlation matrix to an unconstrained vector.
`uplo` selects the upper (`:U` or `'U'`) or lower (`:L` or `'L'`) factor.

# Reference
https://mc-stan.org/docs/reference-manual/transforms.html#cholesky-factors-of-correlation-matrices
"""
struct CorrCholesky <: AbstractBijector
    mode::Symbol
    function CorrCholesky(uplo)
        s = Symbol(uplo)
        if (s === :U) || (s === :L)
            new(s)
        else
            throw(
                ArgumentError(
                    "mode must be either :U (upper triangular) or :L (lower triangular)",
                ),
            )
        end
    end
end

struct InvCorrCholesky <: AbstractBijector
    mode::Symbol
    InvCorrCholesky(b::CorrCholesky) = new(b.mode)
end
InvCorrCholesky(uplo) = InvCorrCholesky(CorrCholesky(uplo))
inverse(b::CorrCholesky) = InvCorrCholesky(b)
inverse(b::InvCorrCholesky) = CorrCholesky(b.mode)

_triu1_dim_from_length(d) = (1 + isqrt(1 + 8d)) ÷ 2

function _link_chol_lkj_from_upper(W::AbstractMatrix)
    K = LA.checksquare(W)
    y = similar(W, ((K - 1) * K) ÷ 2)
    starting_idx = 1
    for j in 2:K
        y[starting_idx] = atanh(W[1, j])
        starting_idx += 1
        # A tail norm also handles zero off-diagonal entries.
        remainder_sq = W[j, j]^2
        for i in (j-1):-1:2
            idx = starting_idx + i - 2
            y[idx] = asinh(W[i, j] / sqrt(remainder_sq))
            remainder_sq += W[i, j]^2
        end
        starting_idx += j - 2
    end
    return y
end

(::Corr)(X) = _link_chol_lkj_from_upper(cholesky_upper(X))
function (b::CorrCholesky)(X)
    W = if b.mode === :U
        cholesky_upper(X)
    else
        transpose(cholesky_lower(X))
    end
    return _link_chol_lkj_from_upper(W)
end

function with_logabsdet_jacobian(b::Union{Corr,CorrCholesky}, x)
    y = b(x)
    return y, -logabsdet_jacobian(inverse(b), y)
end

# The factor and correlation matrix use different independent coordinates.
_lkj_logjac_weight(::InvCorrCholesky, K, i, j) = j - i + 1
_lkj_logjac_weight(::InvCorr, K, i, j) = K - i + 1

function logabsdet_jacobian(b::Union{InvCorr,InvCorrCholesky}, y::AbstractVector)
    Base.require_one_based_indexing(y)
    K = _triu1_dim_from_length(length(y))
    logjac = zero(float(eltype(y)))
    idx = 1
    for j in 2:K, i in 1:(j-1)
        logjac -= _lkj_logjac_weight(b, K, i, j) * logcosh(y[idx])
        idx += 1
    end
    return logjac
end

# Avoid recording multiplication by the factor's unit weight during differentiation.
_lkj_logjac_column(::InvCorrCholesky, K, j, remainder) = remainder
_lkj_logjac_column(::InvCorr, K, j, remainder) = (K - j + 1) * remainder

function _inv_link_chol_lkj(y::AbstractVector, b::Union{InvCorr,InvCorrCholesky})
    K = _triu1_dim_from_length(length(y))
    W = similar(y, float(eltype(y)), K, K)
    logjac = zero(eltype(W))
    # Broadcast the scalar terms to reduce reverse-mode AD work and allocations.
    tanhy = tanh.(y)
    logcoshy = logcosh.(y)
    idx = 1
    for j in 1:K
        log_remainder = zero(logjac)
        for i in 1:(j-1)
            W[i, j] = tanhy[idx] * exp(log_remainder)
            log_remainder -= logcoshy[idx]
            logjac += log_remainder
            idx += 1
        end
        # Correlations add K-j copies of each column's log remainder.
        logjac += _lkj_logjac_column(b, K, j, log_remainder)
        W[j, j] = exp(log_remainder)
        for i in (j+1):K
            W[i, j] = 0
        end
    end
    return W, logjac
end

function with_logabsdet_jacobian(b::InvCorr, y::AbstractVector)
    Base.require_one_based_indexing(y)
    U, logjac = _inv_link_chol_lkj(y, b)
    return U' * U, logjac
end

function with_logabsdet_jacobian(b::InvCorrCholesky, y::AbstractVector)
    Base.require_one_based_indexing(y)
    factors, logjac = _inv_link_chol_lkj(y, b)
    # Fill columns with adjacent writes, then transpose once for a lower factor.
    # This gives faster Enzyme gradients than filling the lower factor by rows.
    if b.mode === :U
        return LA.Cholesky(factors, :U, 0), logjac
    else
        return LA.Cholesky(permutedims(factors), :L, 0), logjac
    end
end
