# src/test_utils.jl

using LinearAlgebra: Cholesky, UpperTriangular, LowerTriangular
using Test: @test

VB._is_continuous(::D.Distribution{<:Any,VS}) where {VS<:D.ValueSupport} = VS

VB._name(d::D.Distribution) = nameof(typeof(d))
VB._name(d::D.Censored) = "censored $(VB._name(d.uncensored)) [$(d.lower),$(d.upper)]"
function VB._name(d::D.Truncated)
    return "truncated $(VB._name(d.untruncated)) [$(d.lower),$(d.upper)]"
end
function VB._name(d::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:D.Distribution})
    return "reshaped $(VB._name(d.dist)) to size $(size(d))"
end
VB._name(d::D.OrderStatistic) = "order statistic $(VB._name(d.dist))"
function VB._name(d::D.JointOrderStatistics)
    return "joint order statistic $(VB._name(d.dist)) with length $(length(d))"
end

VB._rand_safe_ad(d::D.Distribution) = rand(d)
VB._rand_safe_ad(d::D.Censored) = begin
    a, b = d.lower, d.upper
    while true
        x = rand(d)
        if x != a && x != b
            return x
        end
    end
end

VB.to_vec_for_logjac_test(::Union{D.Dirichlet,D.MvLogitNormal}) = x -> x[1:(end-1)]
VB.from_vec_for_logjac_test(::Union{D.Dirichlet,D.MvLogitNormal}) = y -> vcat(y, 1 - sum(y))
function VB.to_vec_for_logjac_test(
    d::Union{<:D.ProductDistribution,<:D.ProductNamedTupleDistribution},
)
    # Internal function, but we use this to avoid a LOT of code duplication
    return VB._make_transform(
        d.dists,
        to_vec_for_logjac_test,
        VB.linked_vec_length,
        VB.ProductVecTransform,
    )
end
function VB.from_vec_for_logjac_test(
    d::Union{<:D.ProductDistribution,<:D.ProductNamedTupleDistribution},
)
    return VB._make_transform(
        d.dists,
        from_vec_for_logjac_test,
        VB.linked_vec_length,
        VB.ProductVecInvTransform,
    )
end
function VB.to_vec_for_logjac_test(
    ::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:Union{D.Dirichlet,D.MvLogitNormal}},
)
    return x -> vec(x)[1:(end-1)]
end
function VB.from_vec_for_logjac_test(
    d::D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:Union{D.Dirichlet,D.MvLogitNormal}},
)
    return y -> reshape(vcat(y, 1 - sum(y)), size(d))
end
struct CholeskyToVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyToVecForLogjac)(x::Cholesky{T}) where {T<:Number}
    # Same as to_vec, but skip the diagonal entries.
    indices = VB._get_cartesian_indices(c.n, c.uplo)
    vec_len = div(c.n * (c.n - 1), 2)
    xvec = Vector{T}(undef, vec_len)
    idx = 1
    for (i, j) in indices
        if i != j
            xvec[idx] = x.UL[i, j]
            idx += 1
        end
    end
    return xvec
end
VB.to_vec_for_logjac_test(d::D.LKJCholesky) = CholeskyToVecForLogjac(first(size(d)), d.uplo)
struct CholeskyFromVecForLogjac
    n::Int
    uplo::Char
end
function (c::CholeskyFromVecForLogjac)(xvec::AbstractVector{T}) where {T<:Number}
    # Same as from_vec, but skip the diagonal entries, and reconstruct them
    # from the fact that the rows/columns are unit-norm.
    indices = VB._get_cartesian_indices(c.n, c.uplo)
    x = if c.uplo == 'U'
        Cholesky(UpperTriangular(zeros(T, c.n, c.n)))
    else
        Cholesky(LowerTriangular(zeros(T, c.n, c.n)))
    end
    idx = 1
    for (i, j) in indices
        if i != j
            x.UL[i, j] = xvec[idx]
            idx += 1
        end
    end
    for i in 1:(c.n)
        # x.UL[i, i] is still zero now, so we can compute the sum-of-squares
        # including it, before then calculating it
        sum_sq = if c.uplo == 'U'
            sum(abs2, x.UL[:, i])
        else
            sum(abs2, x.UL[i, :])
        end
        x.UL[i, i] = sqrt(one(T) - sum_sq)
    end
    return x
end
function VB.from_vec_for_logjac_test(d::D.LKJCholesky)
    return CholeskyFromVecForLogjac(first(size(d)), d.uplo)
end

function VB.to_vec_for_logjac_test(d::D.ReshapedDistribution)
    return rx -> begin
        x = VB._reshape_or_only(rx, size(d.dist))
        return VB.to_vec_for_logjac_test(d.dist)(x)
    end
end
function VB.from_vec_for_logjac_test(d::D.ReshapedDistribution)
    return yvec -> begin
        x = VB.from_vec_for_logjac_test(d.dist)(yvec)
        return VB._reshape_or_only(x, size(d))
    end
end

# These are positive (semi)definite matrix distributions, which are symmetric, so we will
# just vectorise the lower-triangular part.
function VB.to_vec_for_logjac_test(d::Union{D.Wishart,D.InverseWishart})
    n = first(size(d))
    return x -> begin
        vec_len = div(n * (n + 1), 2)
        xvec = zeros(eltype(x), vec_len)
        idx = 1
        for i in 1:n, j in 1:i
            xvec[idx] = x[i, j]
            idx += 1
        end
        return xvec
    end
end
function VB.from_vec_for_logjac_test(d::Union{D.Wishart,D.InverseWishart})
    n = first(size(d))
    return xvec -> begin
        x = zeros(eltype(xvec), n, n)
        idx = 1
        for i in 1:n, j in 1:i
            x[i, j] = xvec[idx]
            x[j, i] = xvec[idx]
            idx += 1
        end
        return x
    end
end

# These are correlation matrices - they are symmetric and the diagonal is all ones
function VB.to_vec_for_logjac_test(d::D.LKJ)
    n = first(size(d))
    return x -> begin
        vec_len = div(n * (n - 1), 2)
        xvec = zeros(eltype(x), vec_len)
        idx = 1
        for i in 1:n, j in 1:(i-1)
            xvec[idx] = x[i, j]
            idx += 1
        end
        return xvec
    end
end
function VB.from_vec_for_logjac_test(d::D.LKJ)
    n = first(size(d))
    return xvec -> begin
        x = ones(eltype(xvec), n, n)
        idx = 1
        for i in 1:n, j in 1:(i-1)
            x[i, j] = xvec[idx]
            x[j, i] = xvec[idx]
            idx += 1
        end
        return x
    end
end

function VB.can_test_in_support(dist::D.Distribution, x)
    # Check that Distributions.jl can actually run insupport. Sometimes it can't, e.g.
    # with product_distribution(MvNormal(), MvNormal()), even though that function is
    # well-defined.
    return hasmethod(D.insupport, Tuple{typeof(dist),typeof(x)})
end
function VB.test_in_support(dist::D.Distribution, x)
    in_support = D.insupport(dist, x)
    if in_support isa Bool
        @test in_support
    elseif in_support isa AbstractArray{Bool,0}
        # This happens sometimes:
        # https://github.com/JuliaStats/Distributions.jl/issues/2026
        @test in_support[]
    else
        # We _could_ just check `all(in_support)`, but I don't want to be
        # caught off-guard by any bugs in the bijector's implementation that
        # returns a wrong shape/type of `x`.
        error(
            "Distributions.insupport returned unexpected type: $(typeof(in_support))",
        )
    end
end
