"""
    OnlyWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
vectors of length one to scalars.
"""
struct OnlyWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::OnlyWrap)(x) = w.bijector(x[])
function with_logabsdet_jacobian(w::OnlyWrap, x::AbstractVector)
    return with_logabsdet_jacobian(w.bijector, x[])
end
inverse(w::OnlyWrap) = VectWrap(inverse(w.bijector))

"""
    VectWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
scalars to vectors of length one.
"""
struct VectWrap{B<:ScalarToScalarBijector}
    bijector::B
end
(w::VectWrap)(x) = [w.bijector(x)]
function with_logabsdet_jacobian(w::VectWrap, x::Number)
    y, ladj = with_logabsdet_jacobian(w.bijector, x)
    return ([y], ladj)
end
inverse(w::VectWrap) = OnlyWrap(inverse(w.bijector))

"""
    VectorBijectors.scalar_to_scalar_bijector(dist)

The VectorBijectors interface is intended to map samples to vectors. However, for univariate
distributions, the 'vectorisation' part of this is trivial (we only need to convert a scalar
to a vector of length one, and vice versa). Therefore, this function is provided to allow
users to specify the 'interesting' part of the transformation, which is the function that
maps values to unconstrained space.

Overloading this function for a univariate distribution is sufficient to implement the
entire VectorBijectors interface for that distribution.

There are three scalar-to-scalar bijectors that are exported, which should be enough for any
univariate distribution:

- [`VectorBijectors.TypedIdentity`](@ref)
- [`VectorBijectors.Log`](@ref)
- [`VectorBijectors.Untruncate`](@ref)

If you need a different scalar-to-scalar bijector, please open an issue.
"""
function scalar_to_scalar_bijector end
