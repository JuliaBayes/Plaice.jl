"""
    Exp(bound, sign) <: ScalarToScalarBijector

Callable struct, defined such that `(e::Exp)(y) = ((e.sign * exp(y)) + e.bound)`. The sign
is determined by the `sign` field.
"""
struct Exp{L<:Real} <: ScalarToScalarBijector
    bound::L
    sign::Int
end
is_monotonically_increasing(e::Exp) = e.sign > 0
is_monotonically_decreasing(e::Exp) = e.sign < 0
(e::Exp)(y::Real) = first(with_logabsdet_jacobian(e, y))
function with_logabsdet_jacobian(e::Exp, y::Real)
    x = exp(y)
    return ((e.sign * x) + e.bound, y)
end
inverse(e::Exp) = Log(e.bound, e.sign)

"""
   Log(bound, sign) <: ScalarToScalarBijector

Callable struct, defined such that `(l::Log)(x) = log(l.sign * (x - l.bound))`. The sign is
determined by the `sign` field.

This is the appropriate scalar-to-scalar bijector for univariate distributions which have a
support of `[l.bound, ∞)` (if `l.sign == 1`), or `(-∞, l.bound]` (if `l.sign == -1`).

!!! warning
    This does not check whether the input is the domain of the transformation.
"""
struct Log{L<:Real} <: ScalarToScalarBijector
    bound::L
    sign::Int
end
is_monotonically_increasing(l::Log) = l.sign > 0
is_monotonically_decreasing(l::Log) = l.sign < 0
(l::Log)(x::Real) = first(with_logabsdet_jacobian(l, x))
function with_logabsdet_jacobian(l::Log, x::Real)
    logx = log(l.sign * (x - l.bound))
    return (logx, -logx)
end
inverse(l::Log) = Exp(l.bound, l.sign)
