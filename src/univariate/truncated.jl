# Bijectors for continuous univariate distributions which have support over arbitrary ranges
# `(a, b)`. In general, this file has to handle possibly infinite values. The reason for
# this is because of type stability: we can't, for example, return to a different bijector
# type in case we find that the bounds are infinite (which can only be determined at
# runtime). So there is some element of code repetition here: the case where both bounds are
# infinite is the same as `TypedIdentity`, and the case where only the upper bound is
# infinite is the same as `Exp` and `Log`.

using LogExpFunctions: logistic, log1pexp

"""
    Truncate(a, b) <: ScalarToScalarBijector

Callable struct, defined such that `(::Truncate(a, b))(x)` converts `x` from `(-Inf, Inf)`
to `(a, b)`.
"""
struct Truncate{L<:Number,U<:Number} <: ScalarToScalarBijector
    lower::L
    upper::U
end
is_monotonically_increasing(t::Truncate) = !is_monotonically_decreasing(t)
is_monotonically_decreasing(t::Truncate) = !isfinite(t.lower) && isfinite(t.upper)
function (t::Truncate)(y::Number)
    lbounded, ubounded = isfinite(t.lower), isfinite(t.upper)
    return if lbounded && ubounded
        ((t.upper - t.lower) * logistic(y)) + t.lower
    elseif lbounded
        exp(y) + t.lower
    elseif ubounded
        t.upper - exp(y)
    else
        y
    end
end
function with_logabsdet_jacobian(t::Truncate, y::Number)
    lbounded, ubounded = isfinite(t.lower), isfinite(t.upper)
    return if lbounded && ubounded
        bma = t.upper - t.lower
        res = (bma * logistic(y)) + t.lower
        # This is the same as `log(bma) + y - (2 * log1pexp(y))`, but that form gives
        # `Inf - Inf = NaN` at `y = Inf`, whereas this one correctly gives `-Inf`. It is
        # also slightly more accurate for large positive `y`.
        absy = abs(y)
        logjac = log(bma) - absy - (2 * log1pexp(-absy))
        res, logjac
    elseif lbounded
        exp(y) + t.lower, y
    elseif ubounded
        t.upper - exp(y), y
    else
        y, zero(y)
    end
end
inverse(t::Truncate) = Untruncate(t.lower, t.upper)

"""
   Untruncate(a, b) <: ScalarToScalarBijector

Callable struct, defined such that `(::Untruncate(a, b))(x)` maps a scalar `x` from `(a, b)`
to `(-Inf, Inf)`.

This is the appropriate scalar-to-scalar bijector for distributions which have support over
`(a, b)`.

!!! warning
    This does not check whether the input is a scalar in `(a, b)`.
"""
struct Untruncate{L<:Number,U<:Number} <: ScalarToScalarBijector
    lower::L
    upper::U
end
is_monotonically_increasing(t::Untruncate) = !is_monotonically_decreasing(t)
is_monotonically_decreasing(t::Untruncate) = !isfinite(t.lower) && isfinite(t.upper)
function (u::Untruncate)(x::Number)
    lbounded, ubounded = isfinite(u.lower), isfinite(u.upper)
    return if lbounded && ubounded
        log(x - u.lower) - log(u.upper - x)
    elseif lbounded
        log(x - u.lower)
    elseif ubounded
        log(u.upper - x)
    else
        x
    end
end
function with_logabsdet_jacobian(u::Untruncate, x::Number)
    lbounded, ubounded = isfinite(u.lower), isfinite(u.upper)
    # This conditional needs some care: if we change the structure to `if lbounded &&
    # ubounded ...` then it runs into https://github.com/EnzymeAD/Enzyme.jl/issues/3679
    # on 1.10.
    # 
    # That's a failure with Enzyme.jacobian rather than Enzyme.gradient, so it's unlikely
    # that it will really be hit in practice. But it's still worth being careful here
    return if lbounded
        log_xma = log(x - u.lower)
        if ubounded
            # We could compute `logit((x - a) / (b - a))`, but when `x` is close to `b`,
            # the rounding error in `(x - a) / (b - a)` gets magnified when `logit`
            # computes `1 - (x - a) / (b - a)`. Calculating `b - x` directly avoids this.
            log_bmx = log(u.upper - x)
            log_xma - log_bmx, log(u.upper - u.lower) - log_xma - log_bmx
        else
            log_xma, -log_xma
        end
    elseif ubounded
        log_bmx = log(u.upper - x)
        log_bmx, -log_bmx
    else
        x, zero(x)
    end
end
inverse(u::Untruncate) = Truncate(u.lower, u.upper)
