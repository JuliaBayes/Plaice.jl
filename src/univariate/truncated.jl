# Bijectors for continuous univariate distributions which have support over arbitrary ranges
# `(a, b)`. In general, this file has to handle possibly infinite values. The reason for
# this is because of type stability: we can't, for example, return to a different bijector
# type in case we find that the bounds are infinite (which can only be determined at
# runtime). So there is some element of code repetition here: the case where both bounds are
# infinite is the same as `TypedIdentity`, and the case where only the upper bound is
# infinite is the same as `Exp` and `Log`.
#
# The bounds may also be traced numbers when compiling with Reactant (e.g. if they depend on
# another parameter, as in `Uniform(0, 1 - alpha)`). Then `isfinite` returns a traced
# boolean, which can't be used in a normal `if`. `@trace if` compiles to a conditional in
# that case, and is a normal `if` otherwise. For the same reason, conditions use `&` rather
# than `&&`.

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
    @trace if lbounded & ubounded
        res = ((t.upper - t.lower) * logistic(y)) + t.lower
    elseif lbounded
        res = exp(y) + t.lower
    elseif ubounded
        res = t.upper - exp(y)
    else
        res = y
    end
    return res
end
function with_logabsdet_jacobian(t::Truncate, y::Number)
    lbounded, ubounded = isfinite(t.lower), isfinite(t.upper)
    @trace if lbounded & ubounded
        bma = t.upper - t.lower
        res = (bma * logistic(y)) + t.lower
        # This is the same as `log(bma) + y - (2 * log1pexp(y))`, but that form gives
        # `Inf - Inf = NaN` at `y = Inf`, whereas this one correctly gives `-Inf`. It is
        # also slightly more accurate for large positive `y`.
        absy = abs(y)
        logjac = log(bma) - absy - (2 * log1pexp(-absy))
    elseif lbounded
        res = exp(y) + t.lower
        logjac = y
    elseif ubounded
        res = t.upper - exp(y)
        logjac = y
    else
        res = y
        logjac = zero(y)
    end
    return res, logjac
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
    @trace if lbounded & ubounded
        res = log(x - u.lower) - log(u.upper - x)
    elseif lbounded
        res = log(x - u.lower)
    elseif ubounded
        res = log(u.upper - x)
    else
        res = x
    end
    return res
end
function with_logabsdet_jacobian(u::Untruncate, x::Number)
    lbounded, ubounded = isfinite(u.lower), isfinite(u.upper)
    # This conditional needs some care: if we change the structure to `if lbounded &&
    # ubounded ...` then it runs into https://github.com/EnzymeAD/Enzyme.jl/issues/3679
    # on 1.10.
    #
    # That's a failure with Enzyme.jacobian rather than Enzyme.gradient, so it's unlikely
    # that it will really be hit in practice. But it's still worth being careful here
    @trace if lbounded
        log_xma = log(x - u.lower)
        @trace if ubounded
            # We could compute `logit((x - a) / (b - a))`, but when `x` is close to `b`,
            # the rounding error in `(x - a) / (b - a)` gets magnified when `logit`
            # computes `1 - (x - a) / (b - a)`. Calculating `b - x` directly avoids this.
            log_bmx = log(u.upper - x)
            res = log_xma - log_bmx
            logjac = log(u.upper - u.lower) - log_xma - log_bmx
        else
            res = log_xma
            logjac = -log_xma
        end
    elseif ubounded
        log_bmx = log(u.upper - x)
        res = log_bmx
        logjac = -log_bmx
    else
        res = x
        logjac = zero(x)
    end
    return res, logjac
end
inverse(u::Untruncate) = Truncate(u.lower, u.upper)
