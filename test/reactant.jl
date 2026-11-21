module PlaiceReactantTests

# The Reactant tests in `test_all` build the bijectors outside the compiled function, so
# their parameters (e.g. the bounds of `Truncate`) are not traced numbers.
#
# However, inside a probabilistic model, these parameters can depend on other parameters
# (e.g. `x ~ dist; Uniform(0, x)`), in which case they are traced numbers.
#
# This module tests that the bijectors can be compiled when their parameters are traced
# numbers.

using Plaice
using Plaice: Truncate, Untruncate, Exp, Log
using Reactant: Reactant, @jit
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Test

traced(x) = Reactant.to_rarray(x; track_numbers=true)

"""
    test_traced_params(bijector_constructor, params, x)

Check that `bijector_constructor(params...)` gives a bijector that can be compiled when
`params` are traced numbers, and that its output, log-Jacobian, and their gradients with
respect to both the parameters and the input `x` match the non-compiled results.
"""
function test_traced_params(bijector_constructor, params, x)
    # Do NOT extract the call `bijector_constructor(params...)`! That would defeat the
    # whole purpose of the test.
    transform(ps, x) = bijector_constructor(ps...)(x)
    wladj(ps, x) = Plaice.with_logabsdet_jacobian(bijector_constructor(ps...), x)

    @testset "$(bijector_constructor(params...)) at $x" begin
        ps_r = map(traced, params)
        x_r = traced(x)

        @testset "transform" begin
            @test Float64(@jit transform(ps_r, x_r)) ≈ transform(params, x)
        end

        @testset "with_logabsdet_jacobian" begin
            expected_y, expected_ladj = wladj(params, x)
            y, ladj = @jit wladj(ps_r, x_r)
            @test Float64(y) ≈ expected_y
            @test Float64(ladj) ≈ expected_ladj
        end

        @testset "gradient of transform" begin
            # Use sum to collapse to a scalar.
            g(args...) = sum(transform(Base.front(args), last(args)))
            ref_grad = ForwardDiff.gradient(v -> g(v...), [params..., x])
            result = @jit Enzyme.gradient(Enzyme.Reverse, g, ps_r..., x_r)
            @test collect(map(Float64, result)) ≈ ref_grad
        end

        @testset "gradient of ladj wrt params and input" begin
            g(args...) = last(wladj(Base.front(args), last(args)))
            ref_grad = ForwardDiff.gradient(v -> g(v...), [params..., x])
            result = @jit Enzyme.gradient(Enzyme.Reverse, g, ps_r..., x_r)
            @test collect(map(Float64, result)) ≈ ref_grad
        end
    end
end

@testset "Reactant: bijectors with traced parameters" begin
    for (lower, upper) in ((0.0, 1.0), (0.0, Inf), (-Inf, 1.0), (-Inf, Inf))
        y = 0.3
        test_traced_params(Truncate, (lower, upper), y)
        test_traced_params(Untruncate, (lower, upper), Truncate(lower, upper)(y))
    end
    for sign in (1, -1)
        y = 0.3
        test_traced_params(b -> Exp(b, sign), (1.0,), y)
        test_traced_params(b -> Log(b, sign), (1.0,), Exp(1.0, sign)(y))
    end
end

end # module PlaiceReactantTests
