module PlaiceCholeskyTests

using Distributions
using LinearAlgebra
using Test
using Plaice
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using Enzyme: Enzyme, set_runtime_activity, Const, Forward, Reverse

# Need runtime activity for some reason.
# TODO(penelopeysm): Report upstream
const adtypes = [
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    DI.AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

dists = [
    # Note: can't test LKJCholesky(1, ...) because its unconstrained vector is length-zero and
    # DifferentiationInterface trips up with empty vectors.
    LKJCholesky(3, 1.0, 'U'),
    LKJCholesky(3, 1.0, 'L'),
    LKJCholesky(5, 1.0, 'U'),
    LKJCholesky(5, 1.0, 'L'),
]

@testset "Cholesky" begin
    @testset "Factor coordinates" begin
        # Unit rows give analytic coordinates and a Jacobian, including zero entries.
        L = Float32[1 0 0 0; 3/5 4/5 0 0; 0 5/13 12/13 0; 0 0 8/17 15/17]
        y = Float32[atanh(3/5), 0, asinh(5/12), 0, 0, asinh(8/15)]
        logjac = -2 * (log(5/4) + log(13/12) + log(17/15))
        for (uplo, factors, xvec) in (
            ('U', Matrix(L'), Float32[1, 3/5, 4/5, 0, 5/13, 12/13, 0, 0, 8/17, 15/17]),
            ('L', L, Float32[1, 3/5, 0, 0, 4/5, 5/13, 0, 12/13, 8/17, 15/17]),
        )
            d = LKJCholesky(4, 1.0, uplo)
            x = Cholesky(factors, uplo, 0)
            @test to_vec(d)(x) ≈ xvec
            @test to_unconstrained_vec(d)(x) ≈ y
            @test from_unconstrained_vec(d)(y).L ≈ L
            @test logabsdet_jacobian(from_unconstrained_vec(d), y) ≈ logjac rtol = 1e-6
        end
    end

    for d in dists
        Plaice.test_all(d; adtypes=adtypes, expected_zero_allocs=())
    end
end

end # module PlaiceCholeskyTests
