module PlaiceMatrixTests

using Distributions
using LinearAlgebra
using Test
using PDMats
import DifferentiationInterface as DI
using Plaice
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake

ν = 5
M = [1 2 3; 4 5 6]
Σ = PDMats.PDMat([1 0.5; 0.5 1])
Ω = PDMats.PDMat([1 0.3 0.2; 0.3 1 0.4; 0.2 0.4 1])

const adtypes = [
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const),
    DI.AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const),
]

# Don't check that from_unconstrained_vec(d)(randn(...)) is in support for LKJ,
# The reason is because the inverse bijector for LKJ causes the diagonal
# entries to be not exactly 1 due to numerical precision issues. This
# should in principle be fixed, but for now we skip the test.
# https://github.com/TuringLang/Bijectors.jl/issues/435
test_in_support(d) = !(d isa LKJ)

matrix_dists = [
    MatrixNormal(2, 4),
    MatrixNormal(3, 5),
    MatrixTDist(ν, M, Σ, Ω),
    Wishart(7, Matrix{Float64}(I, 2, 2)),
    Wishart(7, Matrix{Float64}(I, 4, 4)),
    InverseWishart(7, Matrix{Float64}(I, 2, 2)),
    InverseWishart(7, Matrix{Float64}(I, 4, 4)),
    LKJ(3, 1.0),
    LKJ(7, 1.0),
]

@testset "Matrix distributions" begin
    @testset "Positive definite coordinates" begin
        d = Wishart(5, Matrix{Float64}(I, 3, 3))
        x = [4 2 0; 2 10 3; 0 3 5]
        y = [log(2), 1, log(3), 0, 1, log(2)]
        logjac = 9log(2) + 3log(3)
        result, result_logjac = with_logabsdet_jacobian(to_unconstrained_vec(d), x)
        @test result ≈ y
        @test logabsdet_jacobian(to_unconstrained_vec(d), x) ≈ result_logjac ≈ -logjac
        @test logabsdet_jacobian(from_unconstrained_vec(d), y) ≈ logjac
    end

    @testset "Correlation inverse Jacobian at large coordinates" begin
        f = from_unconstrained_vec(LKJ(3, 1.0))
        y = [800.0, 0.0, 0.0]
        logjac = 3 * (log(2) - 800)
        @test logabsdet_jacobian(f, y) ≈ logjac
        @test last(with_logabsdet_jacobian(f, y)) ≈ logjac
    end

    for d in matrix_dists
        kwargs = (expected_zero_allocs=(), test_in_support=test_in_support(d))
        Plaice.test_all(d; adtypes=adtypes, kwargs...)
    end
end

end # module PlaiceMatrixTests
