module PlaiceOrderTests

using Distributions
using LinearAlgebra
using Test
using Plaice
import DifferentiationInterface as DI
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

base_dists = [
    Normal(),
    InverseGamma(2, 3),
    InverseGamma(2, 3) * -2,
    Beta(2, 2),
    truncated(Normal(); lower=0),
    DiscreteUniform(10),
]

# TODO(penelopeysm): ReverseDiff can't differentiate through JointOrderStatistics transform
# because of the heavy setindex! usage.
# https://github.com/JuliaDiff/ReverseDiff.jl/issues/43 We just avoid testing it for now.
joint_test_adtypes = [
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=Enzyme.Forward),
    DI.AutoEnzyme(; mode=Enzyme.Reverse),
]

@testset "Order statistics" begin
    for d in base_dists
        unvec_only = (from_vec, from_unconstrained_vec)
        Plaice.test_all(OrderStatistic(d, 10, 1); expected_zero_allocs=unvec_only)
        Plaice.test_all(OrderStatistic(d, 10, 10); expected_zero_allocs=unvec_only)
        # JointOrderStatistics is only defined for continuous distributions (technically, it
        # *should* work for discrete distributions whose support is some set which has a
        # total order, but Distributions.jl doesn't actually implement that).
        if d isa ContinuousUnivariateDistribution
            # In the constrained case, the transform is identity.
            #
            # See https://github.com/TuringLang/Bijectors.jl/issues/441 for details about
            # the unusually large atol.
            constrained_only = (from_vec, to_vec)
            Plaice.test_all(
                JointOrderStatistics(d, 4);
                expected_zero_allocs=constrained_only,
                adtypes=joint_test_adtypes,
                roundtrip_atol=1e-1,
            )
            Plaice.test_all(
                JointOrderStatistics(d, 10, 2:5);
                expected_zero_allocs=constrained_only,
                adtypes=joint_test_adtypes,
                roundtrip_atol=1e-1,
            )
        end
    end
end

@testset "Bounded joint order gaps" begin
    @testset "Beta tail roundtrip" begin
        for T in (Float32, Float64)
            d = JointOrderStatistics(Beta(T(2), T(2)), 2)
            y = T[0, 4]
            x, lj = with_logabsdet_jacobian(from_unconstrained_vec(d), y)
            y_recon, lj_recon = with_logabsdet_jacobian(to_unconstrained_vec(d), x)
            @test eltype(x) === eltype(y_recon) === typeof(lj) === typeof(lj_recon) === T
            @test y_recon ≈ y
        end
    end

    @testset "Long interior initialization" begin
        d = JointOrderStatistics(Beta(2, 2), 64)
        x = from_unconstrained_vec(d)(zeros(64))
        @test all(diff([0; x; 1]) .> 0)
    end

    @testset "Affine support Jacobian" begin
        d = JointOrderStatistics(Uniform(1, 5), 3)
        x, lj = with_logabsdet_jacobian(from_unconstrained_vec(d), zeros(3))
        _, lj_recon = with_logabsdet_jacobian(to_unconstrained_vec(d), x)
        @test x ≈ [2, 3, 4]
        # The triangular entries at zero are 3/4, 2/3 and 1/2.
        @test lj ≈ -log(4)
        @test lj_recon ≈ log(4)
    end
end

end # module PlaiceOrderTests
