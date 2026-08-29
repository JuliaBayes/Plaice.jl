module PlaicePMUnivariateTests

using ProbabilityMeasures
using Test
using Plaice
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

if !Sys.iswindows()
    import Pkg
    Pkg.add("Reactant")
    using Reactant: Reactant
end

const PM = ProbabilityMeasures

univariates = [
    # Continuous
    PM.Cauchy(), # iden
    PM.Exponential(), # pos
    PM.Laplace(), # iden
    PM.LogNormal(), # pos
    PM.Normal(), # iden
    PM.Uniform(0.0, 1.0), # trunc
    # Discrete
    PM.Bernoulli(0.5),
    PM.Binomial(5, 0.5),
    PM.Categorical([0.2, 0.5, 0.3]),
    PM.Geometric(0.3),
    PM.Poisson(3.0),
]

@testset "PM Univariates" begin
    for d in univariates
        Plaice.test_all(d; expected_zero_allocs=(from_vec, from_unconstrained_vec))
    end
end

end # module PlaicePMUnivariateTests
