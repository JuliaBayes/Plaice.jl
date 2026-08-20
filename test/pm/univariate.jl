module VBPMUnivariateTests

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

univariates = [PM.Normal(), PM.Exponential(), PM.Uniform(0.0, 1.0)]

@testset "PM Univariates" begin
    for d in univariates
        Plaice.test_all(d; expected_zero_allocs=(from_vec, from_unconstrained_vec))
    end
end

end # module VBPMUnivariateTests
