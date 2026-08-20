module PlaiceProbabilityMeasuresExt

using Plaice
using ProbabilityMeasures
using VarNames

const VB = Plaice
const PM = ProbabilityMeasures

include("ProbabilityMeasures/univariates.jl")
include("ProbabilityMeasures/test_utils.jl")

end # module PlaiceProbabilityMeasuresExt
