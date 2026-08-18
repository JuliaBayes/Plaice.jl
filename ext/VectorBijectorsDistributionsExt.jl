module VectorBijectorsDistributionsExt

using VectorBijectors
using Distributions
using VarNames

const VB = VectorBijectors
const D = Distributions

include("Distributions/test_utils.jl")
include("Distributions/univariates.jl")
include("Distributions/multivariates.jl")

end # module VectorBijectorsDistributionsExt
