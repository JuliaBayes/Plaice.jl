module VectorBijectorsDistributionsExt

using VectorBijectors
using Distributions
using VarNames

const VB = VectorBijectors
const D = Distributions

include("Distributions/test_utils.jl")
include("Distributions/univariates.jl")
include("Distributions/multivariates.jl")
include("Distributions/matrix.jl")
include("Distributions/cholesky.jl")
include("Distributions/reshaped.jl")
include("Distributions/product.jl")
include("Distributions/order.jl")

end # module VectorBijectorsDistributionsExt
