module VectorBijectorsTests

using VectorBijectors
using Test

@testset "VectorBijectors.jl" begin
    # include("distributions/univariate.jl")
    # include("distributions/multivariate.jl")
    # include("distributions/matrix.jl")
    # include("distributions/reshaped.jl")
    # include("distributions/cholesky.jl")
    # include("distributions/order.jl")
    # include("distributions/product.jl")

    include("pm/univariate.jl")
end

end # module VectorBijectorsTests
