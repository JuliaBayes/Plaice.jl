module VBTransformedTests

using Distributions
using LinearAlgebra
using Test
using VectorBijectors
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

transformed_dists = [
    # Univariate
    transformed(Normal(), exp),
    transformed(Gamma(2, 1), elementwise(log)),
    # Multivariate
    transformed(product_distribution(fill(Beta(2, 2), 4)), elementwise(exp)),
    transformed(Dirichlet([1.0, 2.0, 3.0])),
    transformed(MvLogNormal(zeros(2), I), elementwise(log)),
    # Matrix
    transformed(MatrixNormal(zeros(2, 3), I(2), I(3)), elementwise(exp)),
]

@testset "TransformedDistributions" begin
    for d in transformed_dists
        VectorBijectors.test_all(d; test_in_support=false)
    end
end

end # module VBTransformedTests
