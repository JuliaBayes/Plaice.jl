module PlaiceReactantTests

using Plaice
using Reactant
using Enzyme
using Test

function truncate_columns(X)
    return mapslices(X; dims=1) do x
        @allowscalar begin
            t = Plaice.Truncate(x[1], x[2])
            vcat(t(x[3]), Plaice.with_logabsdet_jacobian(t, x[3])...)
        end
    end
end

function transform_and_logjac(X)
    return sum(truncate_columns(X)[2:3, :])
end

@testset "Truncate with traced bounds" begin
    bounds = [0.0 0.0 -Inf -Inf; 2.0 Inf 3.0 Inf; 0.5 0.5 0.5 0.5]
    inputs = Reactant.to_rarray(bounds)
    s, e = 1 / (1 + exp(-0.5)), exp(0.5)
    expected = [2s e 3-e 0.5; 2s e 3-e 0.5; log(2s * (1-s)) 0.5 0.5 0]
    @test Array(@jit truncate_columns(inputs)) ≈ expected
    gradients = [1-s-0.5 1 0 0; s+0.5 0 1 0; 2s*(1-s)+1-2s e+1 1-e 1]
    gradient = @jit Enzyme.gradient(Enzyme.Reverse, transform_and_logjac, inputs)
    @test Array(only(gradient)) ≈ gradients
end

end # module PlaiceReactantTests
