module VectorBijectorsReactantExt

using VectorBijectors
using Enzyme
using Reactant
using Test
import DifferentiationInterface as DI

const VB = VectorBijectors

function VB.test_reactant(d)
    x = VB.rand_safe_ad(d)

    xvec = VB.to_vec(d)(x)
    yvec = VB.to_linked_vec(d)(x)

    ffwd = VB.to_linked_vec(d) ∘ VB.from_vec(d)
    frvs = VB.to_vec(d) ∘ VB.from_linked_vec(d)

    xvec_r = Reactant.to_rarray(xvec)
    yvec_r = Reactant.to_rarray(yvec)

    @testset "Reactant: $(VB.test_name(d))" begin
        @testset "correctness: with_logabsdet_jacobian forward" begin
            expected_y, expected_ladj = VB.with_logabsdet_jacobian(ffwd, xvec)
            result = @allowscalar @jit ffwd(xvec_r)
            @test Array(result) ≈ expected_y
            result_and_lj = @allowscalar @jit VB.with_logabsdet_jacobian(ffwd, xvec_r)
            @test Array(result_and_lj[1]) ≈ expected_y
            @test Float64(result_and_lj[2]) ≈ expected_ladj
        end

        @testset "correctness: with_logabsdet_jacobian reverse" begin
            expected_x, expected_ladj = VB.with_logabsdet_jacobian(frvs, yvec)
            result = @allowscalar @jit frvs(yvec_r)
            @test Array(result) ≈ expected_x
            result_and_lj = @allowscalar @jit VB.with_logabsdet_jacobian(frvs, yvec_r)
            @test Array(result_and_lj[1]) ≈ expected_x
            @test Float64(result_and_lj[2]) ≈ expected_ladj
        end

        @testset "gradient: forward ladj" begin
            ladj_fwd(v) = last(VB.with_logabsdet_jacobian(ffwd, v))
            ref_grad = DI.gradient(ladj_fwd, VB.ref_adtype, xvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, ladj_fwd, xvec_r)
            # result is a 1-elem tuple containing the Reactant array
            @test Array(only(result)) ≈ ref_grad
        end

        @testset "gradient: reverse ladj" begin
            ladj_rvs(v) = last(VB.with_logabsdet_jacobian(frvs, v))
            ref_grad = DI.gradient(ladj_rvs, VB.ref_adtype, yvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, ladj_rvs, yvec_r)
            # result is a 1-elem tuple containing the Reactant array
            @test Array(only(result)) ≈ ref_grad
        end
    end
end

end # module VectorBijectorsReactantExt
