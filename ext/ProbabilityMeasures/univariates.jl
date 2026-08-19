using VectorBijectors: TypedIdentity, Log
using Test

VB.from_linked_vec(d::PM.ContinuousUnivariateMeasure) =
    VB.OnlyWrap(VB.inverse(VB.scalar_to_scalar_bijector(d)))
VB.to_linked_vec(d::PM.ContinuousUnivariateMeasure) =
    VB.VectWrap(VB.scalar_to_scalar_bijector(d))
VB.from_vec(::PM.ContinuousUnivariateMeasure) = VB.OnlyWrap(VB.TypedIdentity())
VB.to_vec(::PM.ContinuousUnivariateMeasure) = VB.VectWrap(VB.TypedIdentity())

VB.vec_length(::PM.ContinuousUnivariateMeasure) = 1
VB.linked_vec_length(::PM.ContinuousUnivariateMeasure) = 1

VB.optic_vec(::PM.ContinuousUnivariateMeasure) = [VarNames.Iden()]
VB.linked_optic_vec(::PM.ContinuousUnivariateMeasure) = [VarNames.Iden()]

# Distributions with support over the entire real line.
VB.scalar_to_scalar_bijector(::PM.Normal) = VB.TypedIdentity()

# Distributions with support over the non-negative reals.
VB.scalar_to_scalar_bijector(::PM.Exponential) = VB.Log(0.0, 1)

# Everything else
function VB.scalar_to_scalar_bijector(d::PM.ContinuousUnivariateMeasure)
    return VB.Untruncate(minimum(PM.support(d)), maximum(PM.support(d)))
end

VB.can_test_in_support(d::PM.ContinuousUnivariateMeasure, x) = true
function VB.test_in_support(d::PM.ContinuousUnivariateMeasure, x)
    @test PM.insupport(d, x)
end
