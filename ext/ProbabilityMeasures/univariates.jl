using Plaice: TypedIdentity, Log
using Test

Plaice.from_unconstrained_vec(d::PM.ContinuousUnivariateMeasure) =
    Plaice.OnlyWrap(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d)))
Plaice.to_unconstrained_vec(d::PM.ContinuousUnivariateMeasure) =
    Plaice.VectWrap(Plaice.scalar_to_scalar_bijector(d))
Plaice.from_vec(::PM.ContinuousUnivariateMeasure) = Plaice.OnlyWrap(Plaice.TypedIdentity())
Plaice.to_vec(::PM.ContinuousUnivariateMeasure) = Plaice.VectWrap(Plaice.TypedIdentity())

Plaice.vec_length(::PM.ContinuousUnivariateMeasure) = 1
Plaice.unconstrained_vec_length(::PM.ContinuousUnivariateMeasure) = 1

Plaice.optic_vec(::PM.ContinuousUnivariateMeasure) = [VarNames.Iden()]
Plaice.unconstrained_optic_vec(::PM.ContinuousUnivariateMeasure) = [VarNames.Iden()]

# Distributions with support over the entire real line.
Plaice.scalar_to_scalar_bijector(::PM.Normal) = Plaice.TypedIdentity()

# Distributions with support over the non-negative reals.
Plaice.scalar_to_scalar_bijector(::PM.Exponential) = Plaice.Log(0.0, 1)

# Everything else
function Plaice.scalar_to_scalar_bijector(d::PM.ContinuousUnivariateMeasure)
    return Plaice.Untruncate(minimum(PM.support(d)), maximum(PM.support(d)))
end

Plaice.can_test_in_support(d::PM.ContinuousUnivariateMeasure, x) = true
function Plaice.test_in_support(d::PM.ContinuousUnivariateMeasure, x)
    @test PM.insupport(d, x)
end
