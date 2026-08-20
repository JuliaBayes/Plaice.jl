# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# The AbstractMvNormal abstract type takes care of MvNormal and MvNormalCanon.
VB.from_linked_vec(::D.AbstractMvNormal) = VB.TypedIdentity()
VB.to_linked_vec(::D.AbstractMvNormal) = VB.TypedIdentity()
VB.linked_vec_length(d::D.AbstractMvNormal) = length(d)
VB.linked_optic_vec(d::D.AbstractMvNormal) = VB.optic_vec(d)

# NOTE: AbstractMvTDist is not formally exported from Distributions, but this is the only
# 'correct' place to put it
VB.from_linked_vec(::D.AbstractMvTDist) = VB.TypedIdentity()
VB.to_linked_vec(::D.AbstractMvTDist) = VB.TypedIdentity()
VB.linked_vec_length(d::D.AbstractMvTDist) = length(d)
VB.linked_optic_vec(d::D.AbstractMvTDist) = VB.optic_vec(d)

# For all multivariate distributions, from_vec and to_vec are just the identity function.
VB.from_vec(::D.MultivariateDistribution) = VB.TypedIdentity()
VB.to_vec(::D.MultivariateDistribution) = VB.TypedIdentity()
# which makes vec_length and optic_vec trivial
VB.vec_length(d::D.MultivariateDistribution) = length(d)
# TODO(penelopeysm): We assume here that the axes of the distribution are 1:length(d). This
# is not always true, but we don't (yet) have a good way to determine that... If you're
# reading this, check for updates in:
# https://github.com/JuliaStats/Distributions.jl/issues/734
# https://github.com/JuliaStats/Distributions.jl/pull/2009
function VB.optic_vec(d::D.MultivariateDistribution)
    return [VarNames.@opticof(_[i]) for i in 1:length(d)]
end

# For discrete multivariate distributions, we really can't transform the 'support'.
VB.from_linked_vec(::D.DiscreteMultivariateDistribution) = VB.TypedIdentity()
VB.to_linked_vec(::D.DiscreteMultivariateDistribution) = VB.TypedIdentity()
VB.linked_vec_length(d::D.DiscreteMultivariateDistribution) = VB.vec_length(d)
VB.linked_optic_vec(d::D.DiscreteMultivariateDistribution) = VB.optic_vec(d)

# MvLogNormal
VB.from_linked_vec(::D.AbstractMvLogNormal) = VB.MapExp()
VB.to_linked_vec(::D.AbstractMvLogNormal) = VB.MapLog()
VB.linked_vec_length(d::D.AbstractMvLogNormal) = length(d)
VB.linked_optic_vec(d::D.AbstractMvLogNormal) = VB.optic_vec(d)

# Simplex distributions
const SIMPLEX_MULTIVARIATES = Union{D.Dirichlet,D.MvLogitNormal}
VB.Plaice.from_linked_vec(::SIMPLEX_MULTIVARIATES) = VB.inverse(VB.SimplexBijector())
VB.Plaice.to_linked_vec(::SIMPLEX_MULTIVARIATES) = VB.SimplexBijector()
VB.Plaice.linked_vec_length(d::SIMPLEX_MULTIVARIATES) = length(d) - 1
function VB.linked_optic_vec(d::SIMPLEX_MULTIVARIATES)
    return fill(nothing, VB.linked_vec_length(d))
end
