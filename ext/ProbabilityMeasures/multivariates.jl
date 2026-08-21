# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# The AbstractMvNormal abstract type takes care of MvNormal and MvNormalCanon.
Plaice.from_unconstrained_vec(::PM.MvNormal) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::PM.MvNormal) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::PM.MvNormal) = length(d)
Plaice.unconstrained_optic_vec(d::PM.MvNormal) = Plaice.optic_vec(d)

# For all multivariate distributions, from_vec and to_vec are just the identity function.
Plaice.from_vec(::PM.MultivariateMeasure) = Plaice.TypedIdentity()
Plaice.to_vec(::PM.MultivariateMeasure) = Plaice.TypedIdentity()
# which makes vec_length and optic_vec trivial
Plaice.vec_length(d::PM.MultivariateMeasure) = length(d)
# TODO(penelopeysm): We assume here that the axes of the distribution are 1:length(d). This
# is not always true, but we don't (yet) have a good way to determine that.
function Plaice.optic_vec(d::PM.MultivariateMeasure)
    return [VarNames.@opticof(_[i]) for i in 1:length(d)]
end

# For discrete multivariate distributions, we really can't transform the 'support'.
#
# todo discrete multivariate Not defined in PM yet
#
# Plaice.from_unconstrained_vec(::PM.DiscreteMultivariateMeasure) = Plaice.TypedIdentity()
# Plaice.to_unconstrained_vec(::PM.DiscreteMultivariateMeasure) = Plaice.TypedIdentity()
# Plaice.unconstrained_vec_length(d::PM.DiscreteMultivariateMeasure) = Plaice.vec_length(d)
# Plaice.unconstrained_optic_vec(d::PM.DiscreteMultivariateMeasure) = Plaice.optic_vec(d)
