# OrderStatistic can only ever wrap univariate distributions so these can just delegate to
# the underlying distribution.
VB.to_vec(d::D.OrderStatistic) = VB.to_vec(d.dist)
VB.from_vec(d::D.OrderStatistic) = VB.from_vec(d.dist)
VB.to_unconstrained_vec(d::D.OrderStatistic) = VB.to_unconstrained_vec(d.dist)
VB.from_unconstrained_vec(d::D.OrderStatistic) = VB.from_unconstrained_vec(d.dist)
# We don't need to implement the other methods as OrderStatistic is a subtype of
# UnivariateDistribution, so we can just use the default methods.

# Here, because `d.dist` isa UnivariateDistribution, we can get its scalar-to-scalar
# bijector and then rewrap that inner bijector into a JointOrderWrap to get the desired
# behavior.
VB.to_vec(::D.JointOrderStatistics) = VB.TypedIdentity()
function VB.to_unconstrained_vec(d::D.JointOrderStatistics)
    return VB.JointOrderWrap(VB.scalar_to_scalar_bijector(d.dist))
end
VB.from_vec(::D.JointOrderStatistics) = VB.TypedIdentity()
function VB.from_unconstrained_vec(d::D.JointOrderStatistics)
    return VB.InverseJointOrderWrap(VB.inverse(VB.scalar_to_scalar_bijector(d.dist)))
end
# Since D.JointOrderStatistics is a subtype of MultivariateDistribution, we can use the
# default definitions for vec_length and optic_vec.
VB.unconstrained_vec_length(d::D.JointOrderStatistics) = VB.vec_length(d)
# TODO: Technically, the first element can be @opticof(_[1]) so this is not technically
# correct.
VB.unconstrained_optic_vec(d::D.JointOrderStatistics) = fill(nothing, VB.vec_length(d))

VB._is_joint_order_statistics(::D.JointOrderStatistics) = true
