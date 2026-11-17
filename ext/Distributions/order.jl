# OrderStatistic can only ever wrap univariate distributions so these can just delegate to
# the underlying distribution.
Plaice.to_vec(d::D.OrderStatistic) = Plaice.to_vec(d.dist)
Plaice.from_vec(d::D.OrderStatistic) = Plaice.from_vec(d.dist)
Plaice.to_unconstrained_vec(d::D.OrderStatistic) = Plaice.to_unconstrained_vec(d.dist)
Plaice.from_unconstrained_vec(d::D.OrderStatistic) = Plaice.from_unconstrained_vec(d.dist)
# We don't need to implement the other methods as OrderStatistic is a subtype of
# UnivariateDistribution, so we can just use the default methods.

# Use the base scalar map to choose a chart for the ordered vector.
Plaice.to_vec(::D.JointOrderStatistics) = Plaice.TypedIdentity()
function Plaice.to_unconstrained_vec(d::D.JointOrderStatistics)
    b = Plaice.scalar_to_scalar_bijector(d.dist)
    # Distributions uses Float64 literals for Beta's fixed support.
    if d.dist isa D.Beta &&
       b isa Plaice.Untruncate{Float64,Float64} &&
       b.lower == 0.0 &&
       b.upper == 1.0
        b = Plaice.Untruncate(0, 1)
    end
    return Plaice.joint_order_wrap(b)
end
Plaice.from_vec(::D.JointOrderStatistics) = Plaice.TypedIdentity()
Plaice.from_unconstrained_vec(d::D.JointOrderStatistics) =
    Plaice.inverse(Plaice.to_unconstrained_vec(d))
# Since D.JointOrderStatistics is a subtype of MultivariateDistribution, we can use the
# default definitions for vec_length and optic_vec.
Plaice.unconstrained_vec_length(d::D.JointOrderStatistics) = Plaice.vec_length(d)
# TODO: Technically, the first element can be @opticof(_[1]) so this is not technically
# correct.
Plaice.unconstrained_optic_vec(d::D.JointOrderStatistics) =
    fill(nothing, Plaice.vec_length(d))

Plaice._is_joint_order_statistics(::D.JointOrderStatistics) = true
