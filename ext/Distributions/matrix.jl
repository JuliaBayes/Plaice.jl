VB.to_vec(d::D.MatrixDistribution) = VB.Vec(size(d))
VB.from_vec(d::D.MatrixDistribution) = VB.Reshape(size(d))
VB.vec_length(d::D.MatrixDistribution) = prod(size(d))
function VB.optic_vec(d::D.MatrixDistribution)
    return map(c -> VarNames.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end

# MatrixNormal and MatrixTDist are trivial since all their components are already
# unconstrained.

const UnconsMatrixDist = Union{D.MatrixNormal,D.MatrixTDist}

VB.to_unconstrained_vec(d::UnconsMatrixDist) = VB.Vec(size(d))
VB.from_unconstrained_vec(d::UnconsMatrixDist) = VB.Reshape(size(d))
VB.unconstrained_vec_length(d::UnconsMatrixDist) = prod(size(d))
function VB.unconstrained_optic_vec(d::UnconsMatrixDist)
    return map(c -> VarNames.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end

# TODO(penelopeysm): MatrixBeta also generates positive definite matrices. However, it is
# even more specific than Wishart/InverseWishart in that it generates positive definite
# matrices `M` such that `I - M` is also positive definite. This means that the
# transformation implemented here is not suitable for MatrixBeta, as
# from_unconstrained_vec(d)(randn(...)) may not be in the support of MatrixBeta, and thus sampling
# from a unconstrained vector with e.g. NUTS may fail. Hence, we do not include MatrixBeta here.
const PDMatrixDistribution = Union{D.Wishart,D.InverseWishart}

VB.from_unconstrained_vec(d::PDMatrixDistribution) = VB.InvPosDef(first(size(d)))
VB.to_unconstrained_vec(d::PDMatrixDistribution) = VB.PosDef(first(size(d)))
function VB.unconstrained_vec_length(d::PDMatrixDistribution)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
VB.unconstrained_optic_vec(d::PDMatrixDistribution) = fill(nothing, VB.unconstrained_vec_length(d))

# LKJ correlation matrices.
#
# TODO(penelopeysm) VecCorrBijector has a few bugs. Look at the issue tracker.

VB.from_unconstrained_vec(::D.LKJ) = VB.inverse(VB.VecCorrBijector())
VB.to_unconstrained_vec(::D.LKJ) = VB.VecCorrBijector()
function VB.unconstrained_vec_length(d::D.LKJ)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
VB.unconstrained_optic_vec(d::D.LKJ) = fill(nothing, VB.unconstrained_vec_length(d))
