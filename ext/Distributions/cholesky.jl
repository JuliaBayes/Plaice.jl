function VB.optic_vec(d::D.LKJCholesky)
    n = first(size(d))
    sym = if d.uplo == 'U'
        :U
    else
        :L
    end
    return [
        VarNames.@opticof(_.$sym[i, j]) for (i, j) in VB._get_cartesian_indices(n, d.uplo)
    ]
end

VB.from_vec(d::D.LKJCholesky) = VB.CholeskyUnVec(first(size(d)), d.uplo)
VB.to_vec(d::D.LKJCholesky) = VB.CholeskyVec(first(size(d)), d.uplo)
function VB.vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
VB.from_linked_vec(d::D.LKJCholesky) = VB.inverse(VB.VecCholeskyBijector(d.uplo))
VB.to_linked_vec(d::D.LKJCholesky) = VB.VecCholeskyBijector(d.uplo)
function VB.linked_vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
function VB.linked_optic_vec(d::D.LKJCholesky)
    return fill(nothing, VB.linked_vec_length(d))
end
