function _cholesky_uplo(d::D.LKJCholesky)
    if d.uplo == 'U'
        return :U
    else
        return :L
    end
end

function Plaice.optic_vec(d::D.LKJCholesky)
    n = first(size(d))
    sym = _cholesky_uplo(d)
    return [
        VarNames.@opticof(_.$sym[i, j]) for
        (i, j) in Plaice._get_cartesian_indices(n, d.uplo)
    ]
end

Plaice.from_vec(d::D.LKJCholesky) = Plaice.CholeskyUnVec(first(size(d)), d.uplo)
Plaice.to_vec(d::D.LKJCholesky) = Plaice.CholeskyVec(first(size(d)), d.uplo)
function Plaice.vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
Plaice.from_unconstrained_vec(d::D.LKJCholesky) =
    Plaice.inverse(Plaice.CorrCholesky(_cholesky_uplo(d)))
Plaice.to_unconstrained_vec(d::D.LKJCholesky) = Plaice.CorrCholesky(_cholesky_uplo(d))
function Plaice.unconstrained_vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
function Plaice.unconstrained_optic_vec(d::D.LKJCholesky)
    return fill(nothing, Plaice.unconstrained_vec_length(d))
end
