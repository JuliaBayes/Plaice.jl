# Unreleased

Use centered stick breaking for joint order statistics with finite interval bounds. This changes their unconstrained coordinates. To migrate saved coordinates, decode them using `from_unconstrained_vec(d)` in the earlier Plaice version. Encode the constrained points using `to_unconstrained_vec(d)` in the new version. Finite precision can still collapse points close to each other or a support endpoint.

# 0.1.4

Declare and document `ProductVecTransform` and `Elementwise` as public API for downstream product bijector implementations.

Fix a bug with Enzyme on named product distribution bijectors.

# 0.1.3

Improve numerical stability of `Truncate` and `Untruncate`.

# 0.1.2

(Substantially) improve the performance of matrix and Cholesky bijectors.

# 0.1.1

Added a workaround for a Julia 1.13 bug which caused `test_allocations` to crash depending on the value of `expected_zero_allocs`.

# 0.1.0

Initial release.
