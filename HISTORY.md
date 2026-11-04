# 0.1.4

Declare and document `ProductVecTransform`, `Elementwise`, and `append_index` as public API for downstream product bijector implementations.
Add Compat for public declarations while keeping Julia 1.10 support.
Fix tuple indices in `append_index` to match its documented behavior.

# 0.1.3

Improve numerical stability of `Truncate` and `Untruncate`.

# 0.1.2

(Substantially) improve the performance of matrix and Cholesky bijectors.

# 0.1.1

Added a workaround for a Julia 1.13 bug which caused `test_allocations` to crash depending on the value of `expected_zero_allocs`.

# 0.1.0

Initial release.
