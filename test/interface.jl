module PlaiceInterfaceTests

using Plaice
using Test
using VarNames: @opticof

struct TestExp <: Plaice.AbstractBijector end
struct TestLog <: Plaice.AbstractBijector end

function Plaice.with_logabsdet_jacobian(::TestExp, x)
    return exp(x), x
end
function Plaice.with_logabsdet_jacobian(::TestLog, x)
    y = log(x)
    return y, -y
end

Plaice.inverse(::TestExp) = TestLog()
Plaice.inverse(::TestLog) = TestExp()

@testset "AbstractBijector interface" begin
    b = TestExp()
    x = 1.5

    @test b(x) == exp(x)
    @test Plaice.logabsdet_jacobian(b, x) == x
    @test Plaice.with_logabsdet_jacobian(b, x) == (exp(x), x)

    ib = inverse(b)
    @test ib isa TestLog
    @test inverse(ib) isa TestExp
    @test ib(b(x)) ≈ x
end

@testset "append tuple indices" begin
    A = reshape(1:6, 2, 3)
    B = reshape(1:24, 4, 2, 3)
    @test Plaice.append_index(@opticof(_), (1, 2))(A) == A[1, 2]
    @test Plaice.append_index(@opticof(_.a), (1, 2))((; a=A)) == A[1, 2]
    @test Plaice.append_index(@opticof(_[3]), (1, 2))(B) == B[3, 1, 2]
end

end # module PlaiceInterfaceTests
