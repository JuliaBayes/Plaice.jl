using Test: @test

Plaice.is_continuous(::PM.ContinuousUnivariateMeasure) = true
Plaice.test_name(d::PM.AbstractProbabilityMeasure) = repr(d)
Plaice.rand_safe_ad(d::PM.AbstractProbabilityMeasure) = rand(d)
