using Test: @test

VB.is_continuous(::PM.ContinuousUnivariateMeasure) = true
VB.test_name(d::PM.AbstractProbabilityMeasure) = repr(d)
VB.rand_safe_ad(d::PM.AbstractProbabilityMeasure) = rand(d)
