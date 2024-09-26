using SIR_pkg
using Test

@testset "SIR_pkg.jl" begin
    # Write your tests here.
    @test time_cdf_gamma(10,0.5) == 0.06931471805599453
    @test time_cdf_gamma(1,1) == Inf
    @test time_cdf_gamma(1,0) == 0

end
