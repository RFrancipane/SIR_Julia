using SIR_pkg
using Test

@testset "SIR_pkg.jl" begin
    # Write your tests here.
    @test time_cdf_gamma(10,0.5) ≈ 0.06931471805599453
    @test time_cdf_gamma(1,1) == Inf
    @test time_cdf_gamma(1,0) == 0

    @testset "Constant Pop in simple model" begin
        sol = simulate_model([4999,1,0], 0.1,0.03, [0,200.0])
        for i in sol.u
            @test i[1] + i[2] + i[3] ≈ 5000
        end
    end
end
