using SIR_pkg
using Test

@testset "SIR_pkg.jl" begin
    # Write your tests here.
    @testset "time_cdf" begin
        @test probability_to_rate(10,0.5) ≈ 0.06931471805599453
        @test probability_to_rate(1,1) ≈ Inf
        @test probability_to_rate(1,0) ≈ 0
    end

    @testset "Constant Pop in simple model" begin
        sol = simulate_model(4999,1,0, 0.1,0.03, [0,200.0])
        for i in sol.u
            @test i[1] + i[2] + i[3] ≈ 5000
        end
    end

    @testset "R0 and pc" begin
        @test get_R0(1,1,1) ≈ 1
        @test get_R0(10, 0.05, 0.1) ≈ 5
        @test get_pc(1,1,1) ≈ 0
        @test get_pc(10, 0.05, 0.1) ≈ 0.8
    end

    @testset "Force of infection model" begin
        S, I, R = 4999, 1, 0
        N = 5000
        c, β, γ = 10, 0.05, 0.1
        sol = simulate_model(S, I, R, c, β, γ, [0,200.0])
        #test constant population
        for i in sol.u
            @test i[1] + i[2] + i[3] ≈ N
        end
        pc = get_pc(c, β, γ)
        prev = [0,0,0]
        for i in sol.u
            #Test that infection grows before pc
            if (prev[3]+prev[2]) > pc*N
                @test i[2] <= prev[2]
            end
            #test infection shrinks after pc
            if (i[3]+i[2]) < pc*N
                @test i[2] >= prev[2]
            end
            prev = i
        end
    end

    sol = simulate_model(5,1,0,0,0,[0,200.])
    error_test_data = [0,1,2,3]
    error_test_data_time = [1,2,3,4]
    @test SIR_error(sol, error_test_data, error_test_data_time, 2) ≈ 6/4

    @testset "Serious Infection" begin
        S, I, Is, R = 4999, 1, 0, 0
        c, β, γ = 10, 0.05, 0.1
        ps, γs = 0.2, 0.1

    end
    @testset "Intervention model" begin
        #intervention after 3 days
        sol = simulate_model(1000,1,0,0,1,1,1,1,1,0,1,1,3,[0,30.])
        for i in range(1,length(sol.u)-1)
            if sol.t[i] > 3
                @test sol.u[i][1] ≈ sol.u[i+1][1]
            end
        end
    end
end
