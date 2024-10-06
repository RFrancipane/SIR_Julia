# S = Susceptible
# I = Infected
# R = Recovered
# N = Total pop
# γ = Force of recovery
# λ = Force of infection
# β = Probability of infection
# c = Number of contacts


using Revise
using Plots
using DifferentialEquations


function time_cdf_gamma(t, p)
    #p = 1-e^-yt
    #1 - p = e^-yt
    #ln(1-p) = -yt
    γ = -log(1-p)/t
    return γ
end


function sir_model_simple(dpop, pop, params::Vector{}, t)
    λ, γ = params
    S, I, R = pop
    N = S + I + R
    dpop[1] = - λ * S
    dpop[2] = λ * S - γ * I
    dpop[3] = γ * I
end

function sir_model(dpop, pop, params::Vector{}, t)
    S, I, R = pop
    c, β, γ = params
    N = S + I + R
    λ = I * β * c / N
    dpop[1] = - S * λ
    dpop[2] = λ * S - γ * I
    dpop[3] = γ * I
end

function sirs_model(dpop, pop, params::Vector{}, t)
    S, I, Is, R = pop
    c, β, γ, ps, γs, α = params
    N = S + I + Is + R
    λ = I * β * c / N + Is * β * c / N

    dpop[1] = - λ * S                       + α * R
    dpop[2] =   λ * S     - γ * I
    dpop[3] =         ps  * γ * I - γs * Is
    dpop[4] =      (1-ps) * γ * I + γs * Is - α * R
end

function simulate_model(S, I, R, λ, γ, tspan::Vector{})
    pop0 = [S, I, R]
    params = [λ, γ]
    model = ODEProblem(sir_model_simple, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

function simulate_model(S, I, R, c, β, γ, tspan::Vector{})
    pop0 = [S, I, R]
    params = [c, β, γ]
    model = ODEProblem(sir_model, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

function simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, tspan::Vector{})
    pop0 = [S, I, Is, R]
    params = [c, β, γ, ps, γs, α]
    model = ODEProblem(sirs_model, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

function plot_solution(sol::ODESolution)
    plot(sol, xlabel="Time", ylabel = "Population",label = ["S" "I" "R"], title = "Solution of SIR")
    plot!(legend=:outerbottom, legendcolumns=3)
end

function plot_solution_SIRS(sol::ODESolution)
    plot(sol, xlabel="Time", ylabel = "Population",label = ["S" "I" "Is" "R"], title = "Solution of SIR")
    plot!(legend=:outerbottom, legendcolumns=4)
end

function plot_solution(sol::ODESolution, data, tspan)
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="pop",legend=false)
    plot!(sol.t,sol[2,:])
    plot!(data, seriestype=:scatter, label="data")
    xlims!(tspan[1],tspan[2])
    ylims!(0,1.2*maximum(data))
end

function error(sol, data)
    total = 0
    for x in range(1,length(data))
        for i in range(1,length(sol.t))
            if sol.t[i] ≈ x
                total += (sol.u[i][2]-data[x])^2
            end
        end
    end
    return total
end

#params = (c, β, γ)
#params = [10, 0.03, 0.2]
#pop0 = [5000, 1, 0]
#tspan = [0, 30]




