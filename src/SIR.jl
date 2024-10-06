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

"""
    get_R0(c, β, γ)

Gets value for R0 from SIR parameters
"""
function get_R0(c, β, γ)
    return c*β/γ
end

"""
    get_pc(c, β, γ)

Gets value for pc from SIR parameters
"""
function get_pc(c, β, γ)
    return 1 - 1/get_R0(c, β, γ)
end


"""
    probability_to_rate(t, p)

Converts probability to recover after a given time into a rate of recovery

# Arguments
- `t::Float64`: Time of recovery
- `p::Float64`: Probability of recovery before time t
"""
function probability_to_rate(t::Float64, p::Float64)
    #p = 1-e^-yt
    #1 - p = e^-yt
    #ln(1-p) = -yt
    γ = -log(1-p)/t
    return γ
end

"""
    sir_model_simple(dpop, pop, params, t)

Basic SIR infection model

# Arguments
- `pop::Vector{}`: Initial Susceptible, Infected, Recovered populations
- `params::Vector{}`: SIR model parameters λ, γ
"""
function sir_model_simple(dpop, pop::Vector{}, params::Vector{}, t)
    λ, γ = params
    S, I, R = pop
    N = S + I + R
    dpop[1] = - λ * S
    dpop[2] = λ * S - γ * I
    dpop[3] = γ * I
end

"""
    sir_model(dpop, pop, params, t)

SIR infection model with true force of infection

# Arguments
- `pop::Vector{}`: Initial Susceptible, Infected, Recovered populations
- `params::Vector{}`: SIR model parameters c, β, γ
"""
function sir_model(dpop, pop, params::Vector{}, t)
    S, I, R = pop
    c, β, γ = params
    N = S + I + R
    λ = I * β * c / N
    dpop[1] = - S * λ
    dpop[2] = λ * S - γ * I
    dpop[3] = γ * I
end

"""
    simulate_model(S, I, R, λ, γ, tspan::Vector{})

Simulate simple SIR model

# Arguments
- `S`: Initial Susceptible population
- `I`: Initial Infected population
- `R`: Initial Recovered population
- `λ`: Rate of infection
- `γ`: Rate of recovery
- `tspan::Vector{}`: Timespan to model over
"""
function simulate_model(S, I, R, λ, γ, tspan::Vector{})
    pop0 = [S, I, R]
    params = [λ, γ]
    model = ODEProblem(sir_model_simple, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

"""
    simulate_model(S, I, R, λ, γ, tspan::Vector{})

Simulate SIR model with force of infection

# Arguments
- `S`: Initial Susceptible population
- `I`: Initial Infected population
- `R`: Initial Recovered population
- `c`: Number of contacts 
- `β`: Probability of infection
- `γ`: Rate of recovery
- `tspan::Vector{}`: Timespan to model over
"""
function simulate_model(S, I, R, c, β, γ, tspan::Vector{})
    pop0 = [S, I, R]
    params = [c, β, γ]
    model = ODEProblem(sir_model, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

"""
    plot_solution(sol::ODESolution)

Plots solution of SIR model

# Arguments
- `sol::ODESolution`: Solution of SIR model
"""
function plot_solution(sol::ODESolution)
    plot(sol, xlabel="Time", ylabel = "Population",label = ["S" "I" "R"], title = "Solution of SIR")
    plot!(legend=:outerbottom, legendcolumns=3)
end

"""
    plot_solution(sol::ODESolution, data, tspan)

Plots Infected from SIR model against infection data

# Arguments
- `sol::ODESolution`: Solution of SIR model
- `data::Vector{}`: Vector of number of infections over time
- `tspan::Vector{}`: Timespan over which to graph
"""
function plot_solution(sol::ODESolution, data::Vector{}, tspan::Vector{})
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="pop",legend=false)
    plot!(sol.t,sol[2,:])
    plot!(data, seriestype=:scatter, label="data")
    xlims!(tspan[1],tspan[2])
    ylims!(0,1.2*maximum(data))
end


"""
    error(sol::ODESolution, data::Vector{})

Calculates least squares error of SIR solution compared to data

# Arguments
- `sol::ODESolution`: Solution of SIR model
- `data::Vector{}`: Data over time
"""
function error(sol::ODESolution, data::Vector{})
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

"""
    simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, tspan::Vector{})

Simulate SIR model with force of infection

# Arguments
- `S`: Initial Susceptible population
- `I`: Initial Infected population
- `Is`: Initial Seriously Infected population
- `R`: Initial Recovered population
- `c`: Number of contacts 
- `β`: Probability of infection
- `γ`: Rate of recovery
- `ps`: Probability of becoming seriously infected
- `γs`: Rate of recovery from serious infection
- `α`: Rate of becoming Susceptible
- `tspan::Vector{}`: Timespan to model over
"""
function simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, tspan::Vector{})
    pop0 = [S, I, Is, R]
    params = [c, β, γ, ps, γs, α]
    model = ODEProblem(sirs_model, pop0, tspan, params)
    sol = solve(model, saveat=0.2)
    return sol
end

"""
    sir_model_simple(dpop, pop, params, t)

SIRS infection model

# Arguments
- `pop::Vector{}`: Initial Susceptible, Infected, Seriously Infected, Recovered populations
- `params::Vector{}`: SIRS model parameters c, β, γ, ps, γs, α
"""
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

"""
    plot_solution_SIRS(sol::ODESolution)

Plots solution of SIRS model

# Arguments
- `sol::ODESolution`: Solution of SIRS model
"""
function plot_solution_SIRS(sol::ODESolution)
    plot(sol, xlabel="Time", ylabel = "Population",label = ["S" "I" "Is" "R"], title = "Solution of SIR")
    plot!(legend=:outerbottom, legendcolumns=4)
end



