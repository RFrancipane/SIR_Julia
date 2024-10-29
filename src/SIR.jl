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
    get_R0(c, β, γ, ps, γs)

Gets value for R0 from SIRS parameters
"""
function get_R0(c, β, γ, ps, γs)
    R0 = c*β/γ #R0 for SIR
        + ps*c*β/γs #Additional infections if seriously infected
    return R0
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
function probability_to_rate(t, p)
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
- `data_time::Vector{}`: Times of datapoints
- `data_time::Vector{}`: Times of datapoints

"""
function plot_solution(sol::ODESolution, index, data::Vector{}, data_time::Vector{})
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="pop",legend=false)
    plot!(sol.t,sol[index,:])
    plot!(data_time, data, seriestype=:scatter, label="data")
    #xlims!(tspan[1],tspan[2])
    ylims!(0,1.2*maximum(data))
end


"""
    error(sol::ODESolution, data::Vector{})

Calculates least squares error of SIR solution compared to data

# Arguments
- `sol::ODESolution`: Solution of SIR model
- `data::Vector{}`: Data over time
- `index`: SIR index to find error of
"""
function SIR_error(sol::ODESolution, data::Vector{}, data_time::Vector{}, index)
    total = 0
    for x in range(1,length(data))
        for i in range(1,length(sol.t))
            if sol.t[i] ≈ data_time[x]
                total += (sol.u[i][index]-data[x])^2
            end
        end
    end
    return total
end

"""
    simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, tspan::Vector{})

Simulate SIRS model with force of infection

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
    simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, ϵ, Φ, intervention_time, tspan::Vector{})

Simulate SIRS model with force of infection and invervention

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
- `ϵ`: Intervention efficacy
- `Φ`: Proportion following intervention
- `intervention_time`: Time of intervention
- `tspan::Vector{}`: Timespan to model over
"""
function simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, ϵ, Φ, intervention_time, tspan::Vector{})
    pop0 = [S, I, Is, R]
    params = [c, β, γ, ps, γs, α, 0, 0]
    model = ODEProblem(sirs_model_intervention, pop0, tspan, params)
    #Create intervention callback for given time
    function intervention_affect!(integrator)
        integrator.p[7] = ϵ;
        integrator.p[8] = Φ;
    end
    intervention_callback = PresetTimeCallback(intervention_time,intervention_affect!);
    
    sol = solve(model, saveat=0.2, callback = intervention_callback)
    return sol
end



"""
    sirs_model(dpop, pop, params, t)

SIRS infection model

# Arguments
- `pop::Vector{}`: Initial Susceptible, Infected, Seriously Infected, Recovered populations
- `params::Vector{}`: SIRS model parameters c, β, γ, ps, γs, α
"""
function sirs_model(dpop, pop, params::Vector{}, t)
    S, I, Is, R = pop
    c, β, γ, ps, γs, α = params
    N = S + I + Is + R
    λ = I * β * c / N

    dpop[1] = - λ * S                       + α * R
    dpop[2] =   λ * S     - γ * I
    dpop[3] =         ps  * γ * I - γs * Is
    dpop[4] =      (1-ps) * γ * I + γs * Is - α * R
end

"""
    sirs_model_intervention(dpop, pop, params, t)

SIRS infection model with intervention

# Arguments
- `pop::Vector{}`: Initial Susceptible, Infected, Seriously Infected, Recovered populations
- `params::Vector{}`: SIRS model parameters c, β, γ, ps, γs, α
"""
function sirs_model_intervention(dpop, pop, params::Vector{}, t)
    S, I, Is, R = pop
    c, β, γ, ps, γs, α, ϵ, Φ = params
    N = S + I + Is + R
    λ = (1 - ϵ*Φ)*I * β * c / N

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

function general_sim_model(args::Vector{}, tspan::Vector{}) 
    len = size(args, 1)
    if (size == 5) 
        return simulate_model(args[1], args[2], args[3], args[4], args[5], tspan)
    end
    if (size == 6) 
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], tspan)
    end
    if (size == 10)
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], tspan)
    end

    if (size == 13)
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], tspan)
    end
end


function get_error_spread(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_vals)
    error_array = []
    for β in β_vals
        solution = simulate_model(S, I, Is, R, c, β, γ, ps, γs, α, tspan)
        total_error = SIR_error(solution, data, data_time, 2) + SIR_error(solution, data_s, data_time_s, 3)
        push!(error_array, total_error)
    end
    return error_array
end

function plot_error(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_range)
    step = 0.0001
    β_vals = β_range[1]:step:β_range[2]
    error_array = get_error_spread(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_vals)
    plot(β_vals, error_array, xlabel="β", ylabel = "Error", title = "Error vs β")
end

function get_beta_range(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_range, error_diff)
    step = 0.0001
    β_vals = β_range[1]:step:β_range[2]
    error_array = get_error_spread(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_vals)
    min_error = minimum(error_array)
    β_spread = [0,β_vals[argmin(error_array)],0]


    for i in range(1, length(error_array))
        if (error_array[i] <= min_error*(1+error_diff) && β_spread[1] == 0)
            β_spread[1] = β_vals[i]
        end
        if (error_array[i] >= min_error*(1+error_diff) && β_spread[1] != 0)
            β_spread[3] = β_vals[i]
        end
    end
    
    return β_spread
end


function plot_range(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, β_vals, index) 
    sol1 = simulate_model(S, I, Is, R, c, β_vals[1], γ, ps, γs, α, tspan)
    sol2 = simulate_model(S, I, Is, R, c, β_vals[2], γ, ps, γs, α, tspan)
    sol3 = simulate_model(S, I, Is, R, c, β_vals[3], γ, ps, γs, α, tspan)
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="population",legend=false)
    plot!(sol1.t,sol1[index,:])
    plot!(sol2.t,sol2[index,:])
    plot!(sol3.t,sol3[index,:])
    plot!(data_time, data, seriestype=:scatter, label="data")
    ylims!(0,1.5*maximum(data))
    xlims!(0,1.5*maximum(data_time))
end


function plot_range(S, I, Is, R, c, γ, ps, γs, α, tspan, β_vals, index) 
    sol1 = simulate_model(S, I, Is, R, c, β_vals[1], γ, ps, γs, α, tspan)
    sol2 = simulate_model(S, I, Is, R, c, β_vals[2], γ, ps, γs, α, tspan)
    sol3 = simulate_model(S, I, Is, R, c, β_vals[3], γ, ps, γs, α, tspan)
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="population",legend=false)
    plot!(sol1.t,sol1[index,:])
    plot!(sol2.t,sol2[index,:])
    plot!(sol3.t,sol3[index,:])
end