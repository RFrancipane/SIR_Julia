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
    plot_solution(sol::ODESolution, data, tspan, labels)

Plots Infected from SIR model against infection data

# Arguments
- `sol::ODESolution`: Solution of SIR model
- `data::Vector{}`: Vector of number of infections over time
- `data_time::Vector{}`: Times of datapoints
- `labels::Vector{}`: Labels For graph, ordered by Title, X axis, Y axis, Model label, Data label

"""
function plot_solution(sol::ODESolution, index, data::Vector{}, data_time::Vector{}, labels::Vector{})
    plot(linewidth=4, title=labels[1],xaxis=labels[2],yaxis=labels[3],legend=true)
    plot!(sol.t,sol[index,:], label = labels[4])
    plot!(data_time, data, seriestype=:scatter, label=labels[5])
    xlims!(0,2*maximum(data_time))
    ylims!(0,2*maximum(data))
end


"""
    error(sol::ODESolution, data::Vector{})

Calculates Mean squared error of SIR solution compared to data

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
    total /= (length(data)-2)
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
    plot(sol, xlabel="Time", ylabel = "Population",label = ["S" "I" "Is" "R"], title = "SIRS Model")
    plot!(legend=:outerbottom, legendcolumns=4)
end

function simulate_model(args::Vector{}, tspan::Vector{}) 
    len = size(args, 1)
    if (len == 5) 
        return simulate_model(args[1], args[2], args[3], args[4], args[5], tspan)
    end
    if (len == 6) 
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], tspan)
    end
    if (len == 10)
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], tspan)
    end

    if (len == 13)
        return simulate_model(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], tspan)
    end
    println("Wrong params length")
end

function get_error_spread(params, tspan, data, data_time, data_index, index, vals)
    error_array = []
    for param_val in vals
        params[index] = param_val
        solution = simulate_model(params, tspan)
        total_error = SIR_error(solution, data, data_time, data_index)
        push!(error_array, total_error)
    end
    return error_array
end

function get_error_spread(params, tspan, data, data_time, data_s, data_time_s, index, vals)
    error_array = []
    for param_val in vals
        params[index] = param_val
        solution = simulate_model(params, tspan)
        total_error = SIR_error(solution, data, data_time, 2) + SIR_error(solution, data_s, data_time_s, 3)
        push!(error_array, total_error)
    end
    return error_array
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

function plot_error(params, tspan, data, data_time, data_index, param_index, param_range)
    step = 0.0001
    vals = param_range[1]:step:param_range[2]
    error_array = get_error_spread(params, tspan, data, data_time, data_index, param_index, vals)
    plot(vals, error_array, xlabel="Param", ylabel = "Mean Squared Error", title = "Error vs Param", legend=false)
end

function plot_error(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_range)
    step = 0.0001
    β_vals = β_range[1]:step:β_range[2]
    error_array = get_error_spread(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_vals)
    plot(β_vals, error_array, xlabel="β", ylabel = "Mean Squared Error", title = "Error vs β", legend=false)
end

function get_beta_range(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_range, std)
    step = 0.0001
    β_vals = β_range[1]:step:β_range[2]
    error_array = get_error_spread(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, data_s, data_time_s, β_vals)
    min_error = minimum(error_array)
    #Approx Standard deviation of MSE is sqrt(MSE)
    error_diff = std*sqrt(min_error)
    β_spread = [0,β_vals[argmin(error_array)],0]


    for i in range(1, length(error_array))
        if (error_array[i] <= min_error+error_diff && β_spread[1] == 0)
            β_spread[1] = β_vals[i]
        end
        if (error_array[i] >= min_error+error_diff && β_spread[1] != 0)
            β_spread[3] = β_vals[i]
            #Break early once spread found
            return β_spread
        end
    end
    
    return β_spread
end

function get_parameter_range(params, tspan, data, data_time, data_s, data_time_s, param_range, index, std)
    step = 0.0001
    vals = param_range[1]:step:param_range[2]
    error_array = get_error_spread(params, tspan, data, data_time, data_s, data_time_s, index, vals)
    min_error = minimum(error_array)
    #Approx Standard deviation of MSE is sqrt(MSE)
    error_diff = std*sqrt(min_error)
    param_spread = [0,vals[argmin(error_array)],0]


    for i in range(1, length(error_array))
        if (error_array[i] <= min_error+error_diff && param_spread[1] == 0)
            param_spread[1] = vals[i]
        end
        if (error_array[i] >= min_error+error_diff && param_spread[1] != 0)
            param_spread[3] = vals[i]
            #Break early once spread found
            return param_spread
        end
    end
    
    return param_spread
end

function optimise_parameter(params, tspan, data, data_time, data_s, data_time_s, param_range, index)
    step = 0.0001
    vals = param_range[1]:step:param_range[2]
    error_array = get_error_spread(params, tspan, data, data_time, data_s, data_time_s, index, vals)
    min_error = minimum(error_array)
    return vals[argmin(error_array)]
end

function get_parameter_range(params, tspan, data, data_time, param_range, param_index, index, std)
    step = 0.0001
    vals = param_range[1]:step:param_range[2]
    error_array = get_error_spread(params, tspan, data, data_time, param_index, index, vals)
    min_error = minimum(error_array)
    #Approx Standard deviation of MSE is sqrt(MSE)
    error_diff = std*sqrt(min_error)
    param_spread = [0,vals[argmin(error_array)],0]


    for i in range(1, length(error_array))
        if (error_array[i] <= min_error+error_diff && param_spread[1] == 0)
            param_spread[1] = vals[i]
        end
        if (error_array[i] >= min_error+error_diff && param_spread[1] != 0)
            param_spread[3] = vals[i]
            #Break early once spread found
            return param_spread
        end
    end
    
    return param_spread
end


function plot_range(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, β_vals, index) 
    sol1 = simulate_model(S, I, Is, R, c, β_vals[1], γ, ps, γs, α, tspan)
    sol2 = simulate_model(S, I, Is, R, c, β_vals[2], γ, ps, γs, α, tspan)
    sol3 = simulate_model(S, I, Is, R, c, β_vals[3], γ, ps, γs, α, tspan)
    plot(linewidth=4, title="SIRS Solution",xaxis="Time",yaxis="Population",legend=true)
    plot!(sol1.t,sol1[index,:], label = "Best case prediction")
    plot!(sol2.t,sol2[index,:], label = "Mean prediction")
    plot!(sol3.t,sol3[index,:], label = "Worse case prediction")
    plot!(data_time, data, seriestype=:scatter, label="Data")
end

function plot_range(S, I, Is, R, c, γ, ps, γs, α, tspan, data, data_time, β_vals, index, labels) 
    sol1 = simulate_model(S, I, Is, R, c, β_vals[1], γ, ps, γs, α, tspan)
    sol2 = simulate_model(S, I, Is, R, c, β_vals[2], γ, ps, γs, α, tspan)
    sol3 = simulate_model(S, I, Is, R, c, β_vals[3], γ, ps, γs, α, tspan)
    plot(linewidth=7, title=labels[1],xaxis=labels[2],yaxis=labels[3],legend=true, palette = :Dark2_8)
    plot!(sol1.t,sol1[index,:], label = "Best case prediction")
    plot!(sol2.t,sol2[index,:], label = "Mean prediction")
    plot!(sol3.t,sol3[index,:], label = "Worse case prediction")
    plot!(data_time, data, seriestype=:scatter, label=labels[4])
end

function plot_range(S, I, Is, R, c, γ, ps, γs, α, tspan, β_vals, index) 
    sol1 = simulate_model(S, I, Is, R, c, β_vals[1], γ, ps, γs, α, tspan)
    sol2 = simulate_model(S, I, Is, R, c, β_vals[2], γ, ps, γs, α, tspan)
    sol3 = simulate_model(S, I, Is, R, c, β_vals[3], γ, ps, γs, α, tspan)
    plot(linewidth=4, title="Solution",xaxis="t",yaxis="population",legend=false, palette = :davos10)
    plot!(sol1.t,sol1[index,:])
    plot!(sol2.t,sol2[index,:])
    plot!(sol3.t,sol3[index,:])
end

function get_parameter_array(S, I, R, λ, γ) 
    return [S, I, R, λ, γ]
end

function get_parameter_array(S, I, R, c, β, γ) 
    return [S, I, R, c, β, γ]
end

function get_parameter_array(S, I, Is, R, c, β, γ, ps, γs, α) 
    return [S, I, Is, R, c, β, γ, ps, γs, α]
end

function get_parameter_array(S, I, Is, R, c, β, γ, ps, γs, α, ϵ, Φ, intervention_time) 
    return [S, I, Is, R, c, β, γ, ps, γs, α, ϵ, Φ, intervention_time]
end

function optimise_parameters(params, tspan, data, data_time, data_s, data_time_s, variable_params, num_iterations)
    param_range = [0,1]
    for i in range(1, num_iterations)
        for i in variable_params
            optimised_param = optimise_parameter(params, tspan, data, data_time, data_s, data_time_s, param_range, i)
            params[i] = optimised_param
        end
    end
    return params
end