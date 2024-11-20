module SIR_pkg

using Pkg
using Revise
using Plots
using DifferentialEquations
# Write your package code here.

export simulate_model
export plot_solution
export probability_to_rate
export plot_solution_SIRS
export get_R0
export get_pc
export SIR_error
export plot_error
export plot_range
export get_beta_range
export get_parameter_array
export get_parameter_range
export optimise_parameters
export find_start_time
export start_plot
export plot_solution!
export plot_data!
export get_maximum
export get_peak_time

include("SIR.jl")

end
