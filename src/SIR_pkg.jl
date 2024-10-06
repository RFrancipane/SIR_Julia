module SIR_pkg

using Pkg
using Revise
using Plots
using DifferentialEquations
# Write your package code here.

export simulate_model
export plot_solution
export time_cdf_gamma
export plot_solution_SIRS

include("SIR.jl")

end
