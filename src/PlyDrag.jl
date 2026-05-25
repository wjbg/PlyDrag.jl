module PlyDrag

using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Printf

# Exported types and functions
export TemperatureModel, RheologyModel
export Arrhenius, WLF, Constant
export PPS, Newtonian

export calculate_reaction
export get_domain_dimensions, solve_drag_flow, write_drag_flow_vtk, simulate_drag_flow
export setup_spaces, longitudinal_bulk_weak_form, setup_drag_flow_operator, solve_nonlinear_stokes

# Include sub-files
include("viscosity.jl")
include("utils.jl")
include("drag_flow.jl")

end # module
