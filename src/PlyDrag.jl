module PlyDrag

using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Printf

# Exported types and functions
export TemperatureModel, RheologyModel
export Arrhenius, WLF, Constant
export CrossModel, PPS, ConstantViscosity, Newtonian
export shift_factor, viscosity

export calculate_reaction
export get_domain_dimensions, solve_drag_flow, write_drag_flow_vtk, simulate_drag_flow

# Include sub-files
include("viscosity.jl")
include("utils.jl")
include("drag_flow.jl")

end # module
