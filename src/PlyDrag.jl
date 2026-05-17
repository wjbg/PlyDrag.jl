module PlyDrag

using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Printf

# Exported types and functions
export TemperatureModel, RheologyModel
export Arrhenius, WLF, Constant
export CrossModel, PPS
export shift_factor, viscosity

export get_tag_index, calculate_reaction
export get_domain_width, solve_drag_flow, write_drag_flow_vtk, simulate_drag_flow

# Include sub-files
include("viscosity.jl")
include("utils.jl")
include("drag_flow.jl")

end # module
