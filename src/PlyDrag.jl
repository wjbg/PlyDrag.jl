module PlyDrag

using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Printf

# Exported types and functions
export calculate_reaction
export get_domain_dimensions, solve_drag_flow, write_drag_flow_vtk, simulate_drag_flow
export setup_spaces, longitudinal_bulk_weak_form, setup_drag_flow_operator, solve_nonlinear_stokes
export write_drag_flow_solution

# Include sub-files
include("utils.jl")
include("drag_flow.jl")

end # module
