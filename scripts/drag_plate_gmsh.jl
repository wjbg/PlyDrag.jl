"""
    drag_plate_gmsh.jl

Solves 2D drag flow between two plates using gmsh file to validate solver.

"""
using PlyDrag
using Printf
using RheoModels

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const TEMPERATURE = 273.15 + 300.0  # Temperature [K]
const TOP_VELOCITY = 1.0e-3  # Speed of the top plate [m/s]
const MSH_FILE = "../meshes/2D_rectangle.msh"

# Note: The mesh should have `Top` and `Bottom` identifiers, which should also include
#       the respective corner points!

PPS300(γ̇) = PPS(γ̇, TEMPERATURE)

# -------------------------------------------------
# Run Simulation
# -------------------------------------------------

nominal_stress, uh, model = simulate_drag_flow(
    MSH_FILE,
    TOP_VELOCITY,
    PPS300,
    order = 1,
    quad_order = 10,
)

@printf("\nFinal Nominal Shear Stress: %.6e Pa\n", nominal_stress)
