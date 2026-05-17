using PlyDrag
using Printf

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const TEMPERATURE = 273.15 + 300.0  # Temperature [K]
const TOP_VELOCITY = 1.0e-3  # Speed of the top plate [m/s]
const MSH_FILE = "square_packing.msh"

# -------------------------------------------------
# Run Simulation
# -------------------------------------------------

nominal_stress = simulate_drag_flow(
    MSH_FILE,
    TOP_VELOCITY,
    PPS,
    TEMPERATURE;
    order=1,
    quad_order=10
)

@printf("\nFinal Nominal Shear Stress: %.6e Pa\n", nominal_stress)
