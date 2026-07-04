"""
    drag_fiber_hex_vs_sq_gmsh.jl

Compares drag flow for a range of fiber volume fractions and pulling rates.
"""
using DataFrames
using PlyDrag
using Printf
using RheoModels

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const TEMPERATURE = 273.15 + 365.0  # [K]
const TOP_VELOCITY = 1E-3 * (0.1:0.1:2.0) # [m/s]
const RES_FOLDER = "../results/"
const MSH_FOLDER = "../meshes/"
const HEX_FILES = ["2D_unstructured_hexagonal_packing_vf0.30_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.35_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.40_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.45_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.50_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.55_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.60_n20.msh",
                   "2D_unstructured_hexagonal_packing_vf0.65_n20.msh"]
const SQ_FILES = ["2D_unstructured_two_quarter_fiber_vf0.30_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.35_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.40_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.45_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.50_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.55_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.60_n20.msh",
                  "2D_unstructured_two_quarter_fiber_vf0.65_n20.msh"]

# Note: The mesh should have `Top` and `Bottom` identifiers, which should also include
#       the respective corner points!

# Viscosity model
rheomodel(γ̇) = LMPAEK(γ̇, TEMPERATURE)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

function vf_from_fn(msh_fn)
    return parse(Float64,
        split( split(msh_fn, "vf")[2], "_" )[1])
end

function vtu_fn(msh_fn, rate)
    bs = split(msh_fn, ".msh")[1]
    str_rate = @sprintf("%05.2f", rate)
    return bs * "u" * str_rate * ".vtu"
end

# -------------------------------------------------
# Run Simulation for square packign
# -------------------------------------------------

sq_results = DataFrame(vf = Float64[], u = Float64[], τ = Float64[])

for msh_file in SQ_FILES

    println("Starting simulation for square-packed unit cells...")
    println("")

    vf = vf_from_fn(msh_file)

    for v in TOP_VELOCITY
        nominal_stress, uh, model = simulate_drag_flow(
            MSH_FOLDER * msh_file,
            v,
            rheomodel,
            order = 1,
            quad_order = 10,
        )

        push!(sq_results, (vf, v, -nominal_stress))

        res_file = vtu_fn(msh_file, v*1000.0)
        write_drag_flow_solution(uh, model, rheomodel, RES_FOLDER * res_file)
        println("")

    end
end


# -------------------------------------------------
# Run Simulation for square packign
# -------------------------------------------------

hex_results = DataFrame(vf = Float64[], u = Float64[], τ = Float64[])

for msh_file in HEX_FILES

    println("Starting simulation for square-packed unit cells...")
    println("")

    vf = vf_from_fn(msh_file)

    for v in TOP_VELOCITY
        nominal_stress, uh, model = simulate_drag_flow(
            MSH_FOLDER * msh_file,
            v,
            rheomodel,
            order = 1,
            quad_order = 10,
        )

        push!(hex_results, (vf, v, -nominal_stress))

        res_file = vtu_fn(msh_file, v*1000.0)
        write_drag_flow_solution(uh, model, rheomodel, RES_FOLDER * res_file)
        println("")

    end
end
