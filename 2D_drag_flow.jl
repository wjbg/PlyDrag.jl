using Gridap
using LineSearches: BackTracking
using Printf

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const GAS_CONSTANT = 8.31 # Universal gas constant [J/(mol.K)]
const TEMPERATURE = 273.15 + 300.0  # Temperature [K]
const THICKNESS = 0.001 # Domain thickness [m]
const TOP_VELOCITY = 1.0e-3  # Speed of the top plate [m/s]

const PPS_PARAMS = (
    c_0 = 1.25e-4,  # Proportionality constant for zero-shear viscosity
    e_0 = 6.86e4,   # Activation energy zero-shear viscosity
    n  = 0.28,      # Powerlaw index
    c_lambda = 2.21e-8,  # Proportionality constant for relaxation time
    e_lambda = 4.50e4    # Activation energy relaxation time
)

# -------------------------------------------------
# Physical Model Functions
# -------------------------------------------------

function zero_shear_viscosity(temp, params)
    return params.c_0 * exp(params.e_0 / (GAS_CONSTANT * temp))
end

function viscosity(shear_rate, temp, params)
    μ_0 = zero_shear_viscosity(temp, params)
    λ = params.c_lambda * exp(params.e_lambda / (GAS_CONSTANT * temp))
    return μ_0 / (1 + (λ * shear_rate)^(1 - params.n))
end

function analytical_shear_stress(velocity, thickness)
    γ_analytical = velocity / thickness
    η = viscosity(γ_analytical, TEMPERATURE, PPS_PARAMS)
    return η * γ_analytical
end

# -------------------------------------------------
# Mesh and Labels
# -------------------------------------------------

domain = (0, THICKNESS / 10, 0, THICKNESS)
partition = (10, 100)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels, "bottom", [5, 1, 2])
add_tag_from_tags!(labels, "top", [6, 3, 4])

# -------------------------------------------------
# FE Spaces
# -------------------------------------------------

order = 1
v_space = TestFESpace(
    model,
    ReferenceFE(lagrangian, Float64, order);
    conformity=:H1,
    dirichlet_tags=["bottom", "top"]
)

g_top(x) = TOP_VELOCITY
g_bottom(x) = 0.0

u_space = TrialFESpace(v_space, [g_bottom, g_top])

# -------------------------------------------------
# Weak Form
# -------------------------------------------------

Ω = Triangulation(model)
dΩ = Measure(Ω, 2 * order)

function a(u, v)
    γₑ = sqrt ∘ (∇(u) ⋅ ∇(u) + 1e-12)
    μ = (γ -> viscosity(γ, TEMPERATURE, PPS_PARAMS)) ∘ γₑ
    return ∫(μ * (∇(u) ⋅ ∇(v)))dΩ
end

# -------------------------------------------------
# Nonlinear Solve
# -------------------------------------------------

op = FEOperator(a, u_space, v_space)
nls = NLSolver(
    show_trace=true,
    method=:newton,
    linesearch=BackTracking()
)

solver = FESolver(nls)

println("Numerical solving has started...")
uh = solve(solver, op)
println("Numerical solving has ended.")

# -------------------------------------------------
# Post-processing and Output
# -------------------------------------------------

grad_uh = ∇(uh)
γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
μₕ = (γ -> viscosity(γ, TEMPERATURE, PPS_PARAMS)) ∘ γₕ
τₕ = μₕ * grad_uh

# Calculate average numerical shear stress
area = sum(∫(1.0)dΩ)
τₕ_avg = sum(∫(sqrt ∘ (τₕ ⋅ τₕ + 1e-12))dΩ) / area
τₐ = analytical_shear_stress(TOP_VELOCITY, THICKNESS)

@printf("Analytical shear stress: %.1f Pa\n", τₐ)
@printf("Numerical shear stress:  %.1f Pa\n", τₕ_avg)

writevtk(Ω, "carreau_flow", cellfields=[
    "u" => uh,
    "shear_rate" => γₕ,
    "shear_stress" => τₕ
])
