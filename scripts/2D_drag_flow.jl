using Gridap
using PlyDrag
using Printf

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const TEMPERATURE = 273.15 + 300.0  # Temperature [K]
const THICKNESS = 0.001  # Domain thickness [m]
const TOP_VELOCITY = 1.0e-3  # Speed of the top plate [m/s]

# -------------------------------------------------
# Physical Model Functions
# -------------------------------------------------

function analytical_shear_stress(velocity, thickness)
    γ_analytical = velocity / thickness
    η = viscosity(PPS, γ_analytical, TEMPERATURE)
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
dirichlet_tags = ["bottom", "top"]
v_space = TestFESpace(
    model,
    ReferenceFE(lagrangian, Float64, order);
    conformity = :H1,
    dirichlet_tags = dirichlet_tags,
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
    μ = (γ -> viscosity(PPS, γ, TEMPERATURE)) ∘ γₑ
    return ∫(μ * (∇(u) ⋅ ∇(v))) * dΩ
end

# -------------------------------------------------
# Nonlinear Solve
# -------------------------------------------------

op = FEOperator(a, u_space, v_space)
nls = NLSolver(show_trace = true, method = :newton, linesearch = BackTracking())

solver = FESolver(nls)

println("Numerical solving has started...")
uh = solve(solver, op)
println("Numerical solving has ended.")

# -------------------------------------------------
# Post-processing and Output
# -------------------------------------------------

grad_uh = ∇(uh)
γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
μₕ = (γ -> viscosity(PPS, γ, TEMPERATURE)) ∘ γₕ
τₕ = μₕ * grad_uh

# Calculate average numerical shear stress
area = sum(∫(1.0) * dΩ)
τₕ_avg = sum(∫(sqrt ∘ (τₕ ⋅ τₕ + 1e-12)) * dΩ) / area
τₐ = analytical_shear_stress(TOP_VELOCITY, THICKNESS)

# Consistent reactions
force_top = calculate_reaction(a, "top", uh, v_space, dirichlet_tags)
force_bottom = calculate_reaction(a, "bottom", uh, v_space, dirichlet_tags)

@printf("Analytical shear stress: %.1f Pa\n", τₐ)
@printf("Numerical shear stress:  %.1f Pa\n", τₕ_avg)
@printf("Consistent Force Top:    %.6e N/m\n", force_top)
@printf("Consistent Force Bottom: %.6e N/m\n", force_bottom)

writevtk(
    Ω,
    "2D_dragflow_flatplate",
    cellfields = ["u" => uh, "shear_rate" => γₕ, "shear_stress" => τₕ],
)
