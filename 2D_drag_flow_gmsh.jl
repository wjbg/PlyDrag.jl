using Gridap
using GridapGmsh
using LineSearches: BackTracking
using Printf

include("viscosity_models.jl")
include("utils.jl")

# -------------------------------------------------
# Constants and Parameters
# -------------------------------------------------

const TEMPERATURE = 273.15 + 300.0  # Temperature [K]
const TOP_VELOCITY = 1.0e-3  # Speed of the top plate [m/s]

# -------------------------------------------------
# Mesh and Labels
# -------------------------------------------------

# Load the Gmsh mesh
model = GmshDiscreteModel("square_packing.msh")

# -------------------------------------------------
# FE Spaces
# -------------------------------------------------

order = 1
dirichlet_tags = ["Bottom", "Fiber", "Top"]
v_space = TestFESpace(
    model,
    ReferenceFE(lagrangian, Float64, order);
    conformity=:H1,
    dirichlet_tags=dirichlet_tags
)

# Define boundary functions
g_top(x) = TOP_VELOCITY
g_fixed(x) = 0.0

u_space = TrialFESpace(v_space, [g_fixed, g_fixed, g_top])

# -------------------------------------------------
# Weak Form
# -------------------------------------------------

Ω = Triangulation(model)
# Higher integration order for non-linear viscosity
quad_order = 10
dΩ = Measure(Ω, quad_order)

function a(u, v)
    # Small epsilon to avoid singularity at zero shear rate
    γₑ = sqrt ∘ (∇(u) ⋅ ∇(u) + 1e-12)
    μ = (γ -> viscosity(PPS, γ, TEMPERATURE)) ∘ γₑ
    return ∫(μ * (∇(u) ⋅ ∇(v)))dΩ
end

# -------------------------------------------------
# Nonlinear Solve
# -------------------------------------------------

op = FEOperator(a, u_space, v_space)
nls = NLSolver(
    show_trace=true,
    method=:newton,
    linesearch=BackTracking(),
    ftol=1e-10,
    xtol=1e-12
)

solver = FESolver(nls)

println("Numerical solving has started for Gmsh model...")
uh = solve(solver, op)
println("Numerical solving has ended.")

# -------------------------------------------------
# Post-processing and Output
# -------------------------------------------------

# 1. Consistent Reaction Forces (Weak Form Residual)
# We test the weak form 'a' with a function that is 1 on the target boundary and 0 on others.
println("Calculating consistent reaction forces...")

force_top_cons = calculate_reaction(a, "Top", uh, v_space, model, dirichlet_tags)
force_bot_cons = calculate_reaction(a, "Bottom", uh, v_space, model, dirichlet_tags)
force_fiber_cons = calculate_reaction(a, "Fiber", uh, v_space, model, dirichlet_tags)

# 2. Stress Integration (for comparison and symmetry residuals)
grad_uh = ∇(uh)
γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
μₕ = (γ -> viscosity(PPS, γ, TEMPERATURE)) ∘ γₕ
τₕ = μₕ * grad_uh

Γ_left = BoundaryTriangulation(model, tags="Left")
dΓ_left = Measure(Γ_left, quad_order)
n_left = get_normal_vector(Γ_left)

Γ_right = BoundaryTriangulation(model, tags="Right")
dΓ_right = Measure(Γ_right, quad_order)
n_right = get_normal_vector(Γ_right)

force_left = sum(∫(τₕ ⋅ n_left)dΓ_left)
force_right = sum(∫(τₕ ⋅ n_right)dΓ_right)

println("\n--- Force Balance Summary (Consistent) ---")
@printf("Force (Top):          %.6e N/m\n", force_top_cons)
@printf("Force (Bottom):       %.6e N/m\n", force_bot_cons)
@printf("Force (Fiber):        %.6e N/m\n", force_fiber_cons)
@printf("Force (Bottom+Fiber): %.6e N/m\n", force_bot_cons + force_fiber_cons)
@printf("Net Force (T+B+F):    %.6e N/m\n", force_top_cons + force_bot_cons + force_fiber_cons)

println("\n--- Symmetry Residuals (Stress Integration) ---")
@printf("Force (Left Side):    %.6e N/m\n", force_left)
@printf("Force (Right Side):   %.6e N/m\n", force_right)
@printf("Total Balance:        %.6e N/m\n", force_top_cons + force_bot_cons + force_fiber_cons + force_left + force_right)
println("-----------------------------\n")

# Extract directional components
γₕ_x = (v -> v[1]) ∘ grad_uh
γₕ_y = (v -> v[2]) ∘ grad_uh
τₕ_x = (v -> v[1]) ∘ τₕ
τₕ_y = (v -> v[2]) ∘ τₕ

writevtk(Ω, "drag_flow_results", cellfields=[
    "u" => uh,
    "shear_rate" => γₕ,
    "shear_rate_x" => γₕ_x,
    "shear_rate_y" => γₕ_y,
    "shear_stress" => τₕ,
    "shear_stress_x" => τₕ_x,
    "shear_stress_y" => τₕ_y,
    "viscosity" => μₕ
])

println("Results saved to drag_flow_results.vtu")
