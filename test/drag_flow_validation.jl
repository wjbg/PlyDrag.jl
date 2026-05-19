using PlyDrag
using Gridap
using Test

@testset "2D Drag Flow Validation" begin
    # -------------------------------------------------
    # Parameters
    # -------------------------------------------------
    H = 0.001          # Height [m]
    W = 0.1 * H        # Width [m]
    T = 273.15 + 300.0 # Temperature [K]
    V_top = 1.0e-3     # Velocity [m/s]
    
    # -------------------------------------------------
    # Mesh and Labels
    # -------------------------------------------------
    domain = (0, W, 0, H)
    partition = (5, 50)
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "Bottom", [5, 1, 2])
    add_tag_from_tags!(labels, "Top", [6, 3, 4])

    # -------------------------------------------------
    # Solve
    # -------------------------------------------------
    uh, a, v_space, dirichlet_tags = solve_drag_flow(
        model, V_top, PPS, T; order = 1, quad_order = 4
    )

    # -------------------------------------------------
    # Analytical Comparison
    # -------------------------------------------------
    γ_analytical = V_top / H
    η_analytical = viscosity(PPS, γ_analytical, T)
    τ_analytical = η_analytical * γ_analytical

    # Numerical post-processing
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)
    
    grad_uh = ∇(uh)
    γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
    μₕ = (γ -> viscosity(PPS, γ, T)) ∘ γₕ
    τₕ = μₕ * γₕ

    # Average numerical values
    area = sum(∫(1.0) * dΩ)
    γ_avg = sum(∫(γₕ) * dΩ) / area
    τ_avg = sum(∫(τₕ) * dΩ) / area

    # Validations
    @test γ_avg ≈ γ_analytical rtol=1e-6
    @test τ_avg ≈ τ_analytical rtol=1e-6

    # Consistent reactions
    force_top = calculate_reaction(a, "Top", uh, v_space, dirichlet_tags)
    force_bottom = calculate_reaction(a, "Bottom", uh, v_space, dirichlet_tags)

    # force_top should be τ * width (magnitude)
    @test abs(force_top) ≈ τ_analytical * W rtol=1e-4
    @test force_top ≈ -force_bottom rtol=1e-6
end
