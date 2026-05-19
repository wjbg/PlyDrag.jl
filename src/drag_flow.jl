""" drag_flow.jl

This file defines the core simulation logic for 2D drag flow problems,
including solver setup, domain calculations, and post-processing.
"""

# drag_flow.jl
# Logic moved to module scope

"""
    solve_drag_flow(model, top_velocity, rheology, temperature; order = 1, quad_order = 10)

Solve the 2D drag flow problem and return the solution, weak form, and FE space info.
"""
function solve_drag_flow(
    model::DiscreteModel,
    top_velocity::Real,
    rheology,
    temperature::Real;
    order = 1,
    quad_order = 10,
)
    dirichlet_tags = ["Bottom", "Top"]
    v_space = TestFESpace(
        model,
        ReferenceFE(lagrangian, Float64, order);
        conformity = :H1,
        dirichlet_tags = dirichlet_tags,
    )

    g_top(x) = top_velocity
    g_fixed(x) = 0.0
    u_space = TrialFESpace(v_space, [g_fixed, g_top])

    Ω = Triangulation(model)
    dΩ = Measure(Ω, quad_order)

    function a(u, v)
        γₑ = sqrt ∘ (∇(u) ⋅ ∇(u) + 1e-12)
        μ = (γ -> viscosity(rheology, γ, temperature)) ∘ γₑ
        return ∫(μ * (∇(u) ⋅ ∇(v))) * dΩ
    end

    op = FEOperator(a, u_space, v_space)
    nls = NLSolver(
        show_trace = true,
        method = :newton,
        linesearch = BackTracking(),
        ftol = 1e-10,
        xtol = 1e-12,
    )
    solver = FESolver(nls)
    uh = solve(solver, op)

    return uh, a, v_space, dirichlet_tags
end

"""
    write_drag_flow_vtk(uh, model, rheology, temperature, filename; quad_order = 10)

Post-process the solution and write results to a VTU file.
"""
function write_drag_flow_vtk(
    uh,
    model::DiscreteModel,
    rheology,
    temperature::Real,
    filename::String;
    quad_order = 10,
)
    Ω = Triangulation(model)
    grad_uh = ∇(uh)
    γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
    μₕ = (γ -> viscosity(rheology, γ, temperature)) ∘ γₕ
    τₕ = μₕ * grad_uh

    return writevtk(
        Ω,
        filename,
        cellfields = [
            "u" => uh,
            "shear_rate" => γₕ,
            "shear_stress" => τₕ,
            "viscosity" => μₕ,
        ],
    )
end

"""
    simulate_drag_flow(msh_file, top_velocity, rheology, temperature; order = 1, quad_order = 10)

Perform a 2D drag flow simulation on a given Gmsh mesh and return the nominal
shear stress (force / width).
"""
function simulate_drag_flow(
    msh_file::String,
    top_velocity::Real,
    rheology,
    temperature::Real;
    order = 1,
    quad_order = 10,
)
    model = GmshDiscreteModel(msh_file)

    println("Solving drag flow for $msh_file...")
    uh, a, v_space, dirichlet_tags = solve_drag_flow(
        model,
        top_velocity,
        rheology,
        temperature;
        order = order,
        quad_order = quad_order,
    )

    force_top = calculate_reaction(a, "Top", uh, v_space, dirichlet_tags)
    dims = get_domain_dimensions(model)
    width = dims isa Tuple ? dims[1] : dims
    nominal_stress = force_top / width

    vtu_name = splitext(msh_file)[1]
    write_drag_flow_vtk(
        uh,
        model,
        rheology,
        temperature,
        vtu_name;
        quad_order = quad_order,
    )

    println("Simulation finished. Nominal stress: $nominal_stress")
    return nominal_stress
end
