""" drag_flow.jl

This file defines the core simulation logic for 2D drag flow problems,
including solver setup, domain calculations, and post-processing.
"""

"""
    setup_spaces(model::DiscreteModel;
                 order = 1,
                 dirichlet_tags = String[],
                 dirichlet_vals = Float64[])

Setup the trial and test FE spaces for a 2D longitudinal drag flow problem.

This function creates the underlying `TestFESpace` and `TrialFESpace` required for the
finite element simulation.

# Arguments
- `model`: A `DiscreteModel` representing the computational domain.

# Keywords
- `order`: The polynomial order of the Lagrangian finite elements (default: 1).
- `dirichlet_tags`: A vector of boundary tag names where Dirichlet conditions are
                    applied (default: empty).
- `dirichlet_vals`: A vector of values (or functions) associated with each tag in
                    `dirichlet_tags` (default: empty).

# Returns
- `U`: The `TrialFESpace`.
- `V`: The `TestFESpace`.
"""
function setup_spaces(
    model::DiscreteModel;
    order = 1,
    dirichlet_tags = String[],
    dirichlet_vals = Float64[],
)
    V = TestFESpace(
        model,
        ReferenceFE(lagrangian, Float64, order);
        conformity = :H1,
        dirichlet_tags = dirichlet_tags
    )

    # Convert numeric values to functions if necessary
    _dirichlet_vals = map(v -> v isa Function ? v : (x -> Float64(v)), dirichlet_vals)
    U = TrialFESpace(V, _dirichlet_vals)

    return U, V
end

"""
    setup_drag_flow_weak_form(model::DiscreteModel, rheology; quad_order = 10)

Create the default bulk weak form functional for the drag flow problem.

The weak form represents the conservation of momentum in a 1D-flow-in-2D-domain
approximation, assuming a nonlinear viscosity dependent on the shear rate.

# Arguments
- `model`: The computational domain model.
- `rheology`: A function `η(γ)` that returns viscosity given a shear rate.

# Keywords
- `quad_order`: The integration quadrature order (default: 10).

# Returns
- `a(u, v)`: A functional representing the weak form `∫(μ(∇u) * ∇u ⋅ ∇v) dΩ`.
"""
function setup_drag_flow_weak_form(model::DiscreteModel, rheology; quad_order = 10)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, quad_order)

    function a(u, v)
        γₑ = sqrt ∘ (∇(u) ⋅ ∇(u) + 1e-12)
        μ = (γ -> rheology(γ)) ∘ γₑ
        return ∫(μ * (∇(u) ⋅ ∇(v))) * dΩ
    end
    return a
end

"""
    setup_drag_flow_operator(u_space, v_space, a)

Construct the `FEOperator` from the given FE spaces and weak form functional.

# Arguments
- `u_space`: The trial FE space.
- `v_space`: The test FE space.
- `a`: The weak form functional `a(u, v)`.

# Returns
- A `FEOperator` ready to be solved.
"""
function setup_drag_flow_operator(u_space, v_space, a)
    return FEOperator(a, u_space, v_space)
end

"""
    solve_nonlinear_stokes(op; show_trace = true)

Solve a nonlinear FE problem using a Newton-Raphson solver with backtracking line search.

# Arguments
- `op`: The `FEOperator` defining the system.

# Keywords
- `show_trace`: Boolean flag to print solver convergence information (default: true).

# Returns
- `uh`: The FE solution field.
"""
function solve_nonlinear_stokes(op; show_trace = true)
    nls = NLSolver(
        show_trace = show_trace,
        method = :newton,
        linesearch = BackTracking(),
        ftol = 1e-10,
        xtol = 1e-12,
    )
    solver = FESolver(nls)
    return solve(solver, op)
end

"""
    solve_drag_flow(model::DiscreteModel, top_velocity::Real, rheology;
                    order = 1, quad_order = 10)

Orchestrate the setup and solving of a 2D drag flow problem.

This is the main entry point for solving a simulation given a pre-loaded model.
It modularly calls space setup, weak form construction, and the nonlinear solver.

# Arguments
- `model`: The domain model.
- `top_velocity`: prescribed top plate velocity.
- `rheology`: Viscosity function `η(γ)`.

# Keywords
- `order`: FE polynomial order (default: 1).
- `quad_order`: Quadrature order (default: 10).

# Returns
- `uh`: The solution field.
- `a`: The weak form functional.
- `V`: The test FE space.
- `dirichlet_tags`: Boundary tags used in the simulation.
"""
function solve_drag_flow(
    model::DiscreteModel,
    top_velocity::Real,
    rheology;
    order = 1,
    quad_order = 10,
)
    dirichlet_tags = ["Bottom", "Top"]
    U, V = setup_spaces(
        model;
        order = order,
        dirichlet_tags = dirichlet_tags,
        dirichlet_vals = [0.0, top_velocity],
    )
    a = setup_drag_flow_weak_form(model, rheology; quad_order = quad_order)
    op = setup_drag_flow_operator(U, V, a)
    uh = solve_nonlinear_stokes(op)

    return uh, a, V, dirichlet_tags
end

"""
    write_drag_flow_vtk(uh, model::DiscreteModel, rheology, filename; quad_order = 10)

Compute derived fields (shear rate, stress, viscosity) and export to VTK format.

# Arguments
- `uh`: The velocity solution field.
- `model`: The domain model.
- `rheology`: Viscosity function `η(γ)`.
- `filename`: Target path for the VTU file (without extension).

# Keywords
- `quad_order`: Quadrature order used for field calculations (default: 10).
"""
function write_drag_flow_vtk(
    uh,
    model::DiscreteModel,
    rheology,
    filename::String;
    quad_order = 10,
)
    Ω = Triangulation(model)
    grad_uh = ∇(uh)
    γₕ = sqrt ∘ (grad_uh ⋅ grad_uh + 1e-12)
    μₕ = (γ -> rheology(γ)) ∘ γₕ
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
    simulate_drag_flow(msh_file::String, top_velocity::Real, rheology;
                       order = 1, quad_order = 10)

Run a complete drag flow simulation starting from a Gmsh file.

This high-level function handles model loading, solving, reaction force
calculation, and VTK output. It returns the nominal shear stress calculated
at the top boundary.

# Arguments
- `msh_file`: Path to the `.msh` file.
- `top_velocity`: Velocity of the top boundary.
- `rheology`: Viscosity function `η(γ)`.

# Keywords
- `order`: FE polynomial order (default: 1).
- `quad_order`: Quadrature order (default: 10).

# Returns
- `nominal_stress`: The integrated reaction force at the top boundary divided
  by the domain width.
"""
function simulate_drag_flow(
    msh_file::String,
    top_velocity::Real,
    rheology;
    order = 1,
    quad_order = 10,
)
    model = GmshDiscreteModel(msh_file)

    println("Solving drag flow for $msh_file...")
    uh, a, V, dirichlet_tags = solve_drag_flow(
        model,
        top_velocity,
        rheology;
        order = order,
        quad_order = quad_order,
    )

    force_top = calculate_reaction(a, "Top", uh, V, dirichlet_tags)
    dims = get_domain_dimensions(model)
    width = dims isa Tuple ? dims[1] : dims
    nominal_stress = force_top / width

    vtu_name = splitext(msh_file)[1]
    write_drag_flow_vtk(
        uh,
        model,
        rheology,
        vtu_name;
        quad_order = quad_order,
    )

    println("Simulation finished. Nominal stress: $nominal_stress")
    return nominal_stress
end
