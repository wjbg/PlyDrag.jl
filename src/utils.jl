""" utils.jl

This file provides generic utility functions for FE simulations using Gridap.jl,
including consistent reaction force calculations.
"""

"""
    create_boundary_weight(v_space, tag_name, dirichlet_tags)

Create a FE function `w` that is 1.0 on the boundary `tag_name` and 0.0 on all
other Dirichlet boundaries specified in `dirichlet_tags`.

# Arguments
- `v_space`: The test FE space.
- `tag_name`: The name of the boundary where the weight should be 1.0.
- `dirichlet_tags`: A collection of all Dirichlet tag names used in `v_space`.

# Returns
- A `CellField` (FE function) that acts as a test function for reaction
  calculations.
"""
function create_boundary_weight(v_space::FESpace, tag_name::String, dirichlet_tags)
    g_one(x) = 1.0
    g_zero(x) = 0.0

    vals = Function[]
    for name in dirichlet_tags
        push!(vals, name == tag_name ? g_one : g_zero)
    end

    w_space = TrialFESpace(v_space, vals)
    return interpolate(g_one, w_space)
end

"""
    calculate_reaction(a, tag_name, u_sol, v_space, dirichlet_tags)

Calculate the consistent reaction force on a boundary defined by `tag_name`.

The reaction is computed as the residual of the weak form: `R = -a(u, w)`,
where `w` is a test function that is 1.0 on the target boundary and 0.0 on all
other Dirichlet boundaries.

# Arguments
- `a`: The weak form functional (bilinear or nonlinear).
- `tag_name`: Name of the boundary where the reaction is calculated.
- `u_sol`: The FE solution.
- `v_space`: The test FE space.
- `dirichlet_tags`: The list of all Dirichlet tags used when creating `v_space`.

# Returns
- A scalar value representing the integrated reaction force (or N/m in 2D).
"""
function calculate_reaction(
    a::Function,
    tag_name::String,
    u_sol,
    v_space::FESpace,
    dirichlet_tags,
)
    # Create the test function weight
    w = create_boundary_weight(v_space, tag_name, dirichlet_tags)

    # The reaction is the residual: R = -a(u, w)
    return -sum(a(u_sol, w))
end

"""
    get_domain_dimensions(model)

Calculate the dimensions of the domain (length in 1D; width and height in 2D;
width, height, and depth in 3D).
"""
function get_domain_dimensions(model::DiscreteModel)
    coords = get_grid(model).node_coords
    D = num_point_dims(model)

    dims = map(1:D) do d
        x_min, x_max = extrema(c -> c[d], coords)
        return x_max - x_min
    end

    return D == 1 ? dims[1] : Tuple(dims)
end
