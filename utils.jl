""" utils.jl

This file provides utility functions for FE simulations using Gridap.jl,
including boundary tag identification and consistent reaction force
calculations.
"""

using Gridap

"""
    get_tag_index(model, tag_name)

Return the index of the tag with name `tag_name` in the model's face labeling.

# Arguments
- `model`: The `DiscreteModel` containing the face labeling.
- `tag_name`: String name of the target boundary tag.

# Returns
- Integer index of the tag in `labels.tag_to_name`.
"""
function get_tag_index(model::DiscreteModel, tag_name::String)
    labels = get_face_labeling(model)
    for (tag, name) in enumerate(labels.tag_to_name)
        if name == tag_name
            return tag
        end
    end
    error("Tag $tag_name not found in model labels")
end

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
    calculate_reaction(a, tag_name, u_sol, v_space, model, dirichlet_tags)

Calculate the consistent reaction force on a boundary defined by `tag_name`.

The reaction is computed as the residual of the weak form: `R = -a(u, w)`,
where `w` is a test function that is 1.0 on the target boundary and 0.0 on all
other Dirichlet boundaries.

# Arguments
- `a`: The weak form functional (bilinear or nonlinear).
- `tag_name`: Name of the boundary where the reaction is calculated.
- `u_sol`: The FE solution.
- `v_space`: The test FE space.
- `model`: The discrete model.
- `dirichlet_tags`: The list of all Dirichlet tags used when creating `v_space`.

# Returns
- A scalar value representing the integrated reaction force (or N/m in 2D).
"""
function calculate_reaction(a::Function, tag_name::String, u_sol, v_space::FESpace, model::DiscreteModel, dirichlet_tags)
    # Check if tag exists
    get_tag_index(model, tag_name)

    # Create the test function weight
    w = create_boundary_weight(v_space, tag_name, dirichlet_tags)

    # The reaction is the residual: R = -a(u, w)
    return -sum(a(u_sol, w))
end
