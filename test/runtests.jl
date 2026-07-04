using PlyDrag
using Gridap
using Test

@testset "PlyDrag.jl" begin
    @testset "Domain Dimensions" begin
        # 1D Model
        model_1d = CartesianDiscreteModel((0, 2), (10,))
        @test get_domain_dimensions(model_1d) ≈ 2.0

        # 2D Model
        model_2d = CartesianDiscreteModel((0, 1, 0, 0.5), (10, 5))
        dims_2d = get_domain_dimensions(model_2d)
        @test dims_2d isa Tuple
        @test dims_2d[1] ≈ 1.0
        @test dims_2d[2] ≈ 0.5

        # 3D Model
        model_3d = CartesianDiscreteModel((0, 1, 0, 1, 0, 1), (2, 2, 2))
        dims_3d = get_domain_dimensions(model_3d)
        @test dims_3d isa Tuple
        @test length(dims_3d) == 3
        @test all(d ≈ 1.0 for d in dims_3d)
    end

    include("drag_flow_validation.jl")
end
