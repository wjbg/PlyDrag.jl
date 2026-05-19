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

    @testset "Viscosity Models" begin
        # Constant viscosity
        @test viscosity(ConstantViscosity(value=10.0), 1.0, 300.0) ≈ 10.0
        
        # Newtonian (temperature dependent)
        model_newtonian = Newtonian(η = Arrhenius(A=1.0, E=0.0))
        @test viscosity(model_newtonian, 1.0, 300.0) ≈ 1.0

        # PPS (Cross + Arrhenius/WLF)
        η = viscosity(PPS, 1.0, 300.0)
        @test η > 0
    end

    include("drag_flow_validation.jl")
end
