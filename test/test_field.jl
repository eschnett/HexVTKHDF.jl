# MeshField: AbstractArray behavior, broadcast-preserving arithmetic,
# reductions, and the mesh-aware integrate/l2_norm_phys/probe.

using HexMeshes: make_uniform_hex, make_inflated_cube_mesh
using HexSBPSAT: discrete_l2_norm
using StaticArrays

@testset "field" begin
    T = Float64; N = 4
    mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = false)
    d = Discretization(mesh, N)
    geom = geometry(d)
    Ne = mesh.Ne

    # Degree-1 polynomial in the GLL space ⇒ interpolation is exact.
    poly(x, y, z) = 1 + 2x - 3y + z / 2
    data = [poly(geom.coords[1, i, j, k, e], geom.coords[2, i, j, k, e],
                 geom.coords[3, i, j, k, e])
            for i in 1:N, j in 1:N, k in 1:N, e in 1:Ne]
    f = MeshField(data, d; name = "poly", time = 0.0)

    @testset "array interface + reductions" begin
        @test size(f) == (N, N, N, Ne)
        @test f[2, 3, 1, 4] == data[2, 3, 1, 4]
        @test minimum(f) == minimum(data)
        @test maximum(f) == maximum(data)
        @test extrema(f) == extrema(data)
        @test sum(f) ≈ sum(data)
        @test argmax(f) == argmax(data)
        @test vec(f) == vec(data)
    end

    @testset "broadcasting preserves the wrapper" begin
        g = f .+ 1
        @test g isa MeshField{T}
        @test g.disc === d
        @test parent(g) == data .+ 1
        h = 2 .* f .- g
        @test h isa MeshField{T}
        @test parent(h) ≈ 2 .* data .- (data .+ 1)
        # non-dot arithmetic rides broadcasting
        @test (f + f) isa MeshField{T}
        @test parent(f - f) == zero(data)
        @test (2f) isa MeshField{T}
        @test abs.(f) isa MeshField{T}
        # in-place
        h .= f .+ 3
        @test parent(h) == data .+ 3
        # copy/similar keep the handle
        @test copy(f).disc === d
        @test similar(f).disc === d
    end

    @testset "integrate / norms / probe" begin
        one_field = MeshField(ones(T, N, N, N, Ne), d)
        @test integrate(one_field) ≈ 1.0 atol = 1e-12   # unit-box volume
        @test l2_norm_phys(f) ≈ discrete_l2_norm(data, geom,
                                                 operators(d))
        # probe of a degree-1 polynomial is exact
        for p in (SVector(0.3, 0.4, 0.6), SVector(0.91, 0.13, 0.5))
            @test probe(f, p) ≈ poly(p...) atol = 1e-12
        end
        @test probe(f, 0.3, 0.4, 0.6) ≈ poly(0.3, 0.4, 0.6) atol = 1e-12
        @test isnan(probe(f, SVector(2.0, 2.0, 2.0)))   # outside
    end

    @testset "integrate on a curved mesh ≈ ball volume" begin
        meshb = make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1.0), 3;
                                        outer_bc = :dirichlet)
        db = Discretization(meshb, 6)
        ob = MeshField(ones(T, 6, 6, 6, meshb.Ne), db)
        @test integrate(ob) ≈ 4π / 3 rtol = 1e-4
    end
end
