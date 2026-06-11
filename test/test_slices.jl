# Slices: a degree-1 polynomial slices exactly on-mesh; NaN outside the
# inflated-cube sphere; element edges; extrema_finite.

using HexMeshes: make_uniform_hex, make_inflated_cube_mesh
using HexSBPSAT: make_element, make_geometry

@testset "slices" begin
    T = Float64; N = 4
    poly(x, y, z) = 1 + 2x - 3y + z / 2

    @testset "exact slice on a unit box" begin
        mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = false)
        d = Discretization(mesh, N)
        geom = geometry(d)
        data = [poly(geom.coords[1, i, j, k, e], geom.coords[2, i, j, k, e],
                     geom.coords[3, i, j, k, e])
                for i in 1:N, j in 1:N, k in 1:N, e in 1:mesh.Ne]
        f = MeshField(data, d; name = "poly", time = 1.5)
        s = uniform_slice(f; axis = :z, offset = 0.25, res = (21, 17))
        @test s.uaxis === :x && s.vaxis === :y
        @test length(s.us) == 21 && length(s.vs) == 17
        @test all(isfinite, s.values)
        worst = maximum(abs(s.values[i, j] - poly(s.us[i], s.vs[j], 0.25))
                        for i in 1:21, j in 1:17)
        @test worst < 1e-12
        @test occursin("poly", s.label) && occursin("1.5", s.label)
        lo, hi = extrema_finite(s)
        @test lo ≈ minimum(s.values) && hi ≈ maximum(s.values)
        # element edges: z = 0.5 is an element-boundary plane
        eu, ev = element_edges(mesh, :z, 0.5)
        @test !isempty(eu) && length(eu) == length(ev)
        @test iseven(length(eu))
        # x-axis slice orientation
        sx = uniform_slice(f; axis = :x, offset = 0.5, res = 11)
        @test sx.uaxis === :y && sx.vaxis === :z
        @test maximum(abs(sx.values[i, j] - poly(0.5, sx.us[i], sx.vs[j]))
                      for i in 1:11, j in 1:11) < 1e-12
    end

    @testset "NaN outside the inflated-cube ball" begin
        mesh = make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1.0), 2;
                                       outer_bc = :dirichlet)
        d = Discretization(mesh, N)
        f = MeshField(ones(T, N, N, N, mesh.Ne), d)
        s = uniform_slice(f; axis = :z, offset = 0.0, res = 41,
                          extent = (-1.2, 1.2, -1.2, 1.2))
        # centre in-mesh, corners outside the unit sphere
        @test s.values[21, 21] ≈ 1
        @test isnan(s.values[1, 1])
        @test isnan(s.values[end, end])
        lo, hi = extrema_finite(s)
        @test lo ≈ 1 ≈ hi
    end
end
