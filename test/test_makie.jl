# Makie extension smoke test (CairoMakie backend): heatmap of a
# FieldSlice, the fieldsliceplot recipe (slice and field entry points),
# and the figure-level plotslice producing a non-empty PNG.

using CairoMakie
using HexMeshes: make_uniform_hex

@testset "makie extension" begin
    T = Float64; N = 4
    mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = false)
    d = Discretization(mesh, N)
    geom = geometry(d)
    data = [sinpi(geom.coords[1, i, j, k, e]) *
            cospi(geom.coords[2, i, j, k, e])
            for i in 1:N, j in 1:N, k in 1:N, e in 1:mesh.Ne]
    f = MeshField(data, d; name = "wave", time = 0.0)
    s = uniform_slice(f; axis = :z, offset = 0.5, res = 32)

    fig1 = Figure()
    hm = heatmap(fig1[1, 1], s)            # convert_arguments path
    @test hm.plot isa Heatmap

    fig2 = Figure()
    p = fieldsliceplot(fig2[1, 1], s; symmetric = true)
    @test !isempty(p.plot.plots)

    fig3 = Figure()
    p3 = fieldsliceplot(fig3[1, 1], f; sliceaxis = :z, offset = 0.5,
                        res = 24)
    @test !isempty(p3.plot.plots)

    fig = plotslice(f; axis = :z, offset = 0.5, res = 32)
    path = joinpath(mktempdir(), "slice.png")
    save(path, fig)
    @test isfile(path) && filesize(path) > 1000
end
