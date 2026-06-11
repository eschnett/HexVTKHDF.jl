# Reader: write → read equality of fields/times/coordinates; metadata
# and versions; Float32; SWMR live read; show methods.

using HexMeshes: make_radial_shell_mesh
using Random

@testset "reader" begin
    Random.seed!(5)

    @testset "T = $T" for T in (Float64, Float32)
        N = 4
        mesh = make_radial_shell_mesh(T, T(1.0), T(2.0), 2; M_r = 2)
        Ne = mesh.Ne
        path = joinpath(mktempdir(), "rw.vtkhdf")
        coords = node_coordinates(mesh, N)
        w = VTKHDFWriter(path, coords; mesh,
                         metadata = (; run = "test", seed = 5))
        fa = rand(T, N, N, N, Ne)
        fb = rand(T, N, N, N, Ne)
        write_step!(w, 0.0; fields = ["phi" => fa, "psi" => fb])
        fa2 = rand(T, N, N, N, Ne)
        write_step!(w, 0.25; fields = ["phi" => fa2, "psi" => fb])
        close(w)

        VTKHDFFile(path) do f
            @test f isa VTKHDFFile{T}
            @test nsteps(f) == 2
            @test times(f) == T[0.0, 0.25]
            @test field_names(f) == ["phi", "psi"]
            @test f.N == N && f.Ne == Ne
            @test coordinates(f) == coords
            @test metadata(f)["parameters"]["run"] == "test"
            v = versions(f)
            @test v.format == 1
            @test VersionNumber(v.hexmeshes) isa VersionNumber

            g1 = f["phi", 1]
            @test g1 isa MeshField{T}
            @test parent(g1) == fa
            @test g1.name == "phi" && g1.time == T(0)
            g2 = readfield(f, "phi", 2)
            @test parent(g2) == fa2
            @test parent(f["psi", 2]) == fb
            @test_throws BoundsError f["phi", 3]
            @test_throws ErrorException f["nope", 1]
            # show methods do not error and stay compact
            @test length(sprint(show, MIME"text/plain"(), f)) < 500
            @test length(sprint(show, MIME"text/plain"(), g1)) < 200
        end
    end

    @testset "SWMR live read" begin
        T = Float64; N = 4
        mesh = make_radial_shell_mesh(T, T(1.0), T(2.0), 2; M_r = 2)
        path = joinpath(mktempdir(), "live.vtkhdf")
        w = VTKHDFWriter(path, mesh, N)
        f1 = rand(T, N, N, N, mesh.Ne)
        write_step!(w, 0.0; fields = ["phi" => f1])
        write_step!(w, 0.5; fields = ["phi" => f1])
        VTKHDFFile(path; swmr = true) do f
            @test nsteps(f) == 2          # Values-based, not the attribute
            @test parent(f["phi", 2]) == f1
            @test read_mesh(f).Ne == mesh.Ne
        end
        close(w)
    end
end
