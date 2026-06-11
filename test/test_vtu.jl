# VTU export: writer round-trip through the VTUSeries reader, the
# .pvd/.visit indexes, the metadata sidecar, and the VTKHDF → VTU
# converter.

using HexMeshes: make_radial_shell_mesh, make_uniform_hex
using Random
using TOML

@testset "vtu" begin
    Random.seed!(7)
    T = Float64; N = 4

    @testset "writer → VTUSeries round-trip" begin
        mesh = make_radial_shell_mesh(T, T(1.0), T(2.0), 2; M_r = 2)
        Ne = mesh.Ne
        dir = mktempdir()
        prefix = joinpath(dir, "wave")
        coords = node_coordinates(mesh, N)
        w = VTUWriter(prefix, coords; metadata = (; run = "vtu", M = 2))
        fa = rand(T, N, N, N, Ne)
        fb = rand(T, N, N, N, Ne, 2)            # multi-channel
        write_step!(w, 0.0; fields = ["phi" => fa, "vec" => fb])
        fa2 = rand(T, N, N, N, Ne)
        write_step!(w, 0.5; fields = ["phi" => fa2, "vec" => fb])
        close(w)

        @test isfile(prefix * "_000001.vtu") && isfile(prefix * "_000002.vtu")
        @test isfile(prefix * ".pvd") && isfile(prefix * ".visit")
        @test length(readlines(prefix * ".visit")) == 2
        meta = TOML.parsefile(prefix * "_metadata.toml")
        @test meta["parameters"]["run"] == "vtu"
        @test haskey(meta["package_versions"], "HexSBPSAT")

        for path in (prefix * ".pvd", prefix * ".visit", prefix)
            s = VTUSeries(path)
            @test s isa VTUSeries{T}
            @test nsteps(s) == 2
            @test times(s) == [0.0, 0.5]
            @test field_names(s) == ["phi", "vec_1", "vec_2"]
            @test s.N == N && s.Ne == Ne
        end

        s = VTUSeries(prefix * ".pvd")
        @test coordinates(s) ≈ coords
        @test s["phi", 1] ≈ fa
        @test readfield(s, "phi", 2) ≈ fa2
        @test s["vec_2", 1] ≈ fb[:, :, :, :, 2]
        @test_throws BoundsError s["phi", 3]
        @test_throws ErrorException s["nope", 1]
        @test length(sprint(show, MIME"text/plain"(), s)) < 300

        # single-file open: time recovered from the TimeValue field data
        s1 = VTUSeries(prefix * "_000002.vtu")
        @test nsteps(s1) == 1
        @test times(s1) == [0.5]
        @test s1["phi", 1] ≈ fa2
    end

    @testset "vtkhdf_to_vtu" begin
        mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = true)
        Ne = mesh.Ne
        dir = mktempdir()
        src = joinpath(dir, "run.vtkhdf")
        w = VTKHDFWriter(src, mesh, N; metadata = (; seed = 7))
        f1 = rand(T, N, N, N, Ne)
        f2 = rand(T, N, N, N, Ne)
        write_step!(w, 0.0; fields = ["phi" => f1, "psi" => f2])
        write_step!(w, 1.0; fields = ["phi" => f2, "psi" => f1])
        close(w)

        pvd = vtkhdf_to_vtu(src)
        @test pvd == joinpath(dir, "run.pvd")
        s = VTUSeries(pvd)
        @test nsteps(s) == 2
        @test times(s) == [0.0, 1.0]
        @test field_names(s) == ["phi", "psi"]
        @test s.N == N && s.Ne == Ne
        VTKHDFFile(src) do f
            @test coordinates(s) ≈ coordinates(f)
            @test s["phi", 1] ≈ parent(f["phi", 1])
            @test s["psi", 2] ≈ parent(f["psi", 2])
        end
        @test TOML.parsefile(joinpath(dir, "run_metadata.toml"
                                      ))["parameters"]["seed"] == 7

        # subset selection
        pvd2 = vtkhdf_to_vtu(src, joinpath(dir, "sub");
                             fields = ["psi"], steps = [2])
        s2 = VTUSeries(pvd2)
        @test nsteps(s2) == 1
        @test field_names(s2) == ["psi"]
        @test s2["psi", 1] ≈ f1
    end

    @testset "prefix in a not-yet-existing directory" begin
        mesh = make_uniform_hex(T, 1, 1, 1, T(0), T(1); periodic = true)
        prefix = joinpath(mktempdir(), "new", "dir", "run")
        w = VTUWriter(prefix, mesh, N)
        write_step!(w, 0.0; fields = ["u" => zeros(T, N, N, N, 1)])
        @test isfile(prefix * ".pvd")
    end

    @testset "mesh convenience constructor + Float32" begin
        mesh = make_uniform_hex(Float32, 1, 1, 1, 0.0f0, 1.0f0;
                                periodic = true)
        dir = mktempdir()
        w = VTUWriter(joinpath(dir, "f32"), mesh, N)
        write_step!(w, 0.25; fields = ["u" => rand(Float32, N, N, N, 1)])
        s = VTUSeries(joinpath(dir, "f32"))
        @test s isa VTUSeries{Float32}
        @test times(s) == [0.25]
        @test size(s["u", 1]) == (N, N, N, 1)
    end
end
