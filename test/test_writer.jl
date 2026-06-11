# Writer: structural layout, exact field round-trip, TOML metadata,
# SWMR live read, killed-run repair, snapshot copy. (Ported from
# GeneralizedHarmonicSecondOrder2's io tests.)

using HexMeshes: make_uniform_hex
using HexSBPSAT: make_element, make_geometry
using HDF5
using TOML

@testset "writer" begin
    T = Float64; N = 4
    mesh = make_uniform_hex(T, 2, 2, 2, T(0), T(1); periodic = true)
    elem = make_element(T, N)
    geom = make_geometry(mesh, elem)
    Ne = mesh.Ne
    npoints = N^3 * Ne
    ncells = (N - 1)^3 * Ne

    @testset "round-trip + structure" begin
        path = joinpath(mktempdir(), "out.vtkhdf")
        w = VTKHDFWriter(path, geom.coords;
                         metadata = (; M = 2, bc = :periodic, seed = 11))
        f1 = rand(T, N, N, N, Ne)
        f2 = rand(T, N, N, N, Ne, 3)
        write_step!(w, 0.0; fields = ["phi" => f1, "vec" => f2])
        f1b = rand(T, N, N, N, Ne)
        write_step!(w, 0.5; fields = ["phi" => f1b, "vec" => f2])
        close(w)

        h5open(path, "r") do h
            root = h["VTKHDF"]
            @test read(attributes(root)["Version"]) == [2, 2]
            @test read(attributes(root)["Type"]) == "UnstructuredGrid"
            @test size(read(root["Points"])) == (3, npoints)
            conn = read(root["Connectivity"])
            @test length(conn) == 8 * ncells
            @test extrema(conn) == (0, npoints - 1)
            @test read(root["Offsets"])[end] == 8 * ncells
            @test all(read(root["Types"]) .== 12)
            steps = root["Steps"]
            @test read(attributes(steps)["NSteps"]) == 2
            @test read(steps["Values"]) == [0.0, 0.5]
            pd = root["PointData"]
            @test sort(keys(pd)) == sort(["phi", "vec_1", "vec_2", "vec_3"])
            phi = read(pd["phi"])
            @test length(phi) == 2 * npoints
            @test phi[1:npoints] == vec(f1)
            @test phi[npoints+1:end] == vec(f1b)
            @test read(steps["PointDataOffsets"]["phi"]) == [0, npoints]
            meta = TOML.parse(read(h["Metadata"]))
            @test meta["parameters"]["M"] == 2
            @test meta["parameters"]["bc"] == "periodic"
            @test haskey(meta["package_versions"], "HexSBPSAT")
            @test haskey(meta, "julia_version")
        end
    end

    @testset "SWMR live read, repair, snapshot" begin
        dir = mktempdir()
        path = joinpath(dir, "live.vtkhdf")
        w = VTKHDFWriter(path, geom.coords)
        f1 = rand(T, N, N, N, Ne)
        write_step!(w, 0.0; fields = ["phi" => f1])
        write_step!(w, 0.5; fields = ["phi" => f1])
        h5open(path, "r"; swmr = true) do h
            steps = h["VTKHDF"]["Steps"]
            @test length(read(steps["Values"])) == 2
            @test read(attributes(steps)["NSteps"]) == 1   # frozen under SWMR
            @test length(read(h["VTKHDF"]["PointData"]["phi"])) == 2 * npoints
        end
        close(w)
        h5open(path, "r") do h
            @test read(attributes(h["VTKHDF"]["Steps"])["NSteps"]) == 2
        end

        path2 = joinpath(dir, "killed.vtkhdf")
        w2 = VTKHDFWriter(path2, geom.coords)
        write_step!(w2, 0.0; fields = ["phi" => f1])
        write_step!(w2, 0.5; fields = ["phi" => f1])
        close(w2.file)                      # simulate a kill (no finalise)
        @test vtkhdf_finalize!(path2) == 2
        h5open(path2, "r") do h
            @test read(attributes(h["VTKHDF"]["Steps"])["NSteps"]) == 2
        end

        path3 = joinpath(dir, "plain.vtkhdf")
        w3 = VTKHDFWriter(path3, geom.coords; swmr = false)
        write_step!(w3, 0.0; fields = ["phi" => f1])
        write_step!(w3, 0.5; fields = ["phi" => f1])
        snap = joinpath(dir, "snap.vtkhdf")
        cp(path3, snap)
        h5open(snap, "r") do h
            @test read(attributes(h["VTKHDF"]["Steps"])["NSteps"]) == 2
        end
        close(w3)
    end
end
