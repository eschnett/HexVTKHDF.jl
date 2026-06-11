# Mesh (de)serialization: every Mesh field round-trips structurally for
# all five patch kinds; node_coordinates equality; locate_point
# agreement on random points; schema-version error.

using HexMeshes
using HexMeshes: locate_point
using HexVTKHDF: FORMAT_VERSION
using HDF5
using Random
using StaticArrays

function _roundtrip(mesh::HexMeshes.Mesh{3,T}, N) where {T}
    path = joinpath(mktempdir(), "mesh.vtkhdf")
    w = VTKHDFWriter(path, node_coordinates(mesh, N); mesh)
    write_step!(w, 0.0;
                fields = ["phi" => zeros(T, N, N, N, mesh.Ne)])
    close(w)
    return VTKHDFFile(path) do f
        read_mesh(f)
    end
end

@testset "meshio" begin
    T = Float64; N = 4
    Random.seed!(3)
    meshes = [
        ("uniform", make_uniform_hex(T, 2, 2, 2, T(0), T(1);
                                     periodic = true)),
        ("warped diagonal", make_warped_uniform_hex(T, 2, T(0), T(1), T(0.05);
                                                    periodic = true)),
        ("warped coupled", make_warped_uniform_hex(T, 2, T(0), T(1), T(0.05);
                                                   periodic = true,
                                                   warp_kind = :coupled)),
        ("cubed cube", make_cubed_cube_mesh(T, 2, T(0.4))),
        ("inflated cube", make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1.0),
                                                  2; outer_bc = :dirichlet)),
        ("radial shell", make_radial_shell_mesh(T, T(1.0), T(2.0), 2;
                                                M_r = 2)),
    ]

    @testset "$name" for (name, mesh) in meshes
        m2 = _roundtrip(mesh, N)
        @test m2.Ne == mesh.Ne
        @test m2.conn.neighbour == mesh.conn.neighbour
        @test m2.conn.neighbour_face == mesh.conn.neighbour_face
        @test m2.conn.orientation == mesh.conn.orientation
        @test m2.conn.bdry == mesh.conn.bdry
        @test m2.vertex_coords == mesh.vertex_coords
        @test m2.vertex_idx == mesh.vertex_idx
        @test m2.patch_id == mesh.patch_id
        @test m2.patch_idx == mesh.patch_idx
        @test m2.patch_element_offset == mesh.patch_element_offset
        @test length(m2.patch_desc) == length(mesh.patch_desc)
        for (pa, pb) in zip(mesh.patch_desc, m2.patch_desc)
            @test pa.kind === pb.kind
            # The active variant must round-trip exactly (isbits +
            # Symbol fields ⇒ structural equality is well-defined).
            @test pa == pb
        end
        # Functional equivalence: identical node coordinates and point
        # location.
        @test node_coordinates(m2, N) == node_coordinates(mesh, N)
        vc = mesh.vertex_coords
        lo = SVector{3,T}(minimum(vc[1, :]), minimum(vc[2, :]),
                          minimum(vc[3, :]))
        hi = SVector{3,T}(maximum(vc[1, :]), maximum(vc[2, :]),
                          maximum(vc[3, :]))
        hits = 0
        for _ in 1:20
            p = lo + rand(T, 3) .* (hi - lo)
            e1, ξ1 = locate_point(mesh, SVector{3,T}(p))
            e2, ξ2 = locate_point(m2, SVector{3,T}(p))
            @test e1 == e2
            e1 == 0 && continue
            hits += 1
            @test maximum(abs, ξ1 - ξ2) < 1e-12
        end
        @test hits > 0
    end

    @testset "schema-version guard" begin
        mesh = make_uniform_hex(T, 1, 1, 1, T(0), T(1); periodic = true)
        path = joinpath(mktempdir(), "future.vtkhdf")
        w = VTKHDFWriter(path, node_coordinates(mesh, N); mesh)
        write_step!(w, 0.0; fields = ["phi" => zeros(T, N, N, N, 1)])
        close(w)
        h5open(path, "r+") do h
            delete_attribute(h["Discretization"], "format_version")
            attrs(h["Discretization"])["format_version"] =
                Int64(FORMAT_VERSION + 1)
        end
        f = VTKHDFFile(path)
        @test_throws ErrorException read_mesh(f)
        close(f)
    end

    @testset "missing /Discretization" begin
        mesh = make_uniform_hex(T, 1, 1, 1, T(0), T(1); periodic = true)
        path = joinpath(mktempdir(), "bare.vtkhdf")
        w = VTKHDFWriter(path, node_coordinates(mesh, N))   # no mesh kwarg
        write_step!(w, 0.0; fields = ["phi" => zeros(T, N, N, N, 1)])
        close(w)
        f = VTKHDFFile(path)
        @test nsteps(f) == 1
        @test_throws ErrorException read_mesh(f)
        close(f)
    end
end
