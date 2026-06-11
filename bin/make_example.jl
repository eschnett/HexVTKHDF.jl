#!/usr/bin/env julia
# Create a small example VTKHDF file: an outgoing spherical wave on a
# coarse radial-shell mesh, sampled at a few time steps, with the
# /Discretization group (the file is fully self-describing) and TOML
# metadata. Reopens the file with the reader and prints a summary.
# Useful as reader/ParaView test input and as a writer-API example.
#
# Usage:
#     julia --project=<HexVTKHDF> bin/make_example.jl [path]
#
# Default path: example.vtkhdf in the current directory.

using HexMeshes: make_radial_shell_mesh
using HexVTKHDF

function main(path::AbstractString = "example.vtkhdf")
    T = Float64
    R1, R2 = T(1), T(3)             # shell radii
    M, M_r, N = 2, 3, 5             # angular/radial elements, GLL nodes
    nt = 5                          # time samples over one period

    mesh = make_radial_shell_mesh(T, R1, R2, M; M_r)
    coords = node_coordinates(mesh, N)
    r = dropdims(sqrt.(sum(abs2, coords; dims = 1)); dims = 1)

    # Outgoing spherical wave (one wavelength across the shell).
    k = T(2π) / (R2 - R1)
    phi(r, t) = sin(k * (r - t)) / r
    dtphi(r, t) = -k * cos(k * (r - t)) / r

    w = VTKHDFWriter(path, coords; mesh,
                     metadata = (; description = "HexVTKHDF example: " *
                                     "outgoing spherical wave on a radial shell",
                                 R1, R2, M, M_r, N, k))
    for t in range(T(0), T(1); length = nt)
        write_step!(w, t; fields = ["phi" => phi.(r, t),
                                    "dtphi" => dtphi.(r, t)])
    end
    close(w)

    VTKHDFFile(path) do f
        show(stdout, MIME"text/plain"(), f)
        println()
        println("mesh:     Ne = ", read_mesh(f).Ne, ", N = ", f.N)
        println("versions: ", versions(f))
        println("size:     ", filesize(path), " bytes")
    end
    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? "example.vtkhdf" : ARGS[1])
end
