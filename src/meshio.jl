# The self-describing `/Discretization` group: serialization of the
# complete `Mesh{3,T}` state plus the SBP element order N, with schema
# and package versions, so the file alone reconstructs mesh, element,
# operators, and geometry (`make_element`/`make_operators`/
# `make_geometry` are deterministic in (T, N, mesh)).
#
# Layout (all dataset/attribute names mirror the Julia field names):
#
#   /Discretization                 attrs: format_version, scalar_type,
#                                   N, D, hexvtkhdf_version,
#                                   hexmeshes_version, hexsbpsat_version,
#                                   julia_version
#     /Mesh                         attrs: Ne, Nv
#       neighbour (6,Ne) Int32; neighbour_face/orientation/bdry (6,Ne) Int8
#       vertex_coords (3,Nv) T; vertex_idx (8,Ne) Int64
#       patch_id (Ne) Int32; patch_idx (3,Ne) Int32
#       patch_element_offset (npatches+1) Int64
#       /Patches                    attr: npatches
#         /1, /2, …                 attrs: kind (String), dims (Int64[3]),
#                                   variant fields under their Julia names
#                                   (dir as Int64, warp_kind as String)

"Schema version of the `/Discretization` group this package writes."
const FORMAT_VERSION = 1

const _SCALAR_TYPES = Dict("Float64" => Float64, "Float32" => Float32)

_kind_string(k) = string(k)   # PatchKind enum prints as "Cubic" etc.

function _write_patch!(g::HDF5.Group, pd::PatchDesc{3,T}) where {T}
    a = attrs(g)
    a["kind"] = _kind_string(pd.kind)
    if pd.kind === Cubic
        c = pd.cubic
        a["dims"] = collect(Int64, c.dims)
        a["x_lo"] = collect(T, c.x_lo)
        a["x_hi"] = collect(T, c.x_hi)
    elseif pd.kind === Wedge
        w = pd.wedge
        a["dims"] = collect(Int64, w.dims)
        a["dir"] = Int64(w.dir)
        for (nm, v) in (("a_lo", w.a_lo), ("a_hi", w.a_hi), ("b_lo", w.b_lo),
                        ("b_hi", w.b_hi), ("c_lo", w.c_lo), ("c_hi", w.c_hi),
                        ("R1", w.R1), ("R2", w.R2))
            a[nm] = v
        end
    elseif pd.kind === Inflation
        w = pd.inflation
        a["dims"] = collect(Int64, w.dims)
        a["dir"] = Int64(w.dir)
        for (nm, v) in (("a_lo", w.a_lo), ("a_hi", w.a_hi), ("b_lo", w.b_lo),
                        ("b_hi", w.b_hi), ("c_lo", w.c_lo), ("c_hi", w.c_hi),
                        ("L", w.L), ("R1", w.R1))
            a[nm] = v
        end
    elseif pd.kind === Shell
        w = pd.shell
        a["dims"] = collect(Int64, w.dims)
        a["dir"] = Int64(w.dir)
        for (nm, v) in (("a_lo", w.a_lo), ("a_hi", w.a_hi), ("b_lo", w.b_lo),
                        ("b_hi", w.b_hi), ("c_lo", w.c_lo), ("c_hi", w.c_hi),
                        ("R1", w.R1), ("R2", w.R2))
            a[nm] = v
        end
    elseif pd.kind === WarpedCubic
        w = pd.warped_cubic
        a["dims"] = collect(Int64, w.dims)
        a["x_lo"] = collect(T, w.x_lo)
        a["x_hi"] = collect(T, w.x_hi)
        a["amplitude"] = w.amplitude
        a["warp_kind"] = string(w.warp_kind)
    else
        error("HexVTKHDF: unknown patch kind $(pd.kind)")
    end
    return nothing
end

function _read_patch(g::Union{HDF5.Group,HDF5.File}, ::Type{T}) where {T}
    a = attrs(g)
    kind = a["kind"]
    dims = NTuple{3,Int}(a["dims"])
    rd(nm) = T(a[nm])
    if kind == "Cubic"
        return PatchDesc(PatchCubic{3,T}(dims, NTuple{3,T}(a["x_lo"]),
                                         NTuple{3,T}(a["x_hi"])))
    elseif kind == "Wedge"
        return PatchDesc(PatchWedge{3,T}(dims, Int8(a["dir"]),
                                         rd("a_lo"), rd("a_hi"), rd("b_lo"),
                                         rd("b_hi"), rd("c_lo"), rd("c_hi"),
                                         rd("R1"), rd("R2")))
    elseif kind == "Inflation"
        return PatchDesc(PatchInflation{3,T}(dims, Int8(a["dir"]),
                                             rd("a_lo"), rd("a_hi"), rd("b_lo"),
                                             rd("b_hi"), rd("c_lo"), rd("c_hi"),
                                             rd("L"), rd("R1")))
    elseif kind == "Shell"
        return PatchDesc(PatchShell{3,T}(dims, Int8(a["dir"]),
                                         rd("a_lo"), rd("a_hi"), rd("b_lo"),
                                         rd("b_hi"), rd("c_lo"), rd("c_hi"),
                                         rd("R1"), rd("R2")))
    elseif kind == "WarpedCubic"
        return PatchDesc(PatchWarpedCubic{3,T}(dims,
                                               NTuple{3,T}(a["x_lo"]),
                                               NTuple{3,T}(a["x_hi"]),
                                               rd("amplitude"),
                                               Symbol(a["warp_kind"])))
    else
        error("HexVTKHDF: unknown patch kind \"$kind\" — written by a newer " *
              "schema or package version? (see /Discretization attributes)")
    end
end

_pkgver(m::Module) = string(something(Pkg.pkgversion(m), "unknown"))

function _write_discretization!(file::HDF5.File, mesh::Mesh{3,T},
                                N::Int) where {T}
    haskey(_SCALAR_TYPES, string(T)) ||
        error("HexVTKHDF: scalar type $T is not supported by schema " *
              "version $FORMAT_VERSION (supported: " *
              "$(join(keys(_SCALAR_TYPES), ", ")))")
    d = create_group(file, "Discretization")
    a = attrs(d)
    a["format_version"] = Int64(FORMAT_VERSION)
    a["scalar_type"] = string(T)
    a["N"] = Int64(N)
    a["D"] = Int64(3)
    a["hexvtkhdf_version"] = _pkgver(HexVTKHDF)
    a["hexmeshes_version"] = _pkgver(HexMeshes)
    a["hexsbpsat_version"] = _pkgver(HexSBPSAT)
    a["julia_version"] = string(VERSION)

    m = create_group(d, "Mesh")
    attrs(m)["Ne"] = Int64(mesh.Ne)
    attrs(m)["Nv"] = Int64(size(mesh.vertex_coords, 2))
    m["neighbour"] = Matrix{Int32}(mesh.conn.neighbour)
    m["neighbour_face"] = Matrix{Int8}(mesh.conn.neighbour_face)
    m["orientation"] = Matrix{Int8}(mesh.conn.orientation)
    m["bdry"] = Matrix{Int8}(mesh.conn.bdry)
    m["vertex_coords"] = mesh.vertex_coords
    m["vertex_idx"] = Matrix{Int64}(mesh.vertex_idx)
    m["patch_id"] = mesh.patch_id
    m["patch_idx"] = mesh.patch_idx
    m["patch_element_offset"] = Vector{Int64}(mesh.patch_element_offset)

    ps = create_group(m, "Patches")
    attrs(ps)["npatches"] = Int64(length(mesh.patch_desc))
    for (i, pd) in enumerate(mesh.patch_desc)
        _write_patch!(create_group(ps, string(i)), pd)
    end
    return nothing
end

# Returns (mesh, N, T). Throws descriptive errors for unknown schema
# versions / scalar types.
function _read_discretization(file::HDF5.File)
    haskey(file, "Discretization") ||
        error("HexVTKHDF: this file has no /Discretization group (it was " *
              "written without the `mesh` argument); the mesh cannot be " *
              "reconstructed")
    d = file["Discretization"]
    a = attrs(d)
    fv = a["format_version"]
    fv <= FORMAT_VERSION ||
        error("HexVTKHDF: file has /Discretization format version $fv, " *
              "this package reads ≤ $FORMAT_VERSION. The file was written " *
              "with HexVTKHDF $(a["hexvtkhdf_version"]), HexMeshes " *
              "$(a["hexmeshes_version"]) — upgrade this package.")
    st = a["scalar_type"]
    haskey(_SCALAR_TYPES, st) ||
        error("HexVTKHDF: unsupported scalar type \"$st\"")
    T = _SCALAR_TYPES[st]
    N = Int(a["N"])

    m = d["Mesh"]
    Ne = Int(attrs(m)["Ne"])
    conn = MeshConnectivity{3,Matrix{Int32},Matrix{Int8}}(
        read(m["neighbour"]), read(m["neighbour_face"]),
        read(m["orientation"]), read(m["bdry"]))
    vertex_coords = Matrix{T}(read(m["vertex_coords"]))
    vertex_idx = Matrix{Int}(read(m["vertex_idx"]))
    patch_id = Vector{Int32}(read(m["patch_id"]))
    patch_idx = Matrix{Int32}(read(m["patch_idx"]))
    patch_element_offset = Vector{Int}(read(m["patch_element_offset"]))

    ps = m["Patches"]
    npatches = Int(attrs(ps)["npatches"])
    patch_desc = PatchDesc{3,T}[_read_patch(ps[string(i)], T)
                                for i in 1:npatches]

    mesh = Mesh{3,T}(Ne, conn, vertex_coords, vertex_idx;
                     patch_id, patch_idx, patch_desc, patch_element_offset)
    return mesh, N, T
end

"""
    node_coordinates(mesh::Mesh{3,T}, N::Int) -> Array{T,5}

The physical coordinates of the `N³` GLL nodes of every element,
shape `(3, N, N, N, Ne)` — the `coords` argument of
[`VTKHDFWriter`](@ref) when no `MeshGeometry` is at hand. Threaded over
elements.
"""
function node_coordinates(mesh::Mesh{3,T}, N::Int) where {T}
    elem = make_element(T, N)
    xs = collect(T, elem.xs)
    Ne = mesh.Ne
    coords = Array{T,5}(undef, 3, N, N, N, Ne)
    Threads.@threads :static for e in 1:Ne
        @inbounds for k in 1:N, j in 1:N, i in 1:N
            P, _ = element_point_and_jac(mesh, e,
                                         SVector{3,T}(xs[i], xs[j], xs[k]))
            coords[1, i, j, k, e] = P[1]
            coords[2, i, j, k, e] = P[2]
            coords[3, i, j, k, e] = P[3]
        end
    end
    return coords
end
