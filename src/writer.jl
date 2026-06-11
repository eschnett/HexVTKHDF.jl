# VTKHDF output: time series of point data on the spectral-element mesh
# in the VTKHDF file format (a pure-HDF5 layout readable by ParaView ≥
# 5.12), written with HDF5.jl — no Julia package currently provides a
# VTKHDF writer.
#
# Geometry: every element's GLL nodes become points (duplicated across
# element interfaces — valid VTK), and each element is subdivided into
# (N−1)³ linear VTK_HEXAHEDRON cells. The mesh is static; time steps
# append point data along the unlimited dimension with the temporal
# "Steps" group bookkeeping of the VTKHDF 2.x specification.
#
# Reproducibility: `metadata` (any TOML-serialisable Dict/NamedTuple —
# the run parameters, the RNG seed) plus the package versions from
# `Pkg.dependencies()` are written as a TOML string to the dataset
# `/Metadata` at file creation.
#
# The writer is physics-agnostic: it takes coordinates and named scalar
# fields of shape (N,N,N,Ne); `(N,N,N,Ne,C)` arrays are written as C
# scalar fields with suffixed names.

const _VTK_HEXAHEDRON = UInt8(12)

mutable struct VTKHDFWriter{T}
    file::HDF5.File
    path::String
    npoints::Int
    nsteps::Int
    fieldnames::Vector{String}
    swmr::Bool          # requested at creation
    swmr_active::Bool   # start_swmr_write has been called
end

"""
    VTKHDFWriter(path, coords::AbstractArray{T,5}; mesh = nothing,
                 metadata = nothing, swmr = true, chunk_mb = 4)
        -> VTKHDFWriter

Create a VTKHDF time-series file at `path` for the static mesh whose GLL
node coordinates are `coords :: (3, N, N, N, Ne)` (host array, e.g.
`geom.coords`). `metadata` (a `Dict`/`NamedTuple` convertible to TOML)
is stored together with the package versions in `/Metadata`. Passing
the `mesh::HexMeshes.Mesh{3,T}` additionally writes the self-describing
`/Discretization` group, from which [`VTKHDFFile`](@ref)/
[`read_mesh`](@ref) reconstruct the full discretization. Use
[`write_step!`](@ref) to append samples and `close(w)` when done.

With `swmr = true` (default) the writer enters HDF5's single-writer/
multiple-readers mode after the first step: other processes can then
open the *live* file with `h5open(path, "r"; swmr = true)` while the
run writes. SWMR forbids attribute modification, so the spec's
`Steps/NSteps` attribute stays frozen at 1 during the run (count
completed steps as `length` of the `Steps/Values` dataset instead) and
is finalised when the writer is closed; for a killed run, repair it
with [`vtkhdf_finalize!`](@ref).
"""
function VTKHDFWriter(path::AbstractString, coords::AbstractArray{T,5};
                      mesh = nothing, metadata = nothing, swmr::Bool = true,
                      chunk_mb::Real = 4) where {T}
    _, N1, N2, N3, Ne = size(coords)
    @assert size(coords, 1) == 3 && N1 == N2 == N3
    N = N1
    npoints = N^3 * Ne
    ncells = (N - 1)^3 * Ne

    # SWMR needs the ≥ v1.10 file format (start_swmr_write rejects the
    # default earliest-compatible superblock).
    file = swmr ? h5open(String(path), "w"; libver_bounds = :latest) :
                  h5open(String(path), "w")
    root = create_group(file, "VTKHDF")
    attrs(root)["Version"] = Int64[2, 2]
    # The Type attribute must be a fixed-length ASCII string for VTK.
    let dtype = HDF5.datatype("UnstructuredGrid")
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        a = create_attribute(root, "Type", dtype, HDF5.dataspace("UnstructuredGrid"))
        write_attribute(a, dtype, "UnstructuredGrid")
        close(a)
    end

    # --- static geometry -------------------------------------------------
    pts = Matrix{T}(undef, 3, npoints)
    p = 0
    @inbounds for e in 1:Ne, k in 1:N, j in 1:N, i in 1:N
        p += 1
        pts[1, p] = coords[1, i, j, k, e]
        pts[2, p] = coords[2, i, j, k, e]
        pts[3, p] = coords[3, i, j, k, e]
    end
    # Julia (3, npoints) column-major ⇒ HDF5 sees (npoints, 3) row-major,
    # which is the VTKHDF Points layout.
    root["Points"] = pts

    nid(i, j, k, e) = (i - 1) + N*(j - 1) + N^2*(k - 1) + N^3*(e - 1)  # 0-based
    conn = Vector{Int64}(undef, 8 * ncells)
    q = 0
    @inbounds for e in 1:Ne, k in 1:N-1, j in 1:N-1, i in 1:N-1
        conn[q+1] = nid(i,   j,   k,   e)
        conn[q+2] = nid(i+1, j,   k,   e)
        conn[q+3] = nid(i+1, j+1, k,   e)
        conn[q+4] = nid(i,   j+1, k,   e)
        conn[q+5] = nid(i,   j,   k+1, e)
        conn[q+6] = nid(i+1, j,   k+1, e)
        conn[q+7] = nid(i+1, j+1, k+1, e)
        conn[q+8] = nid(i,   j+1, k+1, e)
        q += 8
    end
    root["Connectivity"] = conn
    root["Offsets"] = collect(Int64, 0:8:8ncells)
    root["Types"] = fill(_VTK_HEXAHEDRON, ncells)
    root["NumberOfPoints"] = Int64[npoints]
    root["NumberOfCells"] = Int64[ncells]
    root["NumberOfConnectivityIds"] = Int64[length(conn)]

    create_group(root, "PointData")

    # --- temporal bookkeeping (VTKHDF 2.x "Steps" group) ----------------
    steps = create_group(root, "Steps")
    attrs(steps)["NSteps"] = Int64(0)
    chunk1 = (64,)
    for (name, typ) in (("Values", T), ("PartOffsets", Int64),
                        ("NumberOfParts", Int64), ("PointOffsets", Int64),
                        ("ConnectivityIdOffsets", Int64))
        create_dataset(steps, name, typ, ((0,), (-1,)); chunk = chunk1)
    end
    # CellOffsets is (nsteps × 1) per the spec examples.
    create_dataset(steps, "CellOffsets", Int64, ((0, 1), (-1, 1));
                   chunk = (64, 1))
    create_group(steps, "PointDataOffsets")

    # --- reproducibility metadata ----------------------------------------
    meta = Dict{String,Any}()
    meta["parameters"] = metadata === nothing ? Dict{String,Any}() :
                         _toml_ready(metadata)
    deps = Dict{String,Any}()
    for (_, d) in Pkg.dependencies()
        d.version === nothing || (deps[d.name] = string(d.version))
    end
    meta["package_versions"] = deps
    meta["julia_version"] = string(VERSION)
    file["Metadata"] = sprint(io -> TOML.print(io, meta))

    # Self-describing discretization (mesh + SBP element order): with
    # this group the file alone reconstructs mesh/element/operators/
    # geometry — see meshio.jl for the schema.
    mesh === nothing || _write_discretization!(file, mesh, N)

    flush(file)   # geometry + metadata on disk before the first step
    return VTKHDFWriter{T}(file, String(path), npoints, 0, String[],
                           swmr, false)
end

"""
    VTKHDFWriter(path, mesh::Mesh{3,T}, N::Int; kwargs...)

Convenience constructor computing the GLL node coordinates from the
mesh ([`node_coordinates`](@ref)) and writing the self-describing
`/Discretization` group.
"""
VTKHDFWriter(path::AbstractString, mesh::Mesh{3,T}, N::Int;
             kwargs...) where {T} =
    VTKHDFWriter(path, node_coordinates(mesh, N); mesh, kwargs...)

# Make a NamedTuple/Dict TOML-serialisable (symbols → strings, tuples →
# vectors, unsupported types → string).
function _toml_ready(x)
    x isa NamedTuple && return _toml_ready(Dict(string(k) => v for (k, v) in pairs(x)))
    x isa AbstractDict &&
        return Dict(string(k) => _toml_ready(v) for (k, v) in x)
    x isa Tuple && return [_toml_ready(v) for v in x]
    x isa AbstractVector && return [_toml_ready(v) for v in x]
    x isa Symbol && return string(x)
    x isa Union{Bool,Integer,AbstractFloat,AbstractString} &&
        return x isa AbstractFloat ? Float64(x) : x
    return string(x)
end

# Append `n` entries to a 1-D unlimited dataset.
function _append1!(dset, vals::AbstractVector)
    old = size(dset, 1)
    HDF5.set_extent_dims(dset, (old + length(vals),))
    dset[old+1:old+length(vals)] = vals
    return nothing
end

"""
    write_step!(w::VTKHDFWriter, t; fields) -> w

Append one time sample. `fields` is an iterable of `name => array`
pairs with arrays of shape `(N, N, N, Ne)` (one scalar field) or
`(N, N, N, Ne, C)` (written as `name_1 … name_C`); host arrays of the
writer's scalar type. Field names must be identical at every step.
"""
function write_step!(w::VTKHDFWriter{T}, t::Real; fields) where {T}
    root = w.file["VTKHDF"]
    pd = root["PointData"]
    steps = root["Steps"]
    pdo = steps["PointDataOffsets"]

    # Flatten multi-channel arrays into scalar fields.
    flat = Vector{Pair{String,Any}}()
    for (name, a) in fields
        if ndims(a) == 4
            push!(flat, String(name) => a)
        elseif ndims(a) == 5
            for c in 1:size(a, 5)
                push!(flat, "$(name)_$c" => view(a, :, :, :, :, c))
            end
        else
            error("write_step!: field $name must be 4- or 5-dimensional")
        end
    end
    names = first.(flat)
    if w.nsteps == 0
        w.fieldnames = names
        for nm in names
            create_dataset(pd, nm, T, ((0,), (-1,));
                           chunk = (min(w.npoints, 1 << 16),))
            create_dataset(pdo, nm, Int64, ((0,), (-1,)); chunk = (64,))
        end
    else
        names == w.fieldnames ||
            error("write_step!: field names changed between steps")
    end

    for (nm, a) in flat
        length(a) == w.npoints ||
            error("write_step!: field $nm has $(length(a)) values, " *
                  "expected $(w.npoints)")
        _append1!(pd[nm], vec(Array{T}(a)))
        _append1!(pdo[nm], Int64[w.nsteps * w.npoints])
    end

    _append1!(steps["Values"], T[t])
    _append1!(steps["PartOffsets"], Int64[w.nsteps])
    _append1!(steps["NumberOfParts"], Int64[1])
    _append1!(steps["PointOffsets"], Int64[0])
    _append1!(steps["ConnectivityIdOffsets"], Int64[0])
    co = steps["CellOffsets"]
    HDF5.set_extent_dims(co, (w.nsteps + 1, 1))
    co[w.nsteps + 1, 1] = 0
    w.nsteps += 1
    # SWMR forbids attribute modification, so NSteps is written only
    # while we are not yet in SWMR mode (i.e. once, at the first step);
    # close()/vtkhdf_finalize! set the final value.
    w.swmr_active || (attrs(steps)["NSteps"] = Int64(w.nsteps))
    # Flush metadata + data to disk after every step. Without this the
    # on-disk file is just a superblock until close(): unreadable from
    # other nodes (parallel filesystems do not share the writer's page
    # cache) and entirely lost if the job is killed. With the per-step
    # flush, a killed run loses at most the step in progress, and a
    # *snapshot copy* of the file (`cp run.vtkhdf snap.vtkhdf`) taken
    # between steps opens cleanly for live inspection. Reading the
    # live file in place while the writer holds it open is still not
    # guaranteed by HDF5 (no SWMR — the temporal NSteps attribute is
    # rewritten every step, which SWMR forbids); copy first.
    flush(w.file)
    # Enter SWMR mode once all datasets exist (they are created at the
    # first step; SWMR allows only dataset extension/writes afterwards).
    if w.swmr && !w.swmr_active
        HDF5.start_swmr_write(w.file)
        w.swmr_active = true
    end
    return w
end

function Base.close(w::VTKHDFWriter)
    close(w.file)
    # Finalise the NSteps attribute that SWMR mode kept frozen.
    if w.swmr_active && w.nsteps > 1
        h5open(w.path, "r+") do f
            attrs(f["VTKHDF"]["Steps"])["NSteps"] = Int64(w.nsteps)
        end
    end
    return nothing
end

"""
    vtkhdf_finalize!(path) -> Int

Repair the `Steps/NSteps` attribute of a VTKHDF file from the length of
the `Steps/Values` dataset. Needed for files from runs that were killed
before `close` (the SWMR writer keeps the attribute frozen during the
run); harmless on healthy files. Returns the number of steps.
"""
function vtkhdf_finalize!(path::AbstractString)
    nsteps = h5open(path, "r+") do f
        steps = f["VTKHDF"]["Steps"]
        n = length(read(steps["Values"]))
        attrs(steps)["NSteps"] = Int64(n)
        n
    end
    return nsteps
end
