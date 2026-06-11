# XML VTK (.vtu) time-series export and reading — for visualization
# tools without a VTKHDF reader (e.g. VisIt). One .vtu file per step
# plus two index files, rewritten after every step so a running series
# can be opened live:
#
#   <prefix>.pvd     ParaView collection (carries the sample times)
#   <prefix>.visit   VisIt index (one filename per line)
#
# Unlike a mesh-bearing VTKHDF file, a .vtu series is not
# self-describing: it carries the subdivided visualization geometry,
# the (N, Ne) layout (as FieldData), and the point data — but not the
# Mesh/patch state, so nothing can be reconstructed from it. The VTKHDF
# file remains the primary format; [`vtkhdf_to_vtu`](@ref) converts.

"""
    VTUWriter(prefix, coords::AbstractArray{T,5}; metadata = nothing,
              compress = true) -> VTUWriter
    VTUWriter(prefix, mesh::Mesh{3,T}, N::Int; kwargs...)

Create an XML VTK UnstructuredGrid time series: [`write_step!`](@ref)
writes `<prefix>_NNNNNN.vtu` per step and keeps the `<prefix>.pvd`
(ParaView) and `<prefix>.visit` (VisIt) index files current. `coords ::
(3, N, N, N, Ne)` are the GLL node coordinates; each element becomes
`(N−1)³` linear hexahedra, exactly as in [`VTKHDFWriter`](@ref).
`metadata` is written to the sidecar `<prefix>_metadata.toml` with the
same reproducibility payload as a VTKHDF `/Metadata` dataset. Sample
times are stored per file as the `TimeValue` (ParaView) and `TIME`
(VisIt) field-data arrays, and in the `.pvd`.
"""
mutable struct VTUWriter{T}
    prefix::String                     # path prefix (no extension)
    pts::Matrix{T}                     # (3, npoints), static
    cells::Vector{WriteVTK.MeshCell{WriteVTK.VTKCellTypes.VTKCellType,
                                    Vector{Int64}}}
    npoints::Int
    N::Int                             # 0 when constructed from raw points
    Ne::Int
    nsteps::Int
    fieldnames::Vector{String}
    times::Vector{Float64}
    files::Vector{String}              # basenames, time order
    compress::Bool
end

_strip_vtu_ext(p::String) =
    foldl((q, ext) -> endswith(q, ext) ? q[1:end-length(ext)] : q,
          (".pvd", ".vtu", ".visit"); init = p)

# Shared low-level constructor: raw points + 0-based hex connectivity.
function _vtu_writer(prefix::AbstractString, pts::Matrix{T},
                     conn::Vector{Int64}, N::Int, Ne::Int;
                     compress::Bool = true) where {T}
    ncells = length(conn) ÷ 8
    cells = [WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON,
                               conn[8c-7:8c] .+ 1) for c in 1:ncells]
    dir = dirname(prefix)
    isempty(dir) || mkpath(dir)
    return VTUWriter{T}(_strip_vtu_ext(String(prefix)), pts, cells,
                        size(pts, 2), N, Ne, 0, String[], Float64[],
                        String[], compress)
end

function VTUWriter(prefix::AbstractString, coords::AbstractArray{T,5};
                   metadata = nothing, compress::Bool = true) where {T}
    _, N1, N2, N3, Ne = size(coords)
    @assert size(coords, 1) == 3 && N1 == N2 == N3
    N = N1
    w = _vtu_writer(prefix, _vtk_points(coords),
                    _subcell_connectivity(N, Ne), N, Ne; compress)
    metadata === nothing ||
        write(w.prefix * "_metadata.toml", _metadata_toml(metadata))
    return w
end

VTUWriter(prefix::AbstractString, mesh::Mesh{3,T}, N::Int;
          kwargs...) where {T} =
    VTUWriter(prefix, node_coordinates(mesh, N); kwargs...)

"""
    write_step!(w::VTUWriter, t; fields) -> w

Append one time sample as `<prefix>_NNNNNN.vtu` and refresh the
`.pvd`/`.visit` indexes. `fields` as for the VTKHDF
[`write_step!`](@ref) — `name => array` pairs of shape `(N,N,N,Ne)`,
`(N,N,N,Ne,C)` (suffixed scalar fields), or flat `npoints` vectors.
"""
function write_step!(w::VTUWriter{T}, t::Real; fields) where {T}
    flat = _flatten_fields(fields)
    names = first.(flat)
    if w.nsteps == 0
        w.fieldnames = names
    else
        names == w.fieldnames ||
            error("write_step!: field names changed between steps")
    end

    fname = w.prefix * "_" * lpad(w.nsteps + 1, 6, '0')
    vtk = WriteVTK.vtk_grid(fname, w.pts, w.cells; compress = w.compress)
    for (nm, a) in flat
        length(a) == w.npoints ||
            error("write_step!: field $nm has $(length(a)) values, " *
                  "expected $(w.npoints)")
        vtk[nm] = vec(Array{T}(a))
    end
    vtk["TimeValue", WriteVTK.VTKFieldData()] = Float64(t)
    vtk["TIME", WriteVTK.VTKFieldData()] = Float64(t)
    if w.N > 0
        vtk["HexVTKHDF_N", WriteVTK.VTKFieldData()] = Int64(w.N)
        vtk["HexVTKHDF_Ne", WriteVTK.VTKFieldData()] = Int64(w.Ne)
    end
    saved = WriteVTK.vtk_save(vtk)

    w.nsteps += 1
    push!(w.times, Float64(t))
    push!(w.files, basename(saved[1]))
    _write_vtu_indexes(w)
    return w
end

function _write_vtu_indexes(w::VTUWriter)
    open(w.prefix * ".pvd", "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" ",
                "byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (t, f) in zip(w.times, w.files)
            println(io, "    <DataSet timestep=\"", t,
                    "\" group=\"\" part=\"0\" file=\"", f, "\"/>")
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
    open(w.prefix * ".visit", "w") do io
        foreach(f -> println(io, f), w.files)
    end
    return nothing
end

# The per-step files and indexes are complete after every write_step!.
Base.close(w::VTUWriter) = nothing

"""
    vtkhdf_to_vtu(src, prefix = splitext(src)[1];
                  fields = nothing, steps = nothing, compress = true)
        -> "<prefix>.pvd"
    vtkhdf_to_vtu(f::VTKHDFFile, prefix; ...)

Convert a VTKHDF time series (e.g. a GH run output) to a `.vtu` series
for tools without a VTKHDF reader (VisIt). Copies the stored points and
connectivity verbatim, so it works with or without the
`/Discretization` group; `/Metadata` is copied to
`<prefix>_metadata.toml`. `fields`/`steps` select a subset (default:
everything).
"""
vtkhdf_to_vtu(src::AbstractString,
              prefix::AbstractString = first(splitext(src)); kwargs...) =
    VTKHDFFile(f -> vtkhdf_to_vtu(f, prefix; kwargs...), src)

function vtkhdf_to_vtu(f::VTKHDFFile{T}, prefix::AbstractString;
                       fields = nothing, steps = nothing,
                       compress::Bool = true) where {T}
    root = f.file["VTKHDF"]
    all(read(root["Types"]) .== _VTK_HEXAHEDRON) ||
        error("vtkhdf_to_vtu: only pure-hexahedron files are supported")
    w = _vtu_writer(prefix, read(root["Points"]),
                    Vector{Int64}(read(root["Connectivity"])),
                    f.N, f.Ne; compress)
    haskey(f.file, "Metadata") &&
        write(w.prefix * "_metadata.toml", read(f.file["Metadata"]))

    names = fields === nothing ? field_names(f) :
            String[String(n) for n in fields]
    ts = times(f)
    for s in (steps === nothing ? (1:nsteps(f)) : steps)
        write_step!(w, ts[s];
                    fields = [nm => _raw_field(f, nm, s) for nm in names])
    end
    return w.prefix * ".pvd"
end

"""
    VTUSeries(path) -> VTUSeries

Open a `.vtu` time series for reading: `path` is the `.pvd` index, the
`.visit` index, a series prefix, or a single `.vtu` file. Access mirrors
[`VTKHDFFile`](@ref): [`nsteps`](@ref), [`times`](@ref),
[`field_names`](@ref), [`coordinates`](@ref), and `s[name, step]` (≡
[`readfield`](@ref)). Fields written by [`VTUWriter`](@ref) come back
shaped `(N, N, N, Ne)`; without the layout field data they stay flat
`npoints` vectors. A `.vtu` series carries no `/Discretization`
equivalent, so — unlike the VTKHDF reader — fields are plain arrays,
not [`MeshField`](@ref)s.
"""
struct VTUSeries{T}
    path::String
    files::Vector{String}              # full paths, time order
    times::Vector{Float64}
    npoints::Int
    N::Int                             # 0 when unknown
    Ne::Int
    names::Vector{String}
end

function VTUSeries(path::AbstractString)
    path = String(abspath(path))
    isfile(path) || isfile(path * ".pvd") ||
        error("HexVTKHDF: no such file $path (nor $path.pvd)")
    isfile(path) || (path = path * ".pvd")
    dir = dirname(path)
    local files::Vector{String}, ts::Vector{Float64}
    if endswith(path, ".pvd")
        pvd = ReadVTK.PVDFile(path)
        files = [joinpath(dir, d, f)
                 for (d, f) in zip(pvd.directories, pvd.vtk_filenames)]
        ts = Float64.(pvd.timesteps)
    elseif endswith(path, ".visit")
        files = [joinpath(dir, l) for l in strip.(eachline(path))
                 if !isempty(l) && !startswith(l, "!")]
        ts = Float64[]                 # filled from TIME field data below
    elseif endswith(path, ".vtu")
        files = [path]
        ts = Float64[]
    else
        error("HexVTKHDF: cannot open $path as a .vtu series " *
              "(expected .pvd, .visit, .vtu, or a series prefix)")
    end
    isempty(files) && error("HexVTKHDF: empty series $path")

    if isempty(ts)
        ts = map(files) do fn
            fd = ReadVTK.get_field_data(ReadVTK.VTKFile(fn))
            fd !== nothing && "TimeValue" in keys(fd) ?
                Float64(only(ReadVTK.get_data(fd["TimeValue"]))) : NaN
        end
    end

    f1 = ReadVTK.VTKFile(files[1])
    pts = ReadVTK.get_points(f1)
    T = eltype(pts)
    names = sort(collect(keys(ReadVTK.get_point_data(f1))))
    N = 0; Ne = 0
    fd = ReadVTK.get_field_data(f1)
    if fd !== nothing && "HexVTKHDF_N" in keys(fd)
        N = Int(only(ReadVTK.get_data(fd["HexVTKHDF_N"])))
        Ne = Int(only(ReadVTK.get_data(fd["HexVTKHDF_Ne"])))
    end
    return VTUSeries{T}(path, files, ts, size(pts, 2), N, Ne, names)
end

nsteps(s::VTUSeries) = length(s.files)
times(s::VTUSeries) = s.times
field_names(s::VTUSeries) = s.names

"""
    coordinates(s::VTUSeries)

The stored points, reshaped to `(3, N, N, N, Ne)` when the series
carries the layout field data, else the raw `(3, npoints)` matrix.
"""
function coordinates(s::VTUSeries)
    pts = ReadVTK.get_points(ReadVTK.VTKFile(s.files[1]))
    s.N == 0 && return pts
    return reshape(pts, 3, s.N, s.N, s.N, s.Ne)
end

"""
    readfield(s::VTUSeries, name, step::Int)
    s[name, step]

One stored field at one step, shaped `(N, N, N, Ne)` when the layout is
known (else the flat `npoints` vector).
"""
function readfield(s::VTUSeries, name::AbstractString, step::Int)
    1 <= step <= nsteps(s) ||
        throw(BoundsError("step $step of $(nsteps(s))"))
    pd = ReadVTK.get_point_data(ReadVTK.VTKFile(s.files[step]))
    name in keys(pd) ||
        error("HexVTKHDF: no field \"$name\" (have: " *
              "$(join(s.names, ", ")))")
    data = ReadVTK.get_data(pd[name])
    s.N == 0 && return data
    return reshape(data, s.N, s.N, s.N, s.Ne)
end

Base.getindex(s::VTUSeries, name::AbstractString, step::Int) =
    readfield(s, name, step)

function Base.show(io::IO, ::MIME"text/plain", s::VTUSeries{T}) where {T}
    print(io, "VTUSeries{$T}(\"", s.path, "\"): ")
    s.N > 0 && print(io, "N = ", s.N, ", Ne = ", s.Ne, ", ")
    print(io, s.npoints, " points, ", nsteps(s), " steps")
    isempty(s.names) || print(io, "\n  fields: ", join(s.names, ", "))
    return nothing
end
