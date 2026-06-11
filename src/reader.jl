# Reading VTKHDF time-series files. Works on closed files and — with
# `swmr = true` — on files a running job still holds open (count steps
# via `nsteps`, never the NSteps attribute, which the SWMR writer keeps
# frozen until close).

"""
    VTKHDFFile(path; swmr = false)
    VTKHDFFile(f::Function, path; swmr = false)

Open a VTKHDF time-series file (do-block form closes automatically).
`swmr = true` opens in HDF5 SWMR-read mode for files a running writer
still holds open. Step/field access: [`nsteps`](@ref), [`times`](@ref),
[`field_names`](@ref), `file[name, step]` (≡ [`readfield`](@ref)).
Reconstruction (requires the `/Discretization` group, i.e. the writer
was given the mesh): [`read_mesh`](@ref), [`discretization`](@ref),
[`coordinates`](@ref).
"""
mutable struct VTKHDFFile{T}
    file::HDF5.File
    path::String
    N::Int                       # 0 when /Discretization is absent
    Ne::Int
    npoints::Int
    disc::Union{Nothing,Any}     # cached Discretization
end

function VTKHDFFile(path::AbstractString; swmr::Bool = false)
    file = h5open(String(path), "r"; swmr)
    haskey(file, "VTKHDF") ||
        (close(file); error("HexVTKHDF: $path is not a VTKHDF file"))
    npoints = Int(read(file["VTKHDF"]["NumberOfPoints"])[1])
    T = eltype(file["VTKHDF"]["Points"])
    N = 0; Ne = 0
    if haskey(file, "Discretization")
        a = attrs(file["Discretization"])
        N = Int(a["N"])
        Ne = Int(attrs(file["Discretization"]["Mesh"])["Ne"])
    end
    return VTKHDFFile{T}(file, String(path), N, Ne, npoints, nothing)
end

function VTKHDFFile(f::Function, path::AbstractString; swmr::Bool = false)
    file = VTKHDFFile(path; swmr)
    try
        return f(file)
    finally
        close(file)
    end
end

Base.close(f::VTKHDFFile) = close(f.file)

"""
    nsteps(f::VTKHDFFile) -> Int

Number of completed time steps — the length of the `Steps/Values`
dataset, which is correct even for files a SWMR writer still holds open
(unlike the `NSteps` attribute, which is finalised only at close).
"""
nsteps(f::VTKHDFFile) = length(f.file["VTKHDF"]["Steps"]["Values"])

"The sample times of the completed steps."
times(f::VTKHDFFile) = read(f.file["VTKHDF"]["Steps"]["Values"])

"Names of the stored point-data fields."
field_names(f::VTKHDFFile) = sort(keys(f.file["VTKHDF"]["PointData"]))

"""
    metadata(f::VTKHDFFile) -> Dict{String,Any}

The parsed `/Metadata` TOML: `"parameters"` (the writer's `metadata`
argument), `"package_versions"`, `"julia_version"`.
"""
metadata(f::VTKHDFFile) =
    haskey(f.file, "Metadata") ? TOML.parse(read(f.file["Metadata"])) :
    Dict{String,Any}()

"""
    versions(f::VTKHDFFile) -> NamedTuple

Schema and package versions recorded in `/Discretization`:
`(; format, hexvtkhdf, hexmeshes, hexsbpsat, julia)`.
"""
function versions(f::VTKHDFFile)
    haskey(f.file, "Discretization") ||
        error("HexVTKHDF: no /Discretization group in $(f.path)")
    a = attrs(f.file["Discretization"])
    return (; format = Int(a["format_version"]),
            hexvtkhdf = a["hexvtkhdf_version"],
            hexmeshes = a["hexmeshes_version"],
            hexsbpsat = a["hexsbpsat_version"],
            julia = a["julia_version"])
end

"""
    read_mesh(f::VTKHDFFile) -> Mesh{3,T}

Reconstruct the `HexMeshes.Mesh` from the `/Discretization` group.
(Named `read_mesh` rather than `mesh` to avoid clashing with
`Makie.mesh`.)
"""
read_mesh(f::VTKHDFFile) = _read_discretization(f.file)[1]

"""
    discretization(f::VTKHDFFile) -> Discretization

The lazily-cached [`Discretization`](@ref) handle (mesh + SBP element;
operators and geometry are built on first use).
"""
function discretization(f::VTKHDFFile{T}) where {T}
    if f.disc === nothing
        mesh, N, T′ = _read_discretization(f.file)
        T′ === T || error("HexVTKHDF: scalar type mismatch ($T′ vs $T)")
        f.disc = Discretization(mesh, N)
    end
    return f.disc::Discretization
end

"""
    coordinates(f::VTKHDFFile) -> Array{T,5}

The GLL node coordinates as stored in the file, reshaped to
`(3, N, N, N, Ne)` (requires `/Discretization` for N and Ne; otherwise
returns the raw `(3, npoints)` matrix).
"""
function coordinates(f::VTKHDFFile{T}) where {T}
    pts = read(f.file["VTKHDF"]["Points"])
    f.N == 0 && return pts
    return reshape(pts, 3, f.N, f.N, f.N, f.Ne)
end

"""
    readfield(f::VTKHDFFile, name, step::Int) -> MeshField
    f[name, step]

Read one stored field at one time step (1-based) as a
[`MeshField`](@ref) carrying the reconstructed discretization, the
field name, and the sample time.
"""
function readfield(f::VTKHDFFile{T}, name::AbstractString,
                   step::Int) where {T}
    1 <= step <= nsteps(f) ||
        throw(BoundsError("step $step of $(nsteps(f))"))
    pd = f.file["VTKHDF"]["PointData"]
    haskey(pd, name) ||
        error("HexVTKHDF: no field \"$name\" (have: " *
              "$(join(field_names(f), ", ")))")
    d = discretization(f)
    off = (step - 1) * f.npoints
    flat = pd[name][off+1:off+f.npoints]
    data = reshape(flat, f.N, f.N, f.N, f.Ne)
    t = T(times(f)[step])
    return MeshField(data, d; name = String(name), time = t)
end

Base.getindex(f::VTKHDFFile, name::AbstractString, step::Int) =
    readfield(f, name, step)

function Base.show(io::IO, ::MIME"text/plain", f::VTKHDFFile{T}) where {T}
    print(io, "VTKHDFFile{$T}(\"", f.path, "\"): ")
    if f.N > 0
        print(io, "N = ", f.N, ", Ne = ", f.Ne, ", ")
    end
    print(io, f.npoints, " points, ", nsteps(f), " steps")
    fns = field_names(f)
    isempty(fns) || print(io, "\n  fields: ", join(fns, ", "))
    return nothing
end
