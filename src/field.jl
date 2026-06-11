# The data abstraction: a per-step nodal field on the spectral-element
# mesh, behaving as an AbstractArray{T,4} of shape (N, N, N, Ne) —
# reductions (minimum/maximum/extrema/sum/argmax/…) and iteration come
# from Base — with broadcasting that preserves the wrapper (the
# RecursiveArrayTools.ArrayPartition pattern), plus mesh-aware
# operations (integrate, l2_norm_phys, probe).

"""
    Discretization(mesh::Mesh{3,T}, N::Int)

Bundle of the mesh and the SBP element of order `N`; the operators and
the geometry are built lazily on first use ([`operators`](@ref),
[`geometry`](@ref)) and cached. Shared by all [`MeshField`](@ref)s of
one file/run.
"""
struct Discretization{T, M<:Mesh{3,T}, E}
    mesh::M
    N::Int
    elem::E
    xs::Vector{T}                 # GLL nodes (for interpolation)
    ops::Base.RefValue{Any}
    geom::Base.RefValue{Any}
end

function Discretization(mesh::Mesh{3,T}, N::Int) where {T}
    elem = make_element(T, N)
    return Discretization{T,typeof(mesh),typeof(elem)}(
        mesh, N, elem, collect(T, elem.xs), Ref{Any}(nothing),
        Ref{Any}(nothing))
end

"The `HexSBPSAT.SBPOps` of the discretization (built on first use)."
function operators(d::Discretization)
    d.ops[] === nothing && (d.ops[] = make_operators(d.elem))
    return d.ops[]
end

"The `HexSBPSAT.MeshGeometry` of the discretization (built on first use)."
function geometry(d::Discretization)
    d.geom[] === nothing && (d.geom[] = make_geometry(d.mesh, d.elem))
    return d.geom[]
end

Base.show(io::IO, d::Discretization{T}) where {T} =
    print(io, "Discretization{$T}(N = ", d.N, ", Ne = ", d.mesh.Ne, ")")

"""
    MeshField(data::Array{T,4}, disc::Discretization; name = "", time = NaN)

A nodal field of shape `(N, N, N, Ne)` on a spectral-element mesh.
Behaves as an `AbstractArray{T,4}`: indexing, iteration, and all Base
reductions work; arithmetic via broadcasting (`f1 .+ f2`, `2 .* f`,
`abs.(f)`, and the non-dot `f1 + f2`, `2f`) preserves the `MeshField`
type and the shared discretization. Mesh-aware operations:
[`integrate`](@ref), [`l2_norm_phys`](@ref), [`probe`](@ref),
[`uniform_slice`](@ref).
"""
struct MeshField{T, D<:Discretization} <: AbstractArray{T,4}
    data::Array{T,4}
    disc::D
    name::String
    time::T
end

MeshField(data::Array{T,4}, disc::Discretization; name::String = "",
          time::Real = NaN) where {T} =
    MeshField{T,typeof(disc)}(data, disc, name, T(time))

Base.size(f::MeshField) = size(f.data)
Base.IndexStyle(::Type{<:MeshField}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(f::MeshField, i::Int) = f.data[i]
Base.@propagate_inbounds Base.setindex!(f::MeshField, v, i::Int) =
    (f.data[i] = v)
Base.parent(f::MeshField) = f.data

# similar with matching shape keeps the wrapper; other shapes degrade
# to a plain Array (a reshaped field has no mesh meaning).
Base.similar(f::MeshField{T}) where {T} = similar(f, T)
Base.similar(f::MeshField, ::Type{S}) where {S} =
    MeshField(similar(f.data, S), f.disc; name = f.name, time = f.time)
Base.similar(f::MeshField, ::Type{S}, dims::Dims) where {S} =
    dims == size(f.data) ?
        MeshField(similar(f.data, S), f.disc; name = f.name, time = f.time) :
        similar(f.data, S, dims)

Base.copy(f::MeshField) =
    MeshField(copy(f.data), f.disc; name = f.name, time = f.time)

# --- broadcasting: preserve the wrapper -------------------------------------

Base.BroadcastStyle(::Type{<:MeshField}) = Broadcast.ArrayStyle{MeshField}()

# Find the first MeshField in a (possibly nested) broadcast tree.
_find_field(bc::Broadcast.Broadcasted) = _find_field(bc.args)
_find_field(args::Tuple) =
    _find_field(_find_field(args[1]), Base.tail(args))
_find_field(f::MeshField, ::Tuple) = f
_find_field(::Any, rest::Tuple) = _find_field(rest)
_find_field(f::MeshField) = f
_find_field(::Any) = nothing
_find_field(::Tuple{}) = nothing

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MeshField}},
                      ::Type{S}) where {S}
    f = _find_field(bc)
    return MeshField(similar(f.data, S, axes(bc)...), f.disc;
                     name = f.name, time = f.time)
end

# --- mesh-aware operations ---------------------------------------------------

"""
    integrate(f::MeshField) -> T

The discrete volume integral `Σ_nodes Hphys · f` (GLL quadrature with
the physical mass weights from the mesh geometry).
"""
function integrate(f::MeshField{T}) where {T}
    Hphys = geometry(f.disc).Hphys
    s = zero(T)
    @inbounds for i in eachindex(f.data, Hphys)
        s += Hphys[i] * f.data[i]
    end
    return s
end

"""
    l2_norm_phys(f::MeshField) -> T

The mass-weighted discrete L² norm `√(Σ Hphys f²)` (delegates to
`HexSBPSAT.discrete_l2_norm`).
"""
l2_norm_phys(f::MeshField) =
    discrete_l2_norm(f.data, geometry(f.disc), operators(f.disc))

"""
    probe(f::MeshField, p; default = NaN) -> T
    probe(f, x, y, z)

Interpolate the field at the physical point `p` (spectral-element
Lagrange interpolation; `default` for points outside the mesh).
"""
probe(f::MeshField{T}, p::SVector{3}; default = T(NaN)) where {T} =
    interpolate_field(f.disc.mesh, f.disc.xs, f.data, SVector{3,T}(p);
                      default = T(default))
probe(f::MeshField, x::Real, y::Real, z::Real; kwargs...) =
    probe(f, SVector(x, y, z); kwargs...)

# Compact display: never dump the 4-D array.
function Base.show(io::IO, ::MIME"text/plain", f::MeshField{T}) where {T}
    nm = isempty(f.name) ? "<unnamed>" : f.name
    print(io, "MeshField{$T} ", nm)
    isnan(f.time) || print(io, " @ t = ", f.time)
    print(io, ": N = ", size(f.data, 1), ", Ne = ", size(f.data, 4))
    lo, hi = extrema(f.data)
    print(io, ", range [", lo, ", ", hi, "]")
    return nothing
end
Base.show(io::IO, f::MeshField) = show(io, MIME"text/plain"(), f)
