# Makie-free slice extraction: interpolate a MeshField onto a uniform
# 2-D grid in an axis-aligned plane (NaN outside the mesh), plus the
# element-edge segments lying in the plane for overlay. The Makie
# extension turns a FieldSlice into a heatmap (+ edges) — but the
# struct is plain data and can be plotted with anything.

const _AXES = (; x = 1, y = 2, z = 3)

"""
    FieldSlice{T}

A field sampled on a uniform 2-D grid in an axis-aligned plane:
in-plane coordinates `us`, `vs`, values `values :: Matrix` (NaN outside
the mesh), the in-plane element-edge segments (`edges_u`/`edges_v`,
consecutive point pairs for `linesegments`), the axis labels, and a
plot-ready `label`. Produced by [`uniform_slice`](@ref); reductions:
[`extrema_finite`](@ref), `minimum`/`maximum` of
`filter(isfinite, s.values)`.
"""
struct FieldSlice{T}
    axis::Symbol
    offset::T
    uaxis::Symbol
    vaxis::Symbol
    us::Vector{T}
    vs::Vector{T}
    values::Matrix{T}
    edges_u::Vector{T}
    edges_v::Vector{T}
    label::String
end

"""
    uniform_slice(f::MeshField; axis = :z, offset = 0,
                  res = 200, extent = nothing) -> FieldSlice

Interpolate `f` onto a uniform `res × res` grid (or `res = (nu, nv)`)
in the plane `axis = offset`. `extent = (ulo, uhi, vlo, vhi)` defaults
to the mesh bounding box of the two in-plane axes. Points outside the
mesh evaluate to NaN.
"""
function uniform_slice(f::MeshField{T}; axis::Symbol = :z,
                       offset::Real = 0,
                       res::Union{Int,NTuple{2,Int}} = 200,
                       extent = nothing) where {T}
    haskey(_AXES, axis) ||
        error("uniform_slice: axis must be :x, :y, or :z (got :$axis)")
    n = _AXES[axis]
    ua, va = axis === :x ? (:y, :z) : axis === :y ? (:x, :z) : (:x, :y)
    iu, iv = _AXES[ua], _AXES[va]
    mesh = f.disc.mesh
    if extent === nothing
        vc = mesh.vertex_coords
        ulo, uhi = extrema(@view vc[iu, :])
        vlo, vhi = extrema(@view vc[iv, :])
    else
        ulo, uhi, vlo, vhi = extent
    end
    nu, nv = res isa Int ? (res, res) : res
    us = collect(T, range(T(ulo), T(uhi), nu))
    vs = collect(T, range(T(vlo), T(vhi), nv))
    off = T(offset)
    pts = Matrix{SVector{3,T}}(undef, nu, nv)
    for (jj, v) in enumerate(vs), (ii, u) in enumerate(us)
        p = MVector{3,T}(undef)
        p[n] = off; p[iu] = u; p[iv] = v
        pts[ii, jj] = SVector{3,T}(p)
    end
    values = interpolate_field(mesh, f.disc.xs, f.data, pts;
                               default = T(NaN))
    eu, ev = element_edges(mesh, axis, off)
    nm = isempty(f.name) ? "field" : f.name
    lbl = string(nm, " (", axis, " = ", off,
                 isnan(f.time) ? "" : string(", t = ", f.time), ")")
    return FieldSlice{T}(axis, off, ua, va, us, vs, values, eu, ev, lbl)
end

"""
    element_edges(mesh::Mesh{3,T}, axis, offset; atol) -> (us, vs)

The element-edge segments lying in the plane `axis = offset` (both
endpoints within `atol`), projected to the in-plane coordinates and
returned as consecutive point pairs (ready for `linesegments`).
"""
function element_edges(mesh::Mesh{3,T}, axis::Symbol, offset::Real;
                       atol = sqrt(eps(T))) where {T}
    n = _AXES[axis]
    iu, iv = axis === :x ? (2, 3) : axis === :y ? (1, 3) : (1, 2)
    # The 12 edges of a hexahedron in the Gmsh corner ordering.
    hex_edges = ((1, 2), (2, 3), (3, 4), (4, 1),
                 (5, 6), (6, 7), (7, 8), (8, 5),
                 (1, 5), (2, 6), (3, 7), (4, 8))
    off = T(offset)
    us = T[]; vs = T[]
    for e in 1:mesh.Ne
        verts = HexMeshes.element_vertices(mesh, e)
        for (a, b) in hex_edges
            pa, pb = verts[a], verts[b]
            if abs(pa[n] - off) < atol && abs(pb[n] - off) < atol
                push!(us, pa[iu]); push!(us, pb[iu])
                push!(vs, pa[iv]); push!(vs, pb[iv])
            end
        end
    end
    return us, vs
end

"""
    extrema_finite(s::FieldSlice) -> (lo, hi)

Extrema over the finite (in-mesh) slice values; `(NaN, NaN)` if none.
"""
function extrema_finite(s::FieldSlice{T}) where {T}
    lo = T(Inf); hi = T(-Inf)
    for v in s.values
        isfinite(v) || continue
        lo = min(lo, v); hi = max(hi, v)
    end
    return lo > hi ? (T(NaN), T(NaN)) : (lo, hi)
end

function Base.show(io::IO, ::MIME"text/plain", s::FieldSlice{T}) where {T}
    lo, hi = extrema_finite(s)
    print(io, "FieldSlice{$T} ", s.label, ": ", length(s.us), "×",
          length(s.vs), " grid (", s.uaxis, ", ", s.vaxis,
          "), finite range [", lo, ", ", hi, "]")
    return nothing
end
