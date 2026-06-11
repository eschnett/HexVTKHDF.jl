# Makie integration (loaded automatically when a Makie backend is
# present, e.g. `using CairoMakie`):
#   * `heatmap(slice::FieldSlice)` — plain conversion, the stable core;
#   * `fieldsliceplot(slice)` / `fieldsliceplot(field; axis, offset,
#     res)` — heatmap + element-edge overlay recipe;
#   * `plotslice(field; ...)` — figure-level convenience with Axis,
#     labels, and Colorbar.

module HexVTKHDFMakieExt

using Makie
using HexVTKHDF
using HexVTKHDF: FieldSlice, MeshField, uniform_slice, extrema_finite
# The recipe must extend the stubs in HexVTKHDF so that users reach
# `fieldsliceplot` without referencing the extension module.
import HexVTKHDF: fieldsliceplot, fieldsliceplot!

# --- stable core: FieldSlice is heatmap-able --------------------------------

Makie.convert_arguments(::Type{<:Makie.Heatmap}, s::FieldSlice) =
    (s.us, s.vs, s.values)

# --- recipe: heatmap + element edges -----------------------------------------

Makie.@recipe FieldSlicePlot (slice,) begin
    colormap = :plasma
    "Symmetric colorrange about zero (max |finite value|)."
    symmetric = false
    "Overlay the element edges lying in the slice plane."
    edges = true
    edgecolor = (:black, 0.6)
    edgewidth = 0.6
end

function _slice_colorrange(s::FieldSlice, symmetric::Bool)
    lo, hi = extrema_finite(s)
    isfinite(lo) || return (0.0, 1.0)
    if symmetric
        m = max(abs(lo), abs(hi))
        m = m > 0 ? m : one(m)
        return (-m, m)
    end
    return lo < hi ? (lo, hi) : (lo - 1, hi + 1)
end

function Makie.plot!(p::FieldSlicePlot)
    s = p.slice[]
    cr = _slice_colorrange(s, p.symmetric[])
    heatmap!(p, s.us, s.vs, s.values;
             colormap = p.colormap, colorrange = cr)
    if p.edges[] && !isempty(s.edges_u)
        linesegments!(p, s.edges_u, s.edges_v;
                      color = p.edgecolor, linewidth = p.edgewidth)
    end
    return p
end

# fieldsliceplot(field; sliceaxis, offset, res, extent) slices on the
# fly. (Named `sliceaxis`, not `axis` — Makie reserves the `axis`
# keyword for Axis options.)
Makie.used_attributes(::Type{<:FieldSlicePlot}, ::MeshField) =
    (:sliceaxis, :offset, :res, :extent)
Makie.convert_arguments(::Type{<:FieldSlicePlot}, f::MeshField;
                        sliceaxis = :z, offset = 0, res = 200,
                        extent = nothing) =
    (uniform_slice(f; axis = sliceaxis, offset, res, extent),)

# --- figure-level convenience -------------------------------------------------

function HexVTKHDF.plotslice(f::MeshField; axis::Symbol = :z,
                             offset::Real = 0, res = 200, extent = nothing,
                             size = (700, 600), kwargs...)
    s = uniform_slice(f; axis, offset, res, extent)
    return HexVTKHDF.plotslice(s; size, kwargs...)
end

function HexVTKHDF.plotslice(s::FieldSlice; size = (700, 600), kwargs...)
    fig = Makie.Figure(; size)
    ax = Makie.Axis(fig[1, 1]; title = s.label, xlabel = string(s.uaxis),
                    ylabel = string(s.vaxis), aspect = Makie.DataAspect())
    p = fieldsliceplot!(ax, s; kwargs...)
    hm = p.plots[1]
    Makie.Colorbar(fig[1, 2], hm)
    return fig
end

end
