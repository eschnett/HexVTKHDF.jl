"""
    HexVTKHDF

VTKHDF time-series I/O and basic analysis/visualization for
spectral-element data on HexMeshes hexahedral meshes.

  * **Writing**: [`VTKHDFWriter`](@ref) produces VTKHDF 2.x temporal
    UnstructuredGrid files (ParaView ≥ 5.12), with per-step flushing,
    HDF5 SWMR live-read support, reproducibility metadata, and —
    when given the `Mesh` — a self-describing `/Discretization` group
    from which the mesh, the SBP element, and the geometry can be
    reconstructed.
  * **Reading**: [`VTKHDFFile`](@ref) opens such files;
    [`read_mesh`](@ref)/[`discretization`](@ref) rebuild the
    HexMeshes/HexSBPSAT objects; `f[name, step]` returns a
    [`MeshField`](@ref).
  * **Analysis**: `MeshField <: AbstractArray{T,4}` supports arithmetic
    (broadcasting preserves the type), the standard reductions, and the
    mesh-aware [`integrate`](@ref), [`l2_norm_phys`](@ref),
    [`probe`](@ref).
  * **Visualization**: [`uniform_slice`](@ref) interpolates a field
    onto a uniform 2-D grid; with Makie loaded (e.g. `using CairoMakie`)
    the `HexVTKHDFMakieExt` extension provides `heatmap(::FieldSlice)`,
    the `fieldsliceplot` recipe, and [`plotslice`](@ref).
  * **VTU export**: for tools without a VTKHDF reader (VisIt),
    [`VTUWriter`](@ref) writes XML VTK (`.vtu`) series with
    `.pvd`/`.visit` indexes, [`vtkhdf_to_vtu`](@ref) converts existing
    files, and [`VTUSeries`](@ref) reads the series back.

The on-disk schema is documented in the package README.
"""
module HexVTKHDF

using HDF5
using HexMeshes
using HexMeshes: Mesh, PatchDesc, PatchCubic, PatchWedge, PatchInflation,
                 PatchShell, PatchWarpedCubic, MeshConnectivity,
                 Cubic, Wedge, Inflation, Shell, WarpedCubic,
                 element_point_and_jac, locate_point, interpolate_field,
                 tensor_interp
import HexSBPSAT
using HexSBPSAT: make_element, make_operators, make_geometry,
                 discrete_l2_norm
using LinearAlgebra
using StaticArrays
using TOML
import Pkg
import ReadVTK
import WriteVTK

include("writer.jl")
include("meshio.jl")
include("reader.jl")
include("field.jl")
include("slices.jl")
include("vtu.jl")

# Implemented by the Makie extension (load Makie/CairoMakie to use).
"""
    plotslice(f::MeshField; axis = :z, offset = 0, res = 200, kwargs...)
        -> Makie.Figure

Figure-level convenience: a heatmap of [`uniform_slice`](@ref)`(f)`
with element-edge overlay, axis labels, and a colorbar. Provided by the
`HexVTKHDFMakieExt` package extension — load a Makie backend (e.g.
`using CairoMakie`) first.
"""
function plotslice end

"""
    fieldsliceplot(slice_or_field; colormap = :plasma, symmetric = false,
                   edges = true, edgecolor = (:black, 0.6),
                   edgewidth = 0.6, kwargs...)
    fieldsliceplot!(...)

Makie recipe plotting a [`FieldSlice`](@ref) (or a [`MeshField`](@ref),
sliced on the fly via the `sliceaxis`/`offset`/`res`/`extent` keywords —
`sliceaxis`, since Makie reserves `axis` for Axis options) as a heatmap
with an optional element-edge overlay. Provided by the
`HexVTKHDFMakieExt` package extension — load a Makie backend (e.g.
`using CairoMakie`) first.
"""
function fieldsliceplot end
@doc (@doc fieldsliceplot)
function fieldsliceplot! end

export VTKHDFWriter, write_step!, vtkhdf_finalize!
export VTKHDFFile, nsteps, times, field_names, metadata, versions,
       coordinates, read_mesh, discretization, readfield
export VTUWriter, VTUSeries, vtkhdf_to_vtu
export node_coordinates
export Discretization, operators, geometry
export MeshField, integrate, l2_norm_phys, probe
export FieldSlice, uniform_slice, element_edges, extrema_finite
export plotslice, fieldsliceplot, fieldsliceplot!

end
