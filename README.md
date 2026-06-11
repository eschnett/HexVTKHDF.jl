# HexVTKHDF

VTKHDF5 I/O and visualization for hexahedral meshes

[![CI](https://github.com/eschnett/HexVTKHDF.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/HexVTKHDF.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/eschnett/HexVTKHDF.jl/actions/workflows/docs.yml/badge.svg)](https://eschnett.github.io/HexVTKHDF.jl/)

## Details

VTKHDF time-series I/O and basic analysis/visualization for
spectral-element data on [HexMeshes](https://github.com/eschnett/HexMeshes)
hexahedral meshes with [HexSBPSAT](https://github.com/eschnett/HexSBPSAT)
SBP-SAT discretizations.

* **Writing** — `VTKHDFWriter` produces VTKHDF 2.x temporal
  UnstructuredGrid files readable by ParaView ≥ 5.12, with per-step
  flushing, HDF5 single-writer/multiple-readers (SWMR) live-read
  support, reproducibility metadata, and — when given the `Mesh` — a
  self-describing `/Discretization` group.
* **Reading** — `VTKHDFFile` reopens such files; the mesh, the SBP
  element, and the geometry are reconstructed from the file alone.
* **Analysis** — fields come back as `MeshField <: AbstractArray{T,4}`
  supporting arithmetic, reductions, and mesh-aware operations
  (`integrate`, `l2_norm_phys`, `probe`).
* **Visualization** — `uniform_slice` interpolates a field onto a
  uniform 2-D grid (Makie-free); loading a Makie backend activates the
  `HexVTKHDFMakieExt` extension with `heatmap`, the `fieldsliceplot`
  recipe, and the figure-level `plotslice`.

## Writing

```julia
using HexMeshes, HexVTKHDF

mesh = make_radial_shell_mesh(Float64, 1.5, 10.0, 4; M_r = 8)
N = 6                                     # GLL nodes per direction

w = VTKHDFWriter("run.vtkhdf", mesh, N;   # or (path, coords; mesh)
                 metadata = (; M = 4, cfl = 0.25, seed = 42))
for (t, fields) in timesteps              # your loop
    write_step!(w, t; fields = ["phi" => phi, "Pi" => Pi])
end
close(w)
```

`coords :: (3, N, N, N, Ne)` node coordinates can be passed directly
(e.g. a `HexSBPSAT` `geom.coords`); `node_coordinates(mesh, N)` computes
them from the mesh. Arrays of shape `(N,N,N,Ne,C)` are written as `C`
scalar fields with suffixed names. `metadata` (any TOML-convertible
`Dict`/`NamedTuple`) is stored together with the full package-version
list in `/Metadata`.

### Live inspection (SWMR)

With `swmr = true` (the default) other processes can read the file
*while the run writes*:

```julia
f = VTKHDFFile("run.vtkhdf"; swmr = true)   # live file
nsteps(f)                                   # completed steps so far
```

SWMR forbids attribute modification, so the VTKHDF `Steps/NSteps`
attribute stays frozen at 1 during the run (ParaView needs the closed
file; `nsteps` counts the `Steps/Values` dataset and is always
correct). `close(w)` finalises `NSteps`; after a killed run, repair the
file with `vtkhdf_finalize!(path)`.

## Reading and analysis

```julia
using HexVTKHDF

VTKHDFFile("run.vtkhdf") do f
    nsteps(f), times(f), field_names(f)
    metadata(f)         # parameters + package versions (TOML Dict)
    versions(f)         # /Discretization schema + package versions

    mesh = read_mesh(f)             # HexMeshes.Mesh, rebuilt from file
    disc = discretization(f)        # mesh + SBP element (lazy ops/geom)

    phi = f["phi", nsteps(f)]       # MeshField at the last step
    maximum(abs, phi)               # any reduction
    err = phi .- phi_exact          # broadcasting stays a MeshField
    integrate(phi)                  # ∫ phi dV  (physical volume weights)
    l2_norm_phys(phi)               # physical L² norm
    probe(phi, 1.2, 0.0, 3.4)       # point value (NaN outside the mesh)
end
```

`MeshField` is an `AbstractArray{T,4}` of shape `(N, N, N, Ne)` carrying
the discretization handle plus its field name and sample time;
broadcasts between fields of the same file share that handle.

## Visualization

```julia
using CairoMakie                   # activates HexVTKHDFMakieExt

s = uniform_slice(phi; axis = :z, offset = 0.0, res = 400)
heatmap(s)                                       # plain heatmap
fieldsliceplot(s; symmetric = true)              # + element edges
fig = plotslice(phi; axis = :x, offset = 1.5)    # Figure with colorbar
save("slice.png", fig)
```

`uniform_slice` interpolates onto a uniform 2-D grid through the mesh
(`NaN` off-mesh) and works without Makie; `FieldSlice` carries the grid,
the values, the element-edge segments, and a label. In the recipe form
`fieldsliceplot(field; sliceaxis = :z, offset = 0, res = 200)` the slice
axis keyword is `sliceaxis` (Makie reserves `axis` for Axis options).

## File schema

The `/VTKHDF` group follows the
[VTKHDF 2.x specification](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format)
(temporal UnstructuredGrid: GLL nodes as points, duplicated across
element interfaces; each element subdivided into `(N−1)³` linear
hexahedra; time steps appended along the unlimited dimension with the
`Steps` bookkeeping group). Two extra root members make files
self-describing; ParaView ignores them.

### `/Metadata`

A TOML string: `parameters` (the writer's `metadata` argument),
`package_versions` (every package in the active manifest), and
`julia_version`.

### `/Discretization` (schema `format_version = 1`)

Written when the writer is given the mesh. Attributes:
`format_version`, `scalar_type` (`"Float64"`/`"Float32"`), `N`, `D`,
`hexvtkhdf_version`, `hexmeshes_version`, `hexsbpsat_version`,
`julia_version`. Readers reject `format_version` values they do not
know, citing the recorded versions.

`/Discretization/Mesh` holds the complete `HexMeshes.Mesh{3,T}` state,
datasets named after the Julia fields — `neighbour` (Int32 6×Ne),
`neighbour_face`/`orientation`/`bdry` (Int8 6×Ne), `vertex_coords`
(T 3×Nv), `vertex_idx` (Int64 8×Ne), `patch_id` (Int32 Ne), `patch_idx`
(Int32 3×Ne), `patch_element_offset` (Int64) — plus attributes `Ne`,
`Nv`. Each patch is one group `/Discretization/Mesh/Patches/<i>` with
attributes `kind` (`"Cubic"`, `"Wedge"`, `"Inflation"`, `"Shell"`,
`"WarpedCubic"`), `dims`, and the variant's fields under their Julia
names (`x_lo`/`x_hi` as vectors, `dir` as Int64, `warp_kind` as a
string, scalars as `T`).

The SBP element and geometry are *not* stored: `make_element(T, N)`,
`make_operators`, and `make_geometry` are deterministic in `(T, N,
mesh)`, so the recorded `scalar_type`, `N`, and mesh reconstruct them
exactly (`discretization(f)` does this lazily).

## Example file

```sh
julia --project=<HexVTKHDF> bin/make_example.jl [path]
```

writes a small (~1.3 MB) self-describing example file — an outgoing
spherical wave on a coarse radial-shell mesh, five time steps, two
fields — and prints the reader's summary of it. Useful as test input
for readers and ParaView, and as a writer-API example.

## Installation

The package depends on the unregistered packages HexMeshes and
HexSBPSAT — `Pkg.develop`/`Pkg.add` them by path/URL first. Makie is a
weak dependency: the plotting API materialises only when a Makie
backend is loaded.
