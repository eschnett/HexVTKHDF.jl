using Test
using HexVTKHDF

@testset verbose = true "HexVTKHDF" begin
    include("test_writer.jl")
    include("test_meshio.jl")
    include("test_reader.jl")
    include("test_field.jl")
    include("test_slices.jl")
    include("test_vtu.jl")
    # Makie does not support 32-bit platforms (its texture atlas is
    # serialised with 64-bit dimensions, so CairoMakie fails to even
    # precompile there). CI runs the 32-bit job with
    # JULIA_PKG_PRECOMPILE_AUTO=0 so that skipping `using CairoMakie`
    # here also skips its (failing) precompilation.
    if Sys.WORD_SIZE == 64
        include("test_makie.jl")
    else
        @info "Skipping Makie extension tests on a $(Sys.WORD_SIZE)-bit platform"
    end
end
