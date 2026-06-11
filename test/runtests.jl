using Test
using HexVTKHDF

@testset verbose = true "HexVTKHDF" begin
    include("test_writer.jl")
    include("test_meshio.jl")
    include("test_reader.jl")
    include("test_field.jl")
    include("test_slices.jl")
    include("test_makie.jl")
end
