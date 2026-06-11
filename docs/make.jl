# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using HexVTKHDF

makedocs(; sitename="HexVTKHDF", format=Documenter.HTML(), modules=[HexVTKHDF])

deploydocs(; repo="github.com/eschnett/HexVTKHDF.jl.git", devbranch="main", push_preview=true)
