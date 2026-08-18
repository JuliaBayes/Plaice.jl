using Pkg: Pkg
Pkg.develop(; path = dirname(@__DIR__))

using Documenter, DocumenterCodeBlocks, VectorBijectors

makedocs(;
    sitename = "VectorBijectors",
    format = Documenter.HTML(),
    modules = [VectorBijectors],
    pages = ["index.md"],
    checkdocs = :export,
    plugins = [CodeBlocks()],
)

deploydocs(;
    repo = "github.com/JuliaBayes/VectorBijectors.jl.git",
    push_preview = true,
)
