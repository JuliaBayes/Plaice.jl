using Pkg: Pkg
Pkg.develop(; path=dirname(@__DIR__))

using Documenter, DocumenterCodeBlocks, Plaice, Distributions

makedocs(;
    sitename="Plaice",
    format=Documenter.HTML(),
    modules=[Plaice],
    pages=["index.md", "example.md"],
    checkdocs=:export,
    plugins=[CodeBlocks()],
)

deploydocs(; repo="github.com/JuliaBayes/Plaice.jl.git", push_preview=true)
