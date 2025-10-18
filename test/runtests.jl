# test/runtests.jl
using Test
using Random
using Statistics
using LinearAlgebra
using SparseMaxSRReplication

# ---------------------------
# Reproducibility & metadata
# ---------------------------
ENV["SPARSEMAXSR_DATA"] = get(ENV, "SPARSEMAXSR_DATA", joinpath(dirname(@__DIR__), "data"))
Random.seed!(0x5eed)

const DATA_DIR = get(ENV, "SPARSEMAXSR_DATA", SparseMaxSRReplication.data_dir())
@info "Running tests" unix_time=time() julia=VERSION data_dir=DATA_DIR

# ---------------------------
# Collect test files
# ---------------------------
# Include any file named test-*.jl in this directory (sorted for determinism)
function testfiles()
    dir = @__DIR__
    files = filter(fname -> occursin(r"^test-.*\.jl$", fname), readdir(dir))
    sort!(files)
    return joinpath.(Ref(dir), files)
end

# ---------------------------
# Run all test files
# ---------------------------
@testset "SparseMaxSRReplication tests" begin
    for tf in testfiles()
        @testset "$(basename(tf))" begin
            include(tf)
        end
    end
end

