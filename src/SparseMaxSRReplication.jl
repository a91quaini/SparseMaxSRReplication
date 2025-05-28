module SparseMaxSRReplication

using Serialization

const DATADIR = joinpath(@__DIR__, "..", "data")

"""
    load_matrix(name::String) -> Matrix{Float64}

Load the serialized Matrix stored in `data/<name>.jls`.
"""
function load_matrix(name::AbstractString)
    path = joinpath(DATADIR, name*".jls")
    open(path) do io
        return deserialize(io)
    end
end

export load_matrix

end # module
