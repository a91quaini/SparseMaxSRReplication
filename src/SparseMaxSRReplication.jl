module SparseMaxSRReplication

using SparseMaxSR        # for all the core algorithms
include("Utils.jl")      # defines module Utils

using .Utils: load_matrix,
              compute_mve_sr_decomposition,
              simulate_mve_sr

export load_matrix,
       compute_mve_sr_decomposition,
       simulate_mve_sr

end # module
