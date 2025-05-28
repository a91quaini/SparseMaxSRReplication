module SparseMaxSRReplication

using SparseMaxSR        # for all the core algorithms
include("Utils.jl")      # defines module Utils

using .Utils: load_matrix,
              compute_mve_sr_decomposition,
              simulate_mve_sr,
              calibrate_factor_model,
              calibrate_factor_model_from_data

export load_matrix,
       compute_mve_sr_decomposition,
       simulate_mve_sr,
       calibrate_factor_model,
       calibrate_factor_model_from_data

end # module
