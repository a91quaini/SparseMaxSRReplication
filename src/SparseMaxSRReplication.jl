module SparseMaxSRReplication
"""
SparseMaxSRReplication — lightweight helpers to reproduce experiments and
calibrate inputs for the SparseMaxSR package.

Public API re-exports (see `?name` for docs):
- `data_dir`
- `load_matrix`
- `load_managed_portfolios`
- `compute_mve_sr_decomposition`
- `simulate_mve_sr`
- `calibrate_factor_model`
- `calibrate_factor_model_from_data`
- `compute_simulation_results`
"""

# Depend on the core algorithms provided by SparseMaxSR
using SparseMaxSR
using Random, LinearAlgebra
using Serialization

# Bring in the local utilities (a nested module defined in the included file)
include("SparseMaxSRReplication/Utils.jl")
using .Utils

# ----------------------------
# Re-exports (public surface)
# ----------------------------
export data_dir,
       load_matrix,
       load_managed_portfolios,
       n_choose_k_mve_sr,
       compute_mve_sr_decomposition,
       simulate_mve_sr,
       calibrate_factor_model,
       calibrate_factor_model_from_data,
       compute_simulation_results

# ----------------------------
# Initialization & precompile
# ----------------------------
function __init__()
end

end # module SparseMaxSRReplication
