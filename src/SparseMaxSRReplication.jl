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
using Random
using LinearAlgebra
using Serialization
using Statistics
using Printf
using Base.Threads
using Plots

# Bring in the local utilities (a nested module defined in the included file)
include("SparseMaxSRReplication/Utils.jl")
include("SparseMaxSRReplication/UtilsEmpirics.jl")
using .Utils
using .UtilsEmpirics

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
       compute_simulation_results,
       EmpiricConfig, 
       EmpiricResults,
       run_managed_portfolios_daily,
       print_sr_table, 
       print_status_table,
       save_results!,
       plot_oos_sr_by_k

# ----------------------------
# Initialization & precompile
# ----------------------------
function __init__()
end

end # module SparseMaxSRReplication
