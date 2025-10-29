#!/usr/bin/env julia
# Empirics runner: Managed Portfolios (Daily)
# ──────────────────────────────────────────────────────────────────────────────
# What this does
#   • Loads the US daily managed-portfolio panel (Utils.load_managed_portfolios)
#   • Randomly subselects N_ASSETS columns
#   • Runs non-overlapping OOS experiments over a k-grid
#   • Aggregates average OOS Sharpe by method (LASSO, LASSO-REFIT, MIQP, MIQP-REFIT)
#   • Summarizes solver statuses (Optimal vs Sub-optimal) by k
#   • Prints final tables and saves results (no mid-run prints)
#
# Recommended window sizes (trading days)
#   W_IN ∈ {126 (≈6m), 252 (≈1y), 504 (≈2y)}
#   W_OUT ∈ {63 (≈3m), 126 (≈6m), 252 (≈1y)}
#
# Recommended LASSO params (depends on W_IN)
#   W_IN=126 → (alpha=0.50, nlambda=200, lambda_min_ratio=1e-3, standardize=false)
#   W_IN=252 → (alpha=0.70, nlambda=200, lambda_min_ratio=1e-3, standardize=false)
#   W_IN=504 → (alpha=0.90, nlambda=200, lambda_min_ratio=1e-4, standardize=false)
#
# Recommended MIQP params (depends on W_IN)
#   W_IN=126 → (mipgap=0.01,  time_limit=120, threads=max(nthreads()-1,1))
#   W_IN=252 → (mipgap=0.005, time_limit=120, threads=max(nthreads()-1,1))
#   W_IN=504 → (mipgap=0.002, time_limit=120, threads=max(nthreads()-1,1))
#
# Notes
#   • Stabilization: use epsilon_in = epsilon_out = Utils.EPS_RIDGE and stabilize_Σ=true.
#   • k-grid: starts at k_min, increases by k_step, capped by min(W_OUT - k_min, k_cap, N).
#   • Results are stored under empirics/results/managed_portfolios_daily/ by default.
# ──────────────────────────────────────────────────────────────────────────────

using SparseMaxSRReplication
using SparseMaxSRReplication.UtilsEmpirics
using Printf, Dates
using Plots

# In-sample and out-of-sample window size
# Choose your W_IN / W_OUT first (pick from the sets above)
const W_IN_CHOSEN  = 252
const W_OUT_CHOSEN = 126

# Number of assets
N_ASSETS_CHOSEN = 250

# Portfolio cardinality
# Minimum k - keep it above 1 and below N_ASSETS_CHOSEN-1
K_MIN_CHOSEN = 1 
K_MIN_CHOSEN = max(min(K_MIN_CHOSEN, N_ASSETS_CHOSEN-1), 1)
# Step between consecutive k's
K_STEP_CHOSEN = 5
# Maximum k - keep it below N_ASSETS_CHOSEN-1 and below W_OUT_CHOSEN
K_CAP_CHOSEN = 150
K_CAP_CHOSEN = min(K_CAP_CHOSEN, N_ASSETS_CHOSEN-1, W_OUT_CHOSEN)

# RNG seed
RNG_SEED_CHOSEN = 12345

# Threads for MIQP -> leave to 1 since we parallelize over estimating windows
MIQP_THREADS_CHOSEN = 1 # max(Threads.nthreads()-1,1)

# Recommended param lookups keyed by W_IN (you can override below if desired)
const RECOMMENDED_LASSO = Dict(
    126 => (; alpha=0.50, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7),
    252 => (; alpha=0.70, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7),
    504 => (; alpha=0.90, nlambda=200, lambda_min_ratio=1e-4, epsilon=1e-8),
)
const RECOMMENDED_MIQP = Dict(
    126 => (; exactly_k=true, mipgap=0.1e-2,  time_limit=60, threads=MIQP_THREADS_CHOSEN),
    252 => (; exactly_k=true, mipgap=0.5e-3, time_limit=60, threads=MIQP_THREADS_CHOSEN),
    504 => (; exactly_k=true, mipgap=0.2e-3, time_limit=60, threads=MIQP_THREADS_CHOSEN),
)

# Configure (adjust as desired)
cfg = EmpiricConfig(
    # Windows
    W_IN  = W_IN_CHOSEN,
    W_OUT = W_OUT_CHOSEN,

    # Subset of assets to use (reproducible with RNG_SEED)
    N_ASSETS = N_ASSETS_CHOSEN,
    RNG_SEED = RNG_SEED_CHOSEN,

    # Method params (use recommended per W_IN; override here if needed)
    LASSO_PARAMS = get(RECOMMENDED_LASSO, W_IN_CHOSEN, RECOMMENDED_LASSO[504]),
    MIQP_PARAMS  = get(RECOMMENDED_MIQP,  W_IN_CHOSEN, RECOMMENDED_MIQP[504]),

    # Covariance stabilization for both selection and OOS evaluation
    epsilon_in  = SparseMaxSRReplication.Utils.EPS_RIDGE,
    epsilon_out = SparseMaxSRReplication.Utils.EPS_RIDGE,
    stabilize_Σ = true,

    # k-grid
    k_min = K_MIN_CHOSEN, k_step = K_STEP_CHOSEN, k_cap = K_CAP_CHOSEN, # Recommended: 5, k_step = 5, k_cap = 100,

    # Data source
    handling_missing = :Median,
    panel_freq = :daily,
    panel_type = :US,

    # IO
    save_results = true,
    save_dir = ""   # default results/ path under project; set your own to override
)

println("Starting run at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
res = run_managed_portfolios_daily(cfg)

# Brief run summary
@printf("\nSummary — windows (W_IN=%d, W_OUT=%d), assets N=%d (from N_full=%d)\n",
        res.W_in, res.W_out, res.N, res.N_full)
println("k-grid: ", res.k_grid)
@printf("Elapsed time: %.2f seconds\n", res.elapsed_seconds)

# Final outputs (once)
print_sr_table(res)
print_status_table(res)

# Save
save_results!(res; cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: Average OOS Sharpe by k (one line per method)
# ──────────────────────────────────────────────────────────────────────────────
fig_dir = joinpath(dirname(@__FILE__), "..", "empirics", "figures", "managed_portfolios_daily")

# Let it auto-detect methods present in `res` and save files:
p = plot_oos_sr_by_k(res; save_dir=fig_dir)

# Or explicitly choose fields & labels (order controls legend order):
p = plot_oos_sr_by_k(
    res;
    method_fields = [:avg_lasso, :avg_lasso_refit, :avg_miqp, :avg_miqp_refit],
    method_labels = ["LASSO", "LASSO-REFIT", "MIQP", "MIQP-REFIT"],
    save_dir = fig_dir,
)