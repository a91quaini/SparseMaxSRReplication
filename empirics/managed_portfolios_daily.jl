#!/usr/bin/env julia
# Empirics: Managed Portfolios (Daily) — non-overlapping OOS Sharpe across k
#
# What this script does
# ---------------------
# 1) Loads daily managed-portfolio panels from SparseMaxSRReplication data.
# 2) Prints (T,N) dimensions and the number of rolling windows.
# 3) Sets in-sample (IS) and out-of-sample (OOS) window sizes.
# 4) Sets a k-grid: 5:5:min(OOS-5, 100, N).
# 5) Runs a rolling analysis with non-overlapping OOS windows:
#    - For each window: IS = past W_in days, OOS = next W_out days.
#    - For each k: runs n_choose_k_mve_sr to obtain LASSO / LASSO-REFIT /
#      MIQP / MIQP-REFIT weights in-sample and evaluates SR on the OOS block.
# 6) Aggregates (average) OOS Sharpe across windows for each method and k.
# 7) Prints a compact table at the end and saves results to empirics/results/managed_portfolios_daily.
#
# Parallelization strategy
# -----------------------
# We parallelize across OOS windows (outer loop) using Base.Threads.@threads.
# Typically, the number of windows exceeds the number of k values, giving better
# load balancing and amortizing data access. Within each window we loop over k.

using SparseMaxSRReplication.Utils
using LinearAlgebra
using Statistics
using Random
using Base.Threads
using Serialization

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (you can tweak these)
# ──────────────────────────────────────────────────────────────────────────────
const W_IN_OPTIONS = (126, 252, 504)   # choose one below -> 6 months, 1 year, 2 years
const W_IN  = 252                      # in-sample window size (trading days ~ 1y)
const W_OUT = 63                       # out-of-sample window size (~ 1 quarter)
const SAVE_RESULTS = true              # set true to serialize results table

# Optional knobs passed into searches (can be left empty NamedTuples)
const LASSO_PARAMS = (;)
const MIQP_PARAMS  = (;)

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
R, dates = Utils.load_managed_portfolios(; freq = :daily, get_dates = true)
T, N = size(R)

# Validate W_IN and build initial information banner
if W_IN ∉ W_IN_OPTIONS
    @warn "W_IN not in recommended options; proceeding anyway" W_IN W_IN_OPTIONS
end

# Derive k-grid
k_max = min(W_OUT - 5, 100, N)
if k_max < 5
    error("k_max < 5 (got $k_max). Increase W_OUT or ensure N≥5.")
end
k_grid = collect(5:5:k_max)

# Rolling windows: indices are 1-based. For window index w = 0,1,2,...
# IS = [s_in : e_in] of length W_IN; OOS = [s_out : e_out] of length W_OUT
# We step by W_OUT so that OOS windows are non-overlapping.
function compute_window_indices(T::Int, W_in::Int, W_out::Int)
    idx_pairs = Tuple{Vector{Int},Vector{Int}}[]
    w = 0
    while true
        s_in = 1 + w * W_out
        e_in = s_in + W_in - 1
        s_out = e_in + 1
        e_out = s_out + W_out - 1
        if e_out > T
            break
        end
        push!(idx_pairs, (collect(s_in:e_in), collect(s_out:e_out)))
        w += 1
    end
    return idx_pairs
end

idx_pairs = compute_window_indices(T, W_IN, W_OUT)
W = length(idx_pairs)
# Closed-form expectation for #windows: floor((T - W_IN) / W_OUT) but not below 0
expected_W = max(0, fld(T - W_IN, W_OUT))

# ── Informational header (then stay quiet until results) ──────────────────────
println("Managed portfolios daily panel: T=$(T), N=$(N)")
println("k-grid: ", k_grid)
println("# windows (non-overlapping OOS): $(W) (expected $(expected_W) with W_IN=$(W_IN), W_OUT=$(W_OUT))")

# ──────────────────────────────────────────────────────────────────────────────
# Storage: accumulate sums of SR per (k, method). We'll average at the end.
# ──────────────────────────────────────────────────────────────────────────────
const METHODS = (:lasso, :lasso_refit, :miqp, :miqp_refit)
K = length(k_grid)
S_acc = Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS)
counts = zeros(Int, K)  # how many windows contributed (should be W; kept for safety)

# Use a lock for thread-safe merges (vectorized @atomic is not supported)
const MERGE_LOCK = ReentrantLock()

# Thread-local buffers to minimize contention
threadlocal_buffers() = (Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS), zeros(Int, K))

# ──────────────────────────────────────────────────────────────────────────────
# Main loop parallelized over windows
# ──────────────────────────────────────────────────────────────────────────────
@threads for w in 1:W
    S_local, C_local = threadlocal_buffers()
    idx_in, idx_out = idx_pairs[w]

    for (ik, k) in enumerate(k_grid)
        res = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
            lasso_params = LASSO_PARAMS,
            miqp_params  = MIQP_PARAMS,
            use_refit_lasso = true,
            use_refit_miqp  = true,
            epsilon_in = Utils.EPS_RIDGE,
            epsilon_out = Utils.EPS_RIDGE,
            stabilize_Σ = true,
            do_checks = false,
        )
        S_local[:lasso][ik]       += res.sr_lasso_vanilla
        S_local[:lasso_refit][ik] += res.sr_lasso_refit
        S_local[:miqp][ik]        += res.sr_miqp_vanilla
        S_local[:miqp_refit][ik]  += res.sr_miqp_refit
        C_local[ik] += 1
    end

    # Merge locals into globals
    lock(MERGE_LOCK) do
        for m in METHODS
            @inbounds S_acc[m] .+= S_local[m]
        end
        @inbounds counts .+= C_local
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Finalize averages and print compact table
# ──────────────────────────────────────────────────────────────────────────────
avg = Dict{Symbol, Vector{Float64}}(m => similar(S_acc[m]) for m in METHODS)
for m in METHODS
    @inbounds for i in eachindex(k_grid)
        avg[m][i] = counts[i] == 0 ? NaN : S_acc[m][i] / counts[i]
    end
end

println("
Average OOS Sharpe by k (non-overlapping windows):")
println(rpad("k", 6), rpad("LASSO", 14), rpad("LASSO-REFIT", 14), rpad("MIQP", 14), rpad("MIQP-REFIT", 14))
for (i, k) in enumerate(k_grid)
    @printf("%-6d%-14.4f%-14.4f%-14.4f%-14.4f
", k,
        avg[:lasso][i], avg[:lasso_refit][i], avg[:miqp][i], avg[:miqp_refit][i])
end

# ──────────────────────────────────────────────────────────────────────────────
# Save results table
# ──────────────────────────────────────────────────────────────────────────────
if SAVE_RESULTS
    save_dir = joinpath(dirname(@__FILE__), "results", "managed_portfolios_daily")
    mkpath(save_dir)
    outfile = joinpath(save_dir, @sprintf("result_w_in_%d_w_out_%d.jls", W_IN, W_OUT))

    results = (
        k_grid = k_grid,
        avg_lasso        = avg[:lasso],
        avg_lasso_refit  = avg[:lasso_refit],
        avg_miqp         = avg[:miqp],
        avg_miqp_refit   = avg[:miqp_refit],
        W_in = W_IN, W_out = W_OUT,
        T = T, N = N,
    )
    open(outfile, "w") do io
        serialize(io, results)
    end
    println("
Saved results to $(outfile)")
end