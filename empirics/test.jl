#!/usr/bin/env julia
# Lightweight smoke test for the daily managed-portfolios pipeline.
# - Trim to T=100, N=10.
# - Use W_IN=60, W_OUT=10, and k ∈ {5, 8}.
# - Non-overlapping OOS windows; average OOS Sharpe per method & k.

using SparseMaxSRReplication.Utils
using Statistics
using Random
using Printf

# --------------------------
# Config
# --------------------------
const W_IN  = 60
const W_OUT = 10
const K_LIST = [5, 8]  # explicit per request (no W_OUT-5 cap)
const VERBOSE = true

# Optional knobs to forward to searches (empty is fine)
const LASSO_PARAMS = (;)
const MIQP_PARAMS  = (;)

# For reproducibility (in case any method uses RNG internally)
Random.seed!(123)

# --------------------------
# Load and trim data
# --------------------------
R, dates = Utils.load_managed_portfolios(; freq = :daily, get_dates = true)
T0, N0 = size(R)
T = min(100, T0)
N = min(10, N0)
R = @view R[1:T, 1:N]
dates = dates[1:T]
println("Smoke-test panel: T=$T, N=$N (trimmed from T0=$T0, N0=$N0)")

# --------------------------
# Build non-overlapping window indices
# --------------------------
function window_indices_nonoverlap(T::Int, W_in::Int, W_out::Int)
    pairs = Tuple{Vector{Int},Vector{Int}}[]
    w = 0
    while true
        s_in = 1 + w * W_out
        e_in = s_in + W_in - 1
        s_out = e_in + 1
        e_out = s_out + W_out - 1
        if e_out > T
            break
        end
        push!(pairs, (collect(s_in:e_in), collect(s_out:e_out)))
        w += 1
    end
    return pairs
end

pairs = window_indices_nonoverlap(T, W_IN, W_OUT)
W = length(pairs)
println("# windows (non-overlapping OOS): $W (expected 4 with T=100, W_IN=60, W_OUT=10)")

# --------------------------
# Accumulators
# --------------------------
methods = (:lasso, :lasso_refit, :miqp, :miqp_refit)
S = Dict(m => zeros(length(K_LIST)) for m in methods)
C = zeros(Int, length(K_LIST))

# --------------------------
# Main loop: window → k
# --------------------------
for (w, (idx_in, idx_out)) in enumerate(pairs)
    VERBOSE && println("\nWindow $w: IS=$(first(idx_in))–$(last(idx_in))  OOS=$(first(idx_out))–$(last(idx_out))")
    for (ik, k) in enumerate(K_LIST)
        try
            res = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                lasso_params=LASSO_PARAMS,
                miqp_params=MIQP_PARAMS,
                use_refit_lasso=true,
                use_refit_miqp=true,
                epsilon_in=Utils.EPS_RIDGE,
                epsilon_out=Utils.EPS_RIDGE,
                stabilize_Σ=true,
                do_checks=false)

            S[:lasso][ik]       += res.sr_lasso_vanilla
            S[:lasso_refit][ik] += res.sr_lasso_refit
            S[:miqp][ik]        += res.sr_miqp_vanilla
            S[:miqp_refit][ik]  += res.sr_miqp_refit
            C[ik] += 1

            if VERBOSE
                @printf("  k=%-2d  SR(L)=%7.4f  SR(LR)=%7.4f  SR(M)=%7.4f  SR(MR)=%7.4f\n",
                        k, res.sr_lasso_vanilla, res.sr_lasso_refit,
                        res.sr_miqp_vanilla, res.sr_miqp_refit)
            end
        catch err
            @warn "n_choose_k_mve_sr failed; filling NaNs for this (window,k)" window=w k=k error=err
            # leave accumulators unchanged for this (window,k); count not incremented
        end
    end
end

# --------------------------
# Averages and sanity checks
# --------------------------
avg = Dict(m => similar(S[m]) for m in methods)
for m in methods
    for (i, k) in enumerate(K_LIST)
        avg[m][i] = C[i] == 0 ? NaN : S[m][i] / C[i]
    end
end

println("\nAveraged OOS Sharpe by k (over $W windows):")
println(rpad("k", 6), rpad("LASSO", 12), rpad("LASSO-REFIT", 14), rpad("MIQP", 12), rpad("MIQP-REFIT", 14))
for (i, k) in enumerate(K_LIST)
    @printf("%-6d%-12.4f%-14.4f%-12.4f%-14.4f\n", k, avg[:lasso][i], avg[:lasso_refit][i], avg[:miqp][i], avg[:miqp_refit][i])
end

# Basic assertions to confirm the run "worked"
@assert W == 4 "Expected 4 windows with (T=100, W_IN=60, W_OUT=10), got $W"
@assert all(C .>= 1) "No results aggregated for at least one k. Check warnings above."
@assert all(isfinite.(avg[:lasso])) "LASSO average SR contains non-finite values"
@assert all(isfinite.(avg[:lasso_refit])) "LASSO-REFIT average SR contains non-finite values"
@assert all(isfinite.(avg[:miqp])) "MIQP average SR contains non-finite values"
@assert all(isfinite.(avg[:miqp_refit])) "MIQP-REFIT average SR contains non-finite values"

println("\nSmoke test completed successfully.")
