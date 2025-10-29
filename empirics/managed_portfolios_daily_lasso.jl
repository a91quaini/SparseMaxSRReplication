#!/usr/bin/env julia

# Managed Portfolios (Daily) — LASSO-REFIT alpha grid + one MIQP-REFIT
# Parallelizes over windows, pins solver/BLAS threads to avoid oversubscription.

using SparseMaxSRReplication
using SparseMaxSRReplication.Utils
using SparseMaxSRReplication.UtilsEmpirics
using LinearAlgebra, Random, Statistics
using Printf, Dates
using Plots
import Base.Threads
import SparseMaxSRReplication.Utils: n_choose_k_mve_sr

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

const ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Windows (trading days)
const W_IN  = 252         # ≈ 1y IS
const W_OUT = 126         # ≈ 6m OOS

# Asset subset and RNG
const N_ASSETS = 250
const RNG_SEED = 12345

# k-grid
const K_MIN  = 1
const K_STEP = 5
const K_CAP  = 150        # final cap will be min(K_CAP, N_ASSETS-1, W_IN-1)

# Missing handling & panel choice
const PANEL_FREQ = :daily
const PANEL_TYPE = :US
const HANDLING_MISSING = :Median

# Covariance stabilization
const EPS_IN  = SparseMaxSRReplication.Utils.EPS_RIDGE
const EPS_OUT = SparseMaxSRReplication.Utils.EPS_RIDGE
const STABILIZE = true

# LASSO defaults (besides alpha)
const LASSO_COMMON = (; nlambda=200, lambda_min_ratio=1e-3, standardize=false)

# MIQP defaults — one refit method
const MIQP_PARAMS = (; exactly_k=true, mipgap=5e-3, time_limit=120, threads=1)

# IO
const SAVE_RESULTS   = true
const SAVE_DIR       = joinpath(dirname(@__FILE__), "..", "empirics", "results", "managed_portfolios_daily_lasso_grid")   # use package default if empty
isdir(OUT_DIR) || mkpath(OUT_DIR)
const MAKE_PLOT      = true
const FIG_SAVE_DIR = joinpath(dirname(@__FILE__), "..", "empirics", "results", "managed_portfolios_daily_lasso_grid") #"..", "empirics", "figures", "managed_portfolios_daily_lasso_grid")
 #"..", "empirics", "figures", "managed_portfolios_daily_lasso_grid")
if MAKE_PLOT || SAVE_RESULTS
    isdir(FIG_SAVE_DIR) || mkpath(FIG_SAVE_DIR)
end
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Build window index pairs (non-overlapping)
function _compute_window_indices(T::Int, w_in::Int, w_out::Int)
    idx = Vector{Tuple{Vector{Int},Vector{Int}}}()
    t = 1
    while t + w_in - 1 <= T
        idx_in = collect(t:(t + w_in - 1))
        t_eval_start = t + w_in
        idx_out = Int[]
        if t_eval_start + w_out - 1 <= T
            idx_out = collect(t_eval_start:(t_eval_start + w_out - 1))
        end
        push!(idx, (idx_in, idx_out))
        t += w_out           # non-overlapping evaluation windows
    end
    idx
end

# Nice field name for each alpha, e.g. 0.10 → :avg_lasso_refit_a010
_lasso_field(alpha::Real) = Symbol(@sprintf("avg_lasso_refit_a%03d", round(Int, 100*alpha)))

# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

function run_managed_portfolios_daily_lasso_grid()
    println("Starting run at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))

    # Avoid oversubscription: parallelize over windows only
    LinearAlgebra.BLAS.set_num_threads(1)

    # Load panel and sample assets
    R, dates = UtilsEmpirics.load_managed_portfolios(; freq=PANEL_FREQ,
                                                     type=PANEL_TYPE,
                                                     handling_missing=HANDLING_MISSING,
                                                     get_dates=true)
    T_full, N_full = size(R)
    @assert N_ASSETS ≤ N_full "N_ASSETS=$N_ASSETS > available N=$N_full"

    if RNG_SEED !== nothing
        Random.seed!(RNG_SEED)
    end
    asset_idx = sort(randperm(N_full)[1:N_ASSETS])
    R = R[:, asset_idx]
    T, N = size(R)

    # Windows and k-grid
    idx_pairs = _compute_window_indices(T, W_IN, W_OUT)
    W = length(idx_pairs)
    k_max = min(K_CAP, N-1, W_IN-1)  # cap by N and IS size (safe with stabilization)
    k_grid = collect(K_MIN:K_STEP:k_max)
    K = length(k_grid)

    @printf("Summary — windows (W_IN=%d, W_OUT=%d), assets N=%d (from N_full=%d)\n",
            W_IN, W_OUT, N, N_full)
    println("k-grid: ", k_grid)
    println("alphas: ", ALPHAS)

    # Storage: sum of OOS SR by method & k, and counts (for averaging)
    sums_miqp   = zeros(Float64, K); counts = zeros(Int, K)
    sums_lasso  = Dict{Float64, Vector{Float64}}(α => zeros(Float64, K) for α in ALPHAS)

    # Parallel over windows
    Threads.@threads for w in 1:W
        idx_in, idx_out = idx_pairs[w]
        isempty(idx_out) && continue  # need OOS to evaluate

        for (ik, k) in enumerate(k_grid)

            # --- MIQP-REFIT once per (window,k)
            res_miqp = n_choose_k_mve_sr(R, idx_in, idx_out, k;
                use_refit_lasso = false,
                use_refit_miqp  = true,
                lasso_params = (;),                    # unused
                miqp_params  = MIQP_PARAMS,
                epsilon_in = EPS_IN,  epsilon_out = EPS_OUT,
                stabilize_Σ = STABILIZE,
                do_checks = false,
            )
            if isfinite(res_miqp.sr_miqp_refit)
                @inbounds sums_miqp[ik] += res_miqp.sr_miqp_refit
                @inbounds counts[ik]    += 1
            end

            # --- LASSO-REFIT for each alpha
            for α in ALPHAS
                res_lasso = n_choose_k_mve_sr(R, idx_in, idx_out, k;
                    use_refit_lasso = true,
                    use_refit_miqp  = false,
                    lasso_params = merge(LASSO_COMMON, (; alpha = α)),
                    miqp_params  = (;),
                    epsilon_in = EPS_IN,  epsilon_out = EPS_OUT,
                    stabilize_Σ = STABILIZE,
                    do_checks = false,
                )
                if isfinite(res_lasso.sr_lasso_refit)
                    @inbounds sums_lasso[α][ik] += res_lasso.sr_lasso_refit
                end
            end
        end
    end

    elapsed = NaN # (fill if you time the run externally)

    # Averages
    avg_miqp_refit = [counts[ik] > 0 ? sums_miqp[ik] / counts[ik] : NaN for ik in 1:K]
    avg_lasso_refit = Dict(α => [counts[ik] > 0 ? sums_lasso[α][ik] / counts[ik] : NaN for ik in 1:K]
                           for α in ALPHAS)

    # Assemble a result NamedTuple compatible with your plotting helper
    # (dynamic fields for each LASSO alpha)
    res_pairs = Any[
        (:W_in, W_IN), (:W_out, W_OUT), (:N, N), (:N_full, N_full),
        (:k_grid, k_grid), (:elapsed_seconds, elapsed),
        (:avg_miqp_refit, avg_miqp_refit),
    ]
    for α in ALPHAS
        push!(res_pairs, (_lasso_field(α), avg_lasso_refit[α]))
    end
    res = (; res_pairs...)

    # Save (CSV + JLS) using package utilities if desired
    if SAVE_RESULTS
        isdir(SAVE_DIR) || mkpath(SAVE_DIR)

        outfile = joinpath(SAVE_DIR,
            @sprintf("managed_portfolios_daily_lasso_grid_%s.jls",
                    Dates.format(now(), "yyyy-mm-dd_HHMMSS")))

        try
            # Save both the res NamedTuple and meta-info
            open(outfile, "w") do io
                serialize(io, res)
            end
            println("✅ Results saved to ", outfile)
        catch err
            @warn "Failed to save results" exception=(err, catch_backtrace())
        end
    end

    # Plot: Average OOS SR by k — one line per method
    if MAKE_PLOT
        isdir(FIG_SAVE_DIR) || mkpath(FIG_SAVE_DIR)
        plt = plot(size=(900,520), xlabel="k", ylabel="Average OOS Sharpe", legend=:topleft,
                   title="Daily managed portfolios — LASSO-REFIT α-grid vs MIQP-REFIT")
        plot!(k_grid, avg_miqp_refit, lw=3, label="MIQP-REFIT")

        # plot each α (sorted) with slimmer lines
        for α in sort(ALPHAS)
            plot!(k_grid, avg_lasso_refit[α], lw=1.5, label=@sprintf("LASSO-REFIT (α=%.2f)", α))
        end

        png(joinpath(FIG_SAVE_DIR, "oos_sr_by_k_lasso_grid"))
        close(plt)
        println("Saved plot to ", FIG_SAVE_DIR)
    end

    return res
end

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

res = run_managed_portfolios_daily_lasso_grid()

println("\nSummary — Average OOS Sharpe by k")
println("-----------------------------------")
@printf("%6s  %12s", "k", "MIQP-REFIT")
for α in sort(ALPHAS)
    @printf("  %16s", @sprintf("LASSO-REFIT (α=%.2f)", α))
end
println()
println("-"^160)

for (ik, k) in enumerate(res.k_grid)
    @printf("%6d  %12.4f", k, res.avg_miqp_refit[ik])
    for α in sort(ALPHAS)
        field = Symbol(@sprintf("avg_lasso_refit_a%03d", round(Int, 100*α)))
        vals = getfield(res, field)
        @printf("  %16.4f", vals[ik])
    end
    println()
end

println("Done.")
