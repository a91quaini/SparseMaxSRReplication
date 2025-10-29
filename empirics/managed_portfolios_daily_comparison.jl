#!/usr/bin/env julia

# Managed Portfolios (Daily) — LASSO (vanilla, single α) vs LASSO-REFIT (α-grid) vs MIQP-REFIT
# Parallelizes over windows; avoids oversubscription by pinning BLAS/solver threads to 1.

using SparseMaxSRReplication
using SparseMaxSRReplication.UtilsEmpirics
using Printf, Dates
using Plots

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# LASSO vanilla (single alpha)
const LASSO_VANILLA_ALPHA = 0.70

# LASSO-REFIT alphas (grid)
const ALPHAS_REFIT = [0.15, 0.25, 0.35, 0.45,
                      0.55, 0.65, 0.75, 0.85, 0.95]

# Windows (trading days)
const W_IN  = 252         # ≈ 1y IS
const W_OUT = 126         # ≈ 6m OOS

# Asset subset and RNG
const N_ASSETS = 15 # 250
const RNG_SEED = 12345

# k-grid
const K_MIN  = 1
const K_STEP = 5
const K_CAP  = 150        # final cap will be min(K_CAP, N_ASSETS-1, W_IN-1)

# Threads for MIQP -> leave to 1 since we parallelize over estimating windows
const MIQP_THREADS = 1 # max(Threads.nthreads()-1,1)

# Panel & missing data handling
const PANEL_FREQ = :daily
const PANEL_TYPE = :US
const HANDLING_MISSING = :Median

# Stabilization
const EPS_IN  = SparseMaxSRReplication.Utils.EPS_RIDGE
const EPS_OUT = SparseMaxSRReplication.Utils.EPS_RIDGE
const STABILIZE = true

# LASSO common knobs (besides alpha)
const LASSO_COMMON = (; nlambda=200, lambda_min_ratio=1e-3, standardize=false)

# MIQP (single, refit)
const MIQP_PARAMS = (; exactly_k=true, mipgap=5e-3, time_limit=60, threads=MIQP_THREADS)

# Output (figures + results)
const OUT_DIR = joinpath(dirname(@__FILE__), "..", "empirics", "results", "managed_portfolios_daily")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Non-overlapping windows
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
        t += w_out
    end
    return idx
end

# Field name for each LASSO-REFIT alpha, e.g. 0.10 → :avg_lasso_refit_a010
_lasso_refit_field(alpha::Real) = Symbol(@sprintf("avg_lasso_refit_a%03d", round(Int, 100*alpha)))

# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

function run_managed_portfolios_daily_competition()
    println("Starting run at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))

    # One level of parallelism only
    LinearAlgebra.BLAS.set_num_threads(1)

    # Load data and pick assets
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

    # Windows & k-grid
    idx_pairs = _compute_window_indices(T, W_IN, W_OUT)
    W = length(idx_pairs)
    k_max = min(K_CAP, N, W_OUT - K_MIN)
    k_grid = collect(K_MIN:K_STEP:k_max)
    K = length(k_grid)

    @printf("Summary — windows (W_IN=%d, W_OUT=%d), assets N=%d (from N_full=%d)\n",
            W_IN, W_OUT, N, N_full)
    println("k-grid: ", k_grid)
    @printf("LASSO (vanilla) α=%.2f\n", LASSO_VANILLA_ALPHA)
    println("LASSO-REFIT α-grid: ", ALPHAS_REFIT)

    # Storage: sums and counts for averaging
    counts             = zeros(Int, K)
    sums_miqp_refit    = zeros(Float64, K)
    sums_lasso_vanilla = zeros(Float64, K)
    sums_lasso_refit   = Dict{Float64, Vector{Float64}}(α => zeros(Float64, K) for α in ALPHAS_REFIT)

    # Parallel over windows
    Threads.@threads for w in 1:W
        idx_in, idx_out = idx_pairs[w]
        isempty(idx_out) && continue  # only evaluate when OOS exists

        for (ik, k) in enumerate(k_grid)

            # MIQP-REFIT (one config)
            res_miqp = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                use_refit_lasso = false,
                use_refit_miqp  = true,
                lasso_params = (;), miqp_params = MIQP_PARAMS,
                epsilon_in = EPS_IN, epsilon_out = EPS_OUT, stabilize_Σ = STABILIZE,
                do_checks = false,
            )
            if isfinite(res_miqp.sr_miqp_refit)
                @inbounds sums_miqp_refit[ik] += res_miqp.sr_miqp_refit
                @inbounds counts[ik] += 1
            end

            # LASSO (vanilla, single alpha)
            res_lasso_v = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                use_refit_lasso = false,
                use_refit_miqp  = false,
                lasso_params = merge(LASSO_COMMON, (; alpha = LASSO_VANILLA_ALPHA)),
                miqp_params  = (;),
                epsilon_in = EPS_IN, epsilon_out = EPS_OUT, stabilize_Σ = STABILIZE,
                do_checks = false,
            )
            if isfinite(res_lasso_v.sr_lasso_vanilla)
                @inbounds sums_lasso_vanilla[ik] += res_lasso_v.sr_lasso_vanilla
            end

            # LASSO-REFIT (alpha grid)
            for α in ALPHAS_REFIT
                res_lasso_r = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                    use_refit_lasso = true,
                    use_refit_miqp  = false,
                    lasso_params = merge(LASSO_COMMON, (; alpha = α)),
                    miqp_params  = (;),
                    epsilon_in = EPS_IN, epsilon_out = EPS_OUT, stabilize_Σ = STABILIZE,
                    do_checks = false,
                )
                if isfinite(res_lasso_r.sr_lasso_refit)
                    @inbounds sums_lasso_refit[α][ik] += res_lasso_r.sr_lasso_refit
                end
            end
        end
    end

    elapsed = NaN # fill with @elapsed if you time the run

    # Averages
    avg_miqp_refit    = [counts[ik] > 0 ? sums_miqp_refit[ik]    / counts[ik] : NaN for ik in 1:K]
    avg_lasso_vanilla = [counts[ik] > 0 ? sums_lasso_vanilla[ik] / counts[ik] : NaN for ik in 1:K]
    avg_lasso_refit   = Dict(α => [counts[ik] > 0 ? sums_lasso_refit[α][ik] / counts[ik] : NaN for ik in 1:K]
                             for α in ALPHAS_REFIT)

    # Result named tuple (with dynamic fields for each refit α)
    res_pairs = Any[
        (:W_in, W_IN), (:W_out, W_OUT), (:N, N), (:N_full, N_full),
        (:k_grid, k_grid), (:elapsed_seconds, elapsed),
        (:avg_miqp_refit, avg_miqp_refit),
        (:avg_lasso_vanilla, avg_lasso_vanilla),
    ]
    for α in ALPHAS_REFIT
        push!(res_pairs, (_lasso_refit_field(α), avg_lasso_refit[α]))
    end
    res = (; res_pairs...)

    # ── Save JLS
    jlsfile = joinpath(OUT_DIR, @sprintf("result_w_in_%d_w_out_%d_N_%d.jls", W_IN, W_OUT, N))
    open(jlsfile, "w") do io
        serialize(io, res)   # keep your richer res with per-α fields
    end
    println("✅ Results saved to ", jlsfile)

    # ── Save CSV
    df = DataFrame(k = res.k_grid, MIQP_REFIT = res.avg_miqp_refit, LASSO_VANILLA = res.avg_lasso_vanilla)
    for α in sort(ALPHAS_REFIT)
        fld = _lasso_refit_field(α)
        df[!, @sprintf("LASSO_REFIT_α%.2f", α)] = getfield(res, fld)
    end
    csvfile = replace(jlsfile, ".jls" => ".csv")
    CSV.write(csvfile, df)
    println("💾 CSV summary written to ", csvfile)

    # ── Plot: Average OOS SR by k
    # Build fields & labels for the plot helper
    fields = Symbol[:avg_miqp_refit, :avg_lasso_vanilla]
    labels = String["MIQP-REFIT", "LASSO (vanilla, α=$(round(LASSO_VANILLA_ALPHA, digits=2)))"]
    for α in sort(ALPHAS_REFIT)
        push!(fields, _lasso_refit_field(α))
        push!(labels, @sprintf("LASSO-REFIT (α=%.2f)", α))
    end

    # figures → empirics/figures/managed_portfolios_daily (same as the main script)
    fig_root = joinpath(dirname(@__FILE__), "..", "empirics", "figures")
    plot_oos_sr_by_k(res;
        method_fields = fields,
        method_labels = labels,
        save_dir = fig_root,               # UtilsEmpirics adds /managed_portfolios_daily
        filename_base = @sprintf("oos_sr_by_k_Win_%d_Wout_%d_N_%d", W_IN, W_OUT, N),
    )


    return res
end

# ──────────────────────────────────────────────────────────────────────────────
# Run + print summary
# ──────────────────────────────────────────────────────────────────────────────

res = run_managed_portfolios_daily_competition()

println("\nSummary — Average OOS Sharpe by k")
println("-----------------------------------")
@printf("%6s  %14s  %18s", "k", "MIQP-REFIT", @sprintf("LASSO (α=%.2f)", LASSO_VANILLA_ALPHA))
for α in sort(ALPHAS_REFIT)
    @printf("  %20s", @sprintf("LASSO-REFIT (α=%.2f)", α))
end
println()
println("-"^200)
for (ik, k) in enumerate(res.k_grid)
    @printf("%6d  %14.4f  %18.4f", k, res.avg_miqp_refit[ik], res.avg_lasso_vanilla[ik])
    for α in sort(ALPHAS_REFIT)
        vals = getfield(res, _lasso_refit_field(α))
        @printf("  %20.4f", vals[ik])
    end
    println()
end
println("Done.")
