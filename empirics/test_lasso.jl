#!/usr/bin/env julia

using SparseMaxSR
using SparseMaxSRReplication
using SparseMaxSRReplication.UtilsEmpirics
using SparseMaxSRReplication.Utils  # for EPS_RIDGE if needed
using Printf, Dates
using LinearAlgebra
using Random
using Plots
using Base.Threads
using Statistics
LinearAlgebra.BLAS.set_num_threads(1)

# ------------------------------ helpers ---------------------------------------

# robust nnz with tolerance (treat tiny magnitudes as zero)
nnz_tol(w::AbstractVector; tol::Real=1e-10) = count(x -> abs(x) > tol, w)

function compute_window_indices(T::Int, W_in::Int, W_out::Int)
    idx_pairs = Tuple{Vector{Int},Vector{Int}}[]
    w = 0
    while true
        s_in = 1 + w*W_out
        e_in = s_in + W_in - 1
        s_out = e_in + 1
        e_out = s_out + W_out - 1
        e_out > T && break
        push!(idx_pairs, (collect(s_in:e_in), collect(s_out:e_out)))
        w += 1
    end
    return idx_pairs
end

# try to get LASSO-REFIT weights / support from the result object
# returns (obtained_k::Int, sr::Float64)
function extract_lasso_refit_k_and_sr(res; tol=1e-10)
    # Common field names we might encounter
    # 1) weights directly
    for f in (:w_lasso_refit, :weights_lasso_refit, :w_refit_lasso, :w_lasso_refitted)
        if hasproperty(res, f)
            w = getfield(res, f)
            return (nnz_tol(w; tol=tol), getfield(res, :sr_lasso_refit))
        end
    end
    # 2) support / indices directly
    for f in (:sel_lasso_refit, :support_lasso_refit, :indices_lasso_refit)
        if hasproperty(res, f)
            idx = getfield(res, f)
            return (length(idx), getfield(res, :sr_lasso_refit))
        end
    end
    # 3) fallback: if we only have vanilla weights/support, still check refit SR
    if hasproperty(res, :sr_lasso_refit)
        # Best-effort guess: use vanilla weights to count (lower bound); still prints SR-refit
        for f in (:w_lasso_vanilla, :weights_lasso_vanilla, :w_lasso)
            if hasproperty(res, f)
                w = getfield(res, f)
                return (nnz_tol(w; tol=tol), getfield(res, :sr_lasso_refit))
            end
        end
        for f in (:sel_lasso, :support_lasso, :indices_lasso)
            if hasproperty(res, f)
                idx = getfield(res, f)
                return (length(idx), getfield(res, :sr_lasso_refit))
            end
        end
    end
    error("Could not extract LASSO-REFIT cardinality from result (no recognizable fields).")
end

# pretty print a header line
function print_header()
    println("Per-k summary (LASSO-REFIT):")
    @printf("%6s  %8s  %9s  %12s  %12s  %12s  %12s\n",
            "k", "Nobs", "%viol>", "min(obt_k)", "median(obt_k)",
            "max(obt_k)", "mean(SR)")
end

# ----------------------------- main probe -------------------------------------

function check_lasso_refit_cardinality(cfg::EmpiricConfig; tol=1e-10, plot_dir="")
    # ---------------- load & sample panel ----------------
    R, _dates = UtilsEmpirics.load_managed_portfolios(; freq=cfg.panel_freq,
                                                      type=cfg.panel_type,
                                                      handling_missing=cfg.handling_missing,
                                                      get_dates=true)
    T, N_full = size(R)
    cfg.N_ASSETS ≤ N_full || error("N_ASSETS=$(cfg.N_ASSETS) > total N=$(N_full).")
    if cfg.RNG_SEED !== nothing
        Random.seed!(cfg.RNG_SEED)
    end
    asset_idx = sort(randperm(N_full)[1:cfg.N_ASSETS])
    R = R[:, asset_idx]
    T, N = size(R)

    # --------------- windows & k-grid --------------------
    idx_pairs = compute_window_indices(T, cfg.W_IN, cfg.W_OUT)  # unchanged helper
    W = length(idx_pairs)
    k_max = min(cfg.k_cap, N, cfg.W_IN - 1)   # <— use W_IN (not W_OUT)
    k_max < cfg.k_min && error("k_max < $(cfg.k_min) (got $k_max). Increase W_IN or ensure N≥$(cfg.k_min).")
    k_grid = collect(cfg.k_min:cfg.k_step:k_max)
    K = length(k_grid)

    println("Starting LASSO-REFIT cardinality audit at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    @printf("W_IN=%d, W_OUT=%d, N=%d (from N_full=%d), k_grid=%s\n",
            cfg.W_IN, cfg.W_OUT, N, N_full, string(k_grid))

    # --------------- storage (lock-free) -----------------
    # one observation per (k, window): store in matrices indexed [ik, w]
    obtained_k = fill(-1, K, W)                 # Int matrix, -1 sentinel
    sr_eval    = fill(NaN, K, W)                # Float64 matrix

    # --------------- threaded compute --------------------
    @threads for w in 1:W
        idx_in, idx_out = idx_pairs[w]
        Rin   = @view R[idx_in, :]
        Tin   = length(idx_in)

        # in-sample moments + stabilized Σ for selection
        μ_in  = vec(mean(Rin; dims=1))
        Σ_in  = cov(Rin; dims=1)
        Σ_in_p = Utils._prep_S(Σ_in, cfg.epsilon_in, cfg.stabilize_Σ)

        # out-of-sample moments if needed
        has_oos = !isempty(idx_out)
        if has_oos
            Rout   = @view R[idx_out, :]
            μ_out  = vec(mean(Rout; dims=1))
            Σ_out  = cov(Rout; dims=1)
            Σ_out_p = Utils._prep_S(Σ_out, cfg.epsilon_out, cfg.stabilize_Σ)
        end

        for (ik, k) in enumerate(k_grid)
            # run LASSO search directly so we can read selection & weights
            res = SparseMaxSR.mve_lasso_relaxation_search(
                μ_in, Σ_in_p, Tin;
                k = k,
                compute_weights = true,
                use_refit = true,                # we care about REFIT
                do_checks = false,
                cfg.LASSO_PARAMS...              # alpha / lambda grid etc.
            )

            # count non-zeros robustly
            w_refit = get(res, :weights, nothing)
            sel     = hasproperty(res, :selection) ? getfield(res, :selection) :
                       (w_refit === nothing ? Int[] :
                        findall(x -> abs(x) > tol, w_refit))

            obtained_k[ik, w] = isempty(sel) ? -1 : length(sel)

            # evaluate SR
            if has_oos && w_refit !== nothing && !isempty(sel)
                # OOS SR on stabilized Σ_out, with learned weights & selection
                sr = SparseMaxSR.compute_sr(w_refit, μ_out, Σ_out_p; selection=sel, do_checks=false)
                sr_eval[ik, w] = sr
            else
                # fall back to IS SR reported by the search
                sr_eval[ik, w] = get(res, :sr, NaN)
            end
        end
    end

    # --------------- summarize ---------------------------
    println()
    print_header()
    for (ik, k) in enumerate(k_grid)
        ks   = [x for x in obtained_k[ik, :] if x ≥ 0]
        srs  = [x for x in sr_eval[ik, :]    if isfinite(x)]
        nobs = length(ks)
        if nobs == 0
            @printf("%6d  %8d  %9s  %12s  %12s  %12s  %12s\n",
                    k, 0, "—", "—", "—", "—", "—")
            continue
        end
        viol   = count(>(k), ks)                   # > k should not happen for refit
        pviol  = 100 * viol / nobs
        sort!(ks)
        min_k  = first(ks)
        med_k  = ks[cld(nobs, 2)]
        max_k  = last(ks)
        meanSR = isempty(srs) ? NaN : mean(srs)
        @printf("%6d  %8d  %9.2f  %12d  %12d  %12d  %12.4f\n",
                k, nobs, pviol, min_k, med_k, max_k, meanSR)
    end

    # explicit violations
    any_viol = false
    for (ik, k) in enumerate(k_grid)
        ks = [x for x in obtained_k[ik, :] if x ≥ 0]
        if any(x -> x > k, ks)
            any_viol = true
            println("\nViolations for k=", k, " → obtained cardinalities > k:")
            println([x for x in ks if x > k])
        end
    end
    if !any_viol
        println("\nAll LASSO-REFIT cardinalities ≤ k (within tol) across all windows.")
    end

    # --------------- optional plots ----------------------
    if !isempty(plot_dir)
        isdir(plot_dir) || mkpath(plot_dir)
        for (ik, k) in enumerate(k_grid)
            ks = [x for x in obtained_k[ik, :] if x ≥ 0]
            isempty(ks) && continue
            uniq  = sort(unique(ks))
            freqs = [count(==(u), ks) for u in uniq]
            bar(uniq, freqs; xlabel="obtained cardinality", ylabel="count",
                title="LASSO-REFIT obtained cardinalities (target k=$k)")
            vline!([k])
            png(joinpath(plot_dir, @sprintf("lasso_refit_cardinalities_k%03d", k)))
            close()
        end
        println("\nSaved per-k cardinality histograms to: ", plot_dir)
    end

    return (; k_grid, obtained_k, sr_eval)
end


# ------------------------------ run once --------------------------------------

# Example: reuse EXACTLY your cfg from the runner (paste or import it)
# Here is a light example; you can just `include("your_runner.jl")` and reuse `cfg`.
const W_IN_CHOSEN  = 252
const W_OUT_CHOSEN = 126
N_ASSETS_CHOSEN    = 250
RNG_SEED_CHOSEN    = 12345

const RECOMMENDED_LASSO = Dict(
    126 => (; alpha=0.50, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7),
    252 => (; alpha=0.70, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7),
    504 => (; alpha=0.90, nlambda=200, lambda_min_ratio=1e-4, epsilon=1e-8),
)
const RECOMMENDED_MIQP = Dict(
    126 => (; exactly_k=true, mipgap=0.1e-2,  time_limit=60, threads=1),
    252 => (; exactly_k=true, mipgap=0.5e-3, time_limit=60, threads=1),
    504 => (; exactly_k=true, mipgap=0.2e-3, time_limit=60, threads=1),
)

cfg = EmpiricConfig(
    W_IN  = W_IN_CHOSEN,
    W_OUT = W_OUT_CHOSEN,
    N_ASSETS = N_ASSETS_CHOSEN,
    RNG_SEED = RNG_SEED_CHOSEN,
    LASSO_PARAMS = get(RECOMMENDED_LASSO, W_IN_CHOSEN, RECOMMENDED_LASSO[504]),
    MIQP_PARAMS  = get(RECOMMENDED_MIQP,  W_IN_CHOSEN, RECOMMENDED_MIQP[504]),
    epsilon_in  = Utils.EPS_RIDGE,
    epsilon_out = Utils.EPS_RIDGE,
    stabilize_Σ = true,
    k_min = 1, k_step = 5, k_cap = 150,
    handling_missing = :Median,
    panel_freq = :daily,
    panel_type = :US,
    save_results = false,
    save_dir = ""
)

# where to save the small per-k cardinality plots (set "" to skip plotting)
outdir = joinpath(dirname(@__FILE__), "..", "empirics", "figures", "lasso_refit_cardinality")
check_lasso_refit_cardinality(cfg; tol=1e-10, plot_dir=outdir)
