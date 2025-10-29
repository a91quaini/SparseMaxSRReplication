module UtilsEmpirics

using ..Utils                      # our public helpers & constants
using LinearAlgebra, Statistics
using Random, Serialization, Printf
using Base.Threads
using Plots

export EmpiricConfig, EmpiricResults,
       run_managed_portfolios_daily,
       print_sr_table, print_status_table,
       save_results!,
       plot_oos_sr_by_k

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct EmpiricConfig
    # Windowing
    W_IN::Int  = 504
    W_OUT::Int = 63

    # Asset sub-selection
    N_ASSETS::Int      = 25
    RNG_SEED::Union{Nothing,Int} = 12345

    # Algorithm knobs
    LASSO_PARAMS::NamedTuple = (; alpha=0.90, nlambda=200, lambda_min_ratio=1e-4, standardize=false)
    MIQP_PARAMS::NamedTuple  = (; mipgap=0.002, time_limit=120, threads=max(nthreads()-1,1))

    # Cov stabilization
    epsilon_in::Float64  = Utils.EPS_RIDGE
    epsilon_out::Float64 = Utils.EPS_RIDGE
    stabilize_Σ::Bool    = true

    # k-grid
    k_min::Int = 5
    k_step::Int = 5
    k_cap::Int = 100

    # Data loading
    handling_missing::Symbol = :Median
    panel_freq::Symbol = :daily
    panel_type::Symbol = :US

    # IO
    save_results::Bool = true
    save_dir::String   = ""
end

Base.@kwdef struct EmpiricResults
    # Inputs / meta
    T::Int
    N::Int
    N_full::Int
    asset_idx::Vector{Int}
    W_in::Int
    W_out::Int
    k_grid::Vector{Int}
    counts::Vector{Int}
    elapsed_seconds::Float64

    # SR averages by method
    avg_lasso::Vector{Float64}
    avg_lasso_refit::Vector{Float64}
    avg_miqp::Vector{Float64}
    avg_miqp_refit::Vector{Float64}

    # Status summaries by method (Optimal/Sub)
    opt_counts::Dict{Symbol,Vector{Int}}
    sub_counts::Dict{Symbol,Vector{Int}}
end

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

status_ok(s) = begin
    ss = lowercase(string(s))
    !(occursin("infeas", ss) || occursin("error", ss) || occursin("invalid", ss) || occursin("fail", ss))
end
status_optimal(s) = begin
    ss = lowercase(string(s))
    occursin("optimal", ss) || occursin("exact", ss)   # MIQP vs LASSO paths
end

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

# ──────────────────────────────────────────────────────────────────────────────
# Core runner (no mid-computation prints)
# ──────────────────────────────────────────────────────────────────────────────

const METHODS = (:lasso, :lasso_refit, :miqp, :miqp_refit)
const MERGE_LOCK = ReentrantLock()

function run_managed_portfolios_daily(cfg::EmpiricConfig = EmpiricConfig())
    # 1) Load panel (R: T×N_full), then subselect columns reproducibly
    R, _dates = Utils.load_managed_portfolios(; freq=cfg.panel_freq, type=cfg.panel_type,
                                               handling_missing=cfg.handling_missing, get_dates=true)
    T, N_full = size(R)

    if cfg.N_ASSETS > N_full
        error("N_ASSETS=$(cfg.N_ASSETS) > total available assets N=$(N_full).")
    end
    if cfg.RNG_SEED !== nothing
        Random.seed!(cfg.RNG_SEED)
    end
    asset_idx = sort(randperm(N_full)[1:cfg.N_ASSETS])
    R = R[:, asset_idx]
    T, N = size(R)

    # 2) Build non-overlapping windows & k-grid
    idx_pairs = compute_window_indices(T, cfg.W_IN, cfg.W_OUT)
    W = length(idx_pairs)

    k_max = min(cfg.W_OUT - cfg.k_min, cfg.k_cap, N)
    k_max < cfg.k_min && error("k_max < $(cfg.k_min) (got $k_max). Increase W_OUT or ensure N≥$(cfg.k_min).")
    k_grid = collect(cfg.k_min:cfg.k_step:k_max)
    K = length(k_grid)

    # 3) Accumulators
    METHODS = (:lasso, :lasso_refit, :miqp, :miqp_refit)
    MERGE_LOCK = ReentrantLock()

    S_acc  = Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS)
    counts = zeros(Int, K)
    opt_ct = Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS)
    sub_ct = Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS)

    threadlocal_buffers = () -> (
        Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS),  # S_local
        zeros(Int, K),                                                  # C_local
        Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS), # OPT_local
        Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS), # SUB_local
    )

    # 4) Main computation (timed), bounded concurrency over windows
    elapsed = @elapsed begin
        NWORKERS = max(nthreads() - 1, 1)
        sem = Base.Semaphore(NWORKERS)

        function _process_window!(w::Int)
            # avoid oversubscription from BLAS
            BLAS.set_num_threads(1)

            S_local, C_local, OPT_local, SUB_local = threadlocal_buffers()
            idx_in, idx_out = idx_pairs[w]

            for (ik, k) in enumerate(k_grid)
                res = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                    lasso_params    = cfg.LASSO_PARAMS,
                    miqp_params     = cfg.MIQP_PARAMS,
                    use_refit_lasso = true,
                    use_refit_miqp  = true,
                    epsilon_in      = cfg.epsilon_in,
                    epsilon_out     = cfg.epsilon_out,
                    stabilize_Σ     = cfg.stabilize_Σ,
                    do_checks       = false,
                )

                @inbounds begin
                    S_local[:lasso][ik]       += res.sr_lasso_vanilla
                    S_local[:lasso_refit][ik] += res.sr_lasso_refit
                    S_local[:miqp][ik]        += res.sr_miqp_vanilla
                    S_local[:miqp_refit][ik]  += res.sr_miqp_refit
                    C_local[ik] += 1
                end

                st = Dict(
                    :lasso       => res.status_lasso_vanilla,
                    :lasso_refit => res.status_lasso_refit,
                    :miqp        => res.status_miqp_vanilla,
                    :miqp_refit  => res.status_miqp_refit,
                )
                for m in METHODS
                    s = st[m]
                    if status_ok(s)
                        if status_optimal(s)
                            OPT_local[m][ik] += 1
                        else
                            SUB_local[m][ik] += 1
                        end
                    else
                        SUB_local[m][ik] += 1
                    end
                end
            end

            lock(MERGE_LOCK) do
                for m in METHODS
                    @inbounds S_acc[m] .+= S_local[m]
                    @inbounds opt_ct[m] .+= OPT_local[m]
                    @inbounds sub_ct[m] .+= SUB_local[m]
                end
                @inbounds counts .+= C_local
            end
            return nothing
        end

        tasks = Vector{Task}(undef, W)
        for w in 1:W
            tasks[w] = Threads.@spawn begin
                Base.acquire(sem)
                try
                    _process_window!(w)
                finally
                    Base.release(sem)
                end
            end
        end
        foreach(wait, tasks)
    end

    # 5) Averages
    avg = Dict{Symbol,Vector{Float64}}(m => similar(S_acc[m]) for m in METHODS)
    for m in keys(avg), i in eachindex(k_grid)
        avg[m][i] = counts[i] == 0 ? NaN : S_acc[m][i] / counts[i]
    end

    return EmpiricResults(
        T=T, N=N, N_full=N_full, asset_idx=asset_idx,
        W_in=cfg.W_IN, W_out=cfg.W_OUT, k_grid=k_grid, counts=counts,
        elapsed_seconds=elapsed,
        avg_lasso       = avg[:lasso],
        avg_lasso_refit = avg[:lasso_refit],
        avg_miqp        = avg[:miqp],
        avg_miqp_refit  = avg[:miqp_refit],
        opt_counts = opt_ct,
        sub_counts = sub_ct,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Output utilities (print/save) — called at the END only
# ──────────────────────────────────────────────────────────────────────────────

function print_sr_table(res::EmpiricResults)
    println("\nAverage OOS Sharpe by k (non-overlapping windows):")
    println(rpad("k", 6), rpad("LASSO", 14), rpad("LASSO-REFIT", 14), rpad("MIQP", 14), rpad("MIQP-REFIT", 14))
    for (i, k) in enumerate(res.k_grid)
        @printf("%-6d%-14.4f%-14.4f%-14.4f%-14.4f\n",
                k, res.avg_lasso[i], res.avg_lasso_refit[i], res.avg_miqp[i], res.avg_miqp_refit[i])
    end
end

function print_status_table(res::EmpiricResults)
    println("\nStatus summary by k (Optimal / Sub-optimal):")
    header = ["k", "LASSO", "LASSO-REFIT", "MIQP", "MIQP-REFIT"]
    println(rpad(header[1], 6), rpad(header[2], 18), rpad(header[3], 18), rpad(header[4], 18), rpad(header[5], 18))
    for (i, k) in enumerate(res.k_grid)
        s1 = @sprintf("%d/%d", res.opt_counts[:lasso][i],       res.sub_counts[:lasso][i])
        s2 = @sprintf("%d/%d", res.opt_counts[:lasso_refit][i], res.sub_counts[:lasso_refit][i])
        s3 = @sprintf("%d/%d", res.opt_counts[:miqp][i],        res.sub_counts[:miqp][i])
        s4 = @sprintf("%d/%d", res.opt_counts[:miqp_refit][i],  res.sub_counts[:miqp_refit][i])
        println(@sprintf("%-6d%-18s%-18s%-18s%-18s", k, s1, s2, s3, s4))
    end
end

function save_results!(res::EmpiricResults; cfg::EmpiricConfig, filename::AbstractString="")
    cfg.save_results || return nothing
    save_dir = isempty(cfg.save_dir) ?
        joinpath(dirname(@__FILE__), "..", "..", "empirics", "results", "managed_portfolios_daily") :
        cfg.save_dir
    mkpath(save_dir)
    outfile = isempty(filename) ?
        joinpath(save_dir, @sprintf("result_w_in_%d_w_out_%d_N_%d.jls", res.W_in, res.W_out, res.N)) :
        joinpath(save_dir, filename)

    payload = (
        k_grid = res.k_grid,
        avg_lasso        = res.avg_lasso,
        avg_lasso_refit  = res.avg_lasso_refit,
        avg_miqp         = res.avg_miqp,
        avg_miqp_refit   = res.avg_miqp_refit,
        counts = res.counts,
        status_optimal_counts = (
            lasso       = res.opt_counts[:lasso],
            lasso_refit = res.opt_counts[:lasso_refit],
            miqp        = res.opt_counts[:miqp],
            miqp_refit  = res.opt_counts[:miqp_refit],
        ),
        status_suboptimal_counts = (
            lasso       = res.sub_counts[:lasso],
            lasso_refit = res.sub_counts[:lasso_refit],
            miqp        = res.sub_counts[:miqp],
            miqp_refit  = res.sub_counts[:miqp_refit],
        ),
        W_in = res.W_in, W_out = res.W_out,
        T = res.T, N = res.N, N_full = res.N_full,
        asset_idx = res.asset_idx,
        elapsed_seconds = res.elapsed_seconds,
    )
    open(outfile, "w") do io
        serialize(io, payload)
    end
    println("\nSaved results to $(outfile)")
    return outfile
end

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

"""
    plot_oos_sr_by_k(res;
        method_fields::Vector{Symbol}=Symbol[],
        method_labels::Vector{String}=String[],
        save_dir::Union{Nothing,AbstractString}=nothing,
        filename_base::Union{Nothing,String}=nothing),
        subfolder::String="managed_portfolios_daily"
        )

Plot Average OOS Sharpe by `k` with one line per method found in `res`.

Arguments
---------
- `res`: a results struct/NamedTuple with at least `k_grid` and one or more
  series named like `avg_lasso`, `avg_lasso_refit`, `avg_miqp`, `avg_miqp_refit`.
- `method_fields` (optional): fields to plot, e.g.
  `[:avg_lasso, :avg_lasso_refit, :avg_miqp]`. If empty, the function will
  auto-detect among known method fields present in `res`.
- `method_labels` (optional): labels for the legend, same length & order as
  `method_fields`. If empty, defaults are used for known fields; otherwise the
  field name as a string.
- `save_dir` (optional): directory to save the figure(s). If `nothing`, nothing is saved.
- `filename_base` (optional): base filename without extension. If `nothing`, it tries
  `oos_sr_by_k_Win_<W_in>_Wout_<W_out>_N_<N>` if present in `res`, otherwise `oos_sr_by_k`.
- `subfolder` (optional): subfolder of `/empirics/figures` where figures are stored.

Returns
-------
The `Plots.Plot` object.
"""
function plot_oos_sr_by_k(res;
    method_fields::Vector{Symbol}=Symbol[],
    method_labels::Vector{String}=String[],
    save_dir::Union{Nothing,AbstractString}=nothing,
    filename_base::Union{Nothing,String}=nothing,
    subfolder::String="managed_portfolios_daily",
)
    # x-axis
    ks = hasproperty(res, :k_grid) ? getfield(res, :k_grid) :
         hasproperty(res, :k)      ? getfield(res, :k)      :
         error("`res` must have `k_grid` (or `k`).")

    # Known fields → default labels
    known = Dict(
        :avg_lasso         => "LASSO",
        :avg_lasso_refit   => "LASSO-REFIT",
        :avg_miqp          => "MIQP",
        :avg_miqp_refit    => "MIQP-REFIT",
    )

    # Auto-detect present methods if none explicitly provided
    candidates = isempty(method_fields) ? [s for s in keys(known) if hasproperty(res, s)] : method_fields
    @assert !isempty(candidates) "No method fields to plot. Provide `method_fields` or populate `res` with avg_* fields."

    # Build labels
    labels = if !isempty(method_labels)
        @assert length(method_labels) == length(candidates) "method_labels must match method_fields length."
        method_labels
    else
        [get(known, s, String(s)) for s in candidates]
    end

    # Style & plot
    default(size=(900, 550), legend=:topleft, lw=2, grid=true)
    p = plot()
    markers = [:circle, :utriangle, :square, :diamond, :star5, :xcross, :hexagon, :pentagon, :cross]

    for (i, f) in enumerate(candidates)
        @assert hasproperty(res, f) "`res` has no field $(f)."
        y = getfield(res, f)
        @assert length(y) == length(ks) "Length mismatch for $(f): got $(length(y)) vs k_grid $(length(ks))."
        plot!(p, ks, y; label=labels[i], marker=markers[mod1(i, length(markers))])
    end

    plot!(p; xlabel="k", ylabel="Average OOS Sharpe", title="Average OOS Sharpe by k (non-overlapping windows)")

    # -------- save path (defaults to the package repo) --------
    if save_dir === nothing
        # package root (robust from inside submodules)
        pkgroot = Base.pkgdir(parentmodule(@__MODULE__))
        save_root = joinpath(pkgroot, "empirics", "figures")
    else
        save_root = save_dir
    end
    full_dir = joinpath(save_root, subfolder)

    if !isempty(full_dir)
        mkpath(full_dir)
        base = if filename_base !== nothing
            filename_base
        elseif hasproperty(res, :W_in) && hasproperty(res, :W_out) && hasproperty(res, :N)
            "oos_sr_by_k_Win_$(getfield(res,:W_in))_Wout_$(getfield(res,:W_out))_N_$(getfield(res,:N))"
        else
            "oos_sr_by_k"
        end
        for ext in (:png, :pdf)
            savefig(p, joinpath(full_dir, base * "." * String(ext)))
        end
    end

    return p
end

"""
    pick_lasso_alpha(Rin; k, alphas, method=:cv, kfolds=10, kappa=0.01,
                     epsilon=EPS_RIDGE, stabilize_Σ=true,
                     use_refit=false, do_checks=false, lasso_params=(;))

Select the best **LASSO regularization parameter** `alpha` for a given in-sample
estimation window `Rin` (T×N matrix of excess returns) and desired sparsity
level `k`.

The function automatically decides whether to perform **cross-validation (CV)**
or a **generalized cross-validation (GCV)**-type selection based on sample size
and user input.

### Arguments
- `Rin::AbstractMatrix{<:Real}` : In-sample excess returns (T×N).
- `k::Integer` : Target number of selected assets (`k`-support).
- `alphas::AbstractVector{<:Real}` : Grid of candidate α values in (0,1].
- `method::Symbol = :cv` :
  - `:cv` → K-fold time-series CV (default 10-fold).
  - `:gcv` → Penalized in-sample criterion.
  - `:auto` → Use CV when `T ≥ max(k, kfolds)`, otherwise fall back to GCV.
- `kfolds::Int = 10` : Number of folds for CV.
- `kappa::Real = 0.01` : Complexity penalty weight in GCV criterion.
- `epsilon::Real = EPS_RIDGE` : Ridge stabilization added to sample covariance.
- `stabilize_Σ::Bool = true` : If true, applies Σ stabilization via `_prep_S`.
- `use_refit::Bool = false` : If true, runs LASSO-refit version of the search.
- `do_checks::Bool = false` : Perform input validity checks.
- `lasso_params::NamedTuple = (;)` : Additional keyword arguments forwarded to
  `mve_lasso_relaxation_search`.

### Method
1. **Trivial case:**  
   If `length(alphas) == 1`, the function runs one LASSO fit and returns that α.
2. **Cross-Validation (`method=:cv`):**  
   Performs *blocked* K-fold CV to respect time ordering.  
   For each α:
   - Fit on the training subset (`μ_tr`, `Σ_tr`).
   - Compute validation Sharpe ratio on held-out subset (`μ_val`, `Σ_val`).
   - Average SRs across folds.
   The α giving the **highest average validation Sharpe** is selected.
3. **Generalized Cross-Validation (`method=:gcv`):**  
   Computes a penalized in-sample criterion for each α:
   [
   GCV(α) = log(max(SR_{in}(α), 10^{-12})- κ * nnz(α),
   ]
   where `nnz(α)` is the number of selected assets.  
   The α maximizing this criterion is selected.
4. If no α yields an exact `k`-support, the function picks the **closest support**
   (minimizing `|nnz − k|`) and then the **highest Sharpe ratio** among ties.

### Returns
A named tuple:
(
    alpha        :: Float64,   # best α
    method_used  :: Symbol,    # :single, :cv, or :gcv
    score        :: Float64,   # CV SR or GCV criterion value
    result       :: NamedTuple # full output from mve_lasso_relaxation_search
)
"""
function pick_lasso_alpha(
    Rin::AbstractMatrix{<:Real};
    k::Integer,
    alphas::AbstractVector{<:Real},
    method::Symbol = :cv,          # :cv, :gcv, or :auto
    kfolds::Int    = 10,
    kappa::Real    = 0.01,         # penalty weight for nnz in GCV
    epsilon::Real  = EPS_RIDGE,
    stabilize_Σ::Bool = true,
    use_refit::Bool   = false,
    do_checks::Bool   = false,
    lasso_params::NamedTuple = (;),   # extra knobs passed to LASSO
)
    # trivial case
    if length(alphas) == 1
        α = first(alphas)
        μ  = vec(mean(Rin; dims=1))
        Σ  = cov(Rin; dims=1)
        Σp = _prep_S(Σ, epsilon, stabilize_Σ)
        res = mve_lasso_relaxation_search(
            μ, Σp, size(Rin,1);
            k = k, compute_weights = true, use_refit = use_refit,
            do_checks = do_checks, alpha = α, lasso_params...
        )
        return (alpha = α, method_used = :single, score = float(get(res, :sr, NaN)), result = res)
    end

    T = size(Rin, 1)

    # Decide method
    method_used = if method === :auto
        (T >= k && T >= max(5, kfolds)) ? :cv : :gcv
    elseif method === :cv
        (T >= k && T >= max(5, kfolds)) ? :cv : :gcv
    else
        :gcv
    end

    if method_used === :cv
        # ----- Blocked K-fold CV (contiguous folds to avoid leakage) -----
        kf = min(kfolds, T)                 # guard
        fold_bounds = cumsum(vcat(0, fill(div(T, kf), kf))) .+ (0:kf-1) .* (T % kf .> (0:kf-1))
        # fold i uses validation indices (fold_bounds[i]+1):fold_bounds[i+1]
        function fold_range(i)
            s = fold_bounds[i] + 1
            e = (i < length(fold_bounds)) ? fold_bounds[i+1] : T
            s:e
        end

        # precompute nothing; we recompute μ/Σ per train fold (cheap vs. correctness)
        bestα, bestScore, bestRes = nothing, -Inf, nothing

        for α in alphas
            cv_scores = Float64[]
            any_valid = false
            for i in 1:kf
                idx_val = fold_range(i)
                idx_tr  = setdiff(1:T, idx_val)  # contiguous CV; if T is large, could use views instead

                # moments train / val
                μtr  = vec(mean(@view Rin[idx_tr, :]; dims=1))
                Σtr  = cov(@view Rin[idx_tr, :]; dims=1)
                μval = vec(mean(@view Rin[idx_val,:]; dims=1))
                Σval = cov(@view Rin[idx_val,:]; dims=1)

                Σtr_p  = _prep_S(Σtr,  epsilon, stabilize_Σ)
                Σval_p = _prep_S(Σval, epsilon, stabilize_Σ)

                # fit on train
                res = mve_lasso_relaxation_search(
                    μtr, Σtr_p, length(idx_tr);
                    k = k, compute_weights = true, use_refit = use_refit,
                    do_checks = false, alpha = α, lasso_params...
                )

                # require exact k-support (as requested)
                if length(res.selection) == k && isfinite(get(res, :sr, NaN))
                    sr_val = compute_sr(res.weights, μval, Σval_p; selection = res.selection, do_checks = false)
                    if isfinite(sr_val)
                        push!(cv_scores, sr_val)
                        any_valid = true
                    end
                end
            end

            # aggregate (mean SR across valid folds)
            if any_valid
                sc = mean(cv_scores)
                if sc > bestScore
                    bestScore = sc
                    bestα     = α
                    # also compute a "final" result on all Rin for the chosen α candidate
                    μ  = vec(mean(Rin; dims=1))
                    Σ  = cov(Rin; dims=1)
                    Σp = _prep_S(Σ, epsilon, stabilize_Σ)
                    bestRes = mve_lasso_relaxation_search(
                        μ, Σp, T;
                        k = k, compute_weights = true, use_refit = use_refit,
                        do_checks = do_checks, alpha = α, lasso_params...
                    )
                end
            end
        end

        # fallback: if no α produced exact k-support anywhere, pick closest-support by IS SR
        if bestRes === nothing
            bestGap = typemax(Int)
            bestTie = -Inf
            for α in alphas
                μ  = vec(mean(Rin; dims=1))
                Σ  = cov(Rin; dims=1)
                Σp = _prep_S(Σ, epsilon, stabilize_Σ)
                res = mve_lasso_relaxation_search(
                    μ, Σp, T; k = k, compute_weights = true, use_refit = use_refit,
                    do_checks = do_checks, alpha = α, lasso_params...
                )
                gap = abs(length(res.selection) - k)
                sr  = float(get(res, :sr, NaN))
                if gap < bestGap || (gap == bestGap && sr > bestTie)
                    bestGap, bestTie = gap, sr
                    bestα, bestRes   = α, res
                    bestScore        = sr
                end
            end
        end

        return (alpha = bestα, method_used = :cv, score = bestScore, result = bestRes)
    else
        # ----- GCV-like (penalized IS) -----
        μ  = vec(mean(Rin; dims=1))
        Σ  = cov(Rin; dims=1)
        Σp = _prep_S(Σ, epsilon, stabilize_Σ)

        bestα, bestVal, bestRes = nothing, -Inf, nothing
        for α in alphas
            res = mve_lasso_relaxation_search(
                μ, Σp, T;
                k = k, compute_weights = true, use_refit = use_refit,
                do_checks = do_checks, alpha = α, lasso_params...
            )
            # exact k-support preferred; if none, fallback later
            sr_in = float(get(res, :sr, NaN))
            nnz   = length(res.selection)
            crit  = log(max(sr_in, 1e-12)) - kappa * nnz
            if (nnz == k && crit > bestVal) || (bestα === nothing && crit > bestVal)
                bestVal, bestα, bestRes = crit, α, res
            end
        end

        # fallback to closest-support if nothing with nnz==k won
        if length(bestRes.selection) != k
            bestGap = typemax(Int); bestTie = -Inf
            for α in alphas
                res = mve_lasso_relaxation_search(
                    μ, Σp, T;
                    k = k, compute_weights = true, use_refit = use_refit,
                    do_checks = do_checks, alpha = α, lasso_params...
                )
                gap = abs(length(res.selection) - k)
                sr  = float(get(res, :sr, NaN))
                if gap < bestGap || (gap == bestGap && sr > bestTie)
                    bestGap, bestTie = gap, sr
                    bestα, bestRes   = α, res
                    bestVal          = log(max(sr, 1e-12)) - kappa * length(res.selection)
                end
            end
        end

        return (alpha = bestα, method_used = :gcv, score = bestVal, result = bestRes)
    end
end



end # module