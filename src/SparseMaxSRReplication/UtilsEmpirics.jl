module UtilsEmpirics

using ..Utils                      # our public helpers & constants
using LinearAlgebra, Statistics
using Random, Serialization, Printf
using Base.Threads

export EmpiricConfig, EmpiricResults,
       run_managed_portfolios_daily,
       print_sr_table, print_status_table,
       save_results!

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
    S_acc   = Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS)
    counts  = zeros(Int, K)
    opt_ct  = Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS)
    sub_ct  = Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS)

    threadlocal_buffers() = (
        Dict{Symbol, Vector{Float64}}(m => zeros(K) for m in METHODS),  # S_local
        zeros(Int, K),                                                  # C_local
        Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS), # OPT_local
        Dict{Symbol, Vector{Int}}(m => zeros(Int, K) for m in METHODS), # SUB_local
    )

    # 4) Main computation (timed), parallel over windows; no println inside
    elapsed = @elapsed begin
        @threads for w in 1:W
            S_local, C_local, OPT_local, SUB_local = threadlocal_buffers()
            idx_in, idx_out = idx_pairs[w]

            for (ik, k) in enumerate(k_grid)
                res = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                    lasso_params = cfg.LASSO_PARAMS,
                    miqp_params  = cfg.MIQP_PARAMS,
                    use_refit_lasso = true,
                    use_refit_miqp  = true,
                    epsilon_in = cfg.epsilon_in,
                    epsilon_out = cfg.epsilon_out,
                    stabilize_Σ = cfg.stabilize_Σ,
                    do_checks = false,
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
        end
    end

    # 5) Averages
    avg = Dict{Symbol,Vector{Float64}}(m => similar(S_acc[m]) for m in (:lasso,:lasso_refit,:miqp,:miqp_refit))
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

end # module