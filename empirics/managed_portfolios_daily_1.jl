#!/usr/bin/env julia
# Empirics runner: Managed Portfolios (Daily) — comparison variant
# ──────────────────────────────────────────────────────────────────────────────
# Same structure/IO as managed_portfolios_daily.jl, but compares:
#   • LASSO (vanilla, single α)
#   • LASSO-REFIT (several α values)
#   • MIQP-REFIT
# ──────────────────────────────────────────────────────────────────────────────

using SparseMaxSRReplication
using SparseMaxSRReplication.UtilsEmpirics
using SparseMaxSRReplication.Utils
using Printf, Dates, Random, LinearAlgebra, Serialization
using Base.Threads
using StatsBase  # for sample()

# In-sample and out-of-sample window size
const W_IN_CHOSEN  = 252
const W_OUT_CHOSEN = 126

# Number of assets
N_ASSETS_CHOSEN = 250

# Portfolio cardinality
K_MIN_CHOSEN = 1
K_MIN_CHOSEN = max(min(K_MIN_CHOSEN, N_ASSETS_CHOSEN-1), 1)
K_STEP_CHOSEN = 5
K_CAP_CHOSEN = 150
K_CAP_CHOSEN = min(K_CAP_CHOSEN, N_ASSETS_CHOSEN-1, W_OUT_CHOSEN)

# RNG seed
RNG_SEED_CHOSEN = 12345

# Threads for MIQP -> leave to 1 since we parallelize over estimating windows
MIQP_THREADS_CHOSEN = 1 # max(Threads.nthreads()-1,1)

# Recommended param lookups keyed by W_IN (same as base file)
const RECOMMENDED_LASSO = Dict(
    126 => (; alpha=0.50, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7, standardize=false),
    252 => (; alpha=0.70, nlambda=200, lambda_min_ratio=1e-3, epsilon=1e-7, standardize=false),
    504 => (; alpha=0.90, nlambda=200, lambda_min_ratio=1e-4, epsilon=1e-8, standardize=false),
)
const RECOMMENDED_MIQP = Dict(
    126 => (; exactly_k=true, mipgap=0.1e-2,  time_limit=60, threads=MIQP_THREADS_CHOSEN),
    252 => (; exactly_k=true, mipgap=0.5e-3, time_limit=60, threads=MIQP_THREADS_CHOSEN),
    504 => (; exactly_k=true, mipgap=0.2e-3, time_limit=60, threads=MIQP_THREADS_CHOSEN),
)

# Configure (kept for parity and IO helpers)
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
    k_min = K_MIN_CHOSEN, k_step = K_STEP_CHOSEN, k_cap = K_CAP_CHOSEN,
    handling_missing = :Median,
    panel_freq = :daily,
    panel_type = :US,
    save_results = true,
    save_dir = ""
)

# Our LASSO comparison settings
const LASSO_VANILLA_ALPHA = cfg.LASSO_PARAMS.alpha
const ALPHAS_REFIT = [0.30, 0.50, 0.70, 0.90]  # several α values to compare

# Helper to name per-α refit fields
_lasso_refit_field(α::Real) = Symbol(@sprintf("avg_lasso_refit_a%03d", round(Int, 100*α)))

println("Starting run at ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))

# Load the same data the base runner uses
loaded = Utils.load_managed_portfolios(; freq=cfg.panel_freq, type=cfg.panel_type, handling_missing=cfg.handling_missing)
R, names = if loaded isa Tuple && length(loaded) >= 2
    (loaded[1], loaded[2])
else
    Rm = loaded
    (Rm, ["A$(i)" for i in 1:size(Rm,2)])
end

T, N_full = size(R)

# Subselect N_ASSETS columns reproducibly (same policy as base)
Random.seed!(cfg.RNG_SEED)
perm = sort(sample(1:N_full, cfg.N_ASSETS; replace=false))
R = R[:, perm]; names = names[perm]
N = size(R, 2)

# Build non-overlapping window indices and k-grid with the base rule
idx_pairs = UtilsEmpirics.compute_window_indices(T, cfg.W_IN, cfg.W_OUT)
W = length(idx_pairs)
k_max = min(cfg.W_OUT - cfg.k_min, cfg.k_cap, N)
k_grid = collect(cfg.k_min:cfg.k_step:k_max)
K = length(k_grid)

# Thread-local accumulators
local_miqp    = [zeros(Float64, K) for _ in 1:nthreads()]
local_lasso_v = [zeros(Float64, K) for _ in 1:nthreads()]
local_lasso_r = [Dict(α => zeros(Float64, K) for α in ALPHAS_REFIT) for _ in 1:nthreads()]
local_count   = [zeros(Int, K) for _ in 1:nthreads()]

t_elapsed = @elapsed begin
Threads.@threads for w in 1:W
    tid = threadid()
    idx_in, idx_out = idx_pairs[w]
    isempty(idx_out) && continue

    for (ik, k) in enumerate(k_grid)
        any_ok = false

        # MIQP-REFIT
        res_miqp = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
            lasso_params = (;),
            miqp_params  = cfg.MIQP_PARAMS,
            use_refit_lasso = false,
            use_refit_miqp  = true,
            epsilon_in = cfg.epsilon_in,
            epsilon_out = cfg.epsilon_out,
            stabilize_Σ = cfg.stabilize_Σ,
            do_checks = false,
        )
        if isfinite(res_miqp.sr_miqp_refit)
            @inbounds local_miqp[tid][ik] += res_miqp.sr_miqp_refit
            any_ok = true
        end

        # LASSO (vanilla, single α = cfg.LASSO_PARAMS.alpha)
        res_lv = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
            lasso_params = cfg.LASSO_PARAMS,
            miqp_params  = (;),
            use_refit_lasso = false,
            use_refit_miqp  = false,
            epsilon_in = cfg.epsilon_in,
            epsilon_out = cfg.epsilon_out,
            stabilize_Σ = cfg.stabilize_Σ,
            do_checks = false,
        )
        val = if hasproperty(res_lv, :sr_lasso_vanilla)
            res_lv.sr_lasso_vanilla
        elseif hasproperty(res_lv, :sr_lasso)
            getfield(res_lv, :sr_lasso)
        else
            NaN
        end
        if isfinite(val)
            @inbounds local_lasso_v[tid][ik] += val
            any_ok = true
        end

        # LASSO-REFIT — loop externally over α
        for α in ALPHAS_REFIT
            res_lr = Utils.n_choose_k_mve_sr(R, idx_in, idx_out, k;
                lasso_params = merge(cfg.LASSO_PARAMS, (; alpha=α)),
                miqp_params  = (;),
                use_refit_lasso = true,
                use_refit_miqp  = false,
                epsilon_in = cfg.epsilon_in,
                epsilon_out = cfg.epsilon_out,
                stabilize_Σ = cfg.stabilize_Σ,
                do_checks = false,
            )
            if isfinite(res_lr.sr_lasso_refit)
                @inbounds local_lasso_r[tid][α][ik] += res_lr.sr_lasso_refit
                any_ok = true
            end
        end

        if any_ok
            @inbounds local_count[tid][ik] += 1
        end
    end
end
end # elapsed

# Merge
sums_miqp    = zeros(Float64, K)
sums_lasso_v = zeros(Float64, K)
sums_lasso_r = Dict(α => zeros(Float64, K) for α in ALPHAS_REFIT)
counts       = zeros(Int, K)
for t in 1:nthreads()
    sums_miqp    .+= local_miqp[t]
    sums_lasso_v .+= local_lasso_v[t]
    counts       .+= local_count[t]
    for α in ALPHAS_REFIT
        sums_lasso_r[α] .+= local_lasso_r[t][α]
    end
end

# Averages
_safe_avg(sumv, cnt) = map(((s,c),)-> c==0 ? NaN : s/c, zip(sumv, cnt))
avg_miqp_refit    = _safe_avg(sums_miqp, counts)
avg_lasso_vanilla = _safe_avg(sums_lasso_v, counts)
avg_lasso_refit_byα = Dict(α => _safe_avg(sums_lasso_r[α], counts) for α in ALPHAS_REFIT)

# Build result NamedTuple like the base, adding per-α LASSO-REFIT fields
base = (
    W_in = cfg.W_IN, W_out = cfg.W_OUT, N = N, N_full = N_full,
    asset_names = names, k_grid = k_grid, counts = counts,
    elapsed_seconds = t_elapsed,
    avg_lasso_vanilla = avg_lasso_vanilla,
    avg_miqp_refit = avg_miqp_refit,
)
pairs_vec = collect(pairs(base))
for α in ALPHAS_REFIT
    push!(pairs_vec, (_lasso_refit_field(α) => avg_lasso_refit_byα[α]))
end
res = NamedTuple(pairs_vec)

# Brief run summary (same lines as base)
@printf("\nSummary — windows (W_IN=%d, W_OUT=%d), assets N=%d (from N_full=%d)\n",
        res.W_in, res.W_out, res.N, res.N_full)
println("k-grid: ", res.k_grid)
@printf("Elapsed time: %.2f seconds\n", res.elapsed_seconds)

# Final outputs (once) — reuse base helpers
print_sr_table(res)
print_status_table(res)

# Save — SAME directory and filename pattern as base
save_results!(res; cfg)

# Plot — SAME helper and figures folder; explicitly select the three families
fig_dir = joinpath(dirname(@__FILE__), "..", "empirics", "figures", "managed_portfolios_daily")
fields = Symbol[]
labels = String[]
push!(fields, :avg_lasso_vanilla); push!(labels, @sprintf("LASSO (vanilla, α=%.2f)", LASSO_VANILLA_ALPHA))
for α in sort(ALPHAS_REFIT)
    f = _lasso_refit_field(α)
    if hasproperty(res, f)
        push!(fields, f)
        push!(labels, @sprintf("LASSO-REFIT (α=%.2f)", α))
    end
end
push!(fields, :avg_miqp_refit); push!(labels, "MIQP-REFIT")

plot_oos_sr_by_k(res;
    method_fields = fields,
    method_labels = labels,
    save_dir = fig_dir,
)
