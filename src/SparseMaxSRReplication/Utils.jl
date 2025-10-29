module Utils

"""
Utilities for data loading, portfolio selection experiments, factor-model calibration,
and small simulation helpers used by **SparseMaxSRReplication**.

Public functions are documented and exported; small helpers are internal (prefixed with
an underscore) and kept un-exported to reduce surface area.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
using Serialization            # serialize/deserialize
using Random                   # RNG handling
using LinearAlgebra            # cholesky, Symmetric, I
using Statistics               # mean, var, cov

# Import the core from SparseMaxSR (assumed in the project env)
import SparseMaxSR:
    compute_mve_weights,
    compute_sr,
    compute_mve_sr,
    mve_exhaustive_search,
    mve_lasso_relaxation_search,
    mve_miqp_heuristic_search

# ──────────────────────────────────────────────────────────────────────────────
# Exports (public API)
# ──────────────────────────────────────────────────────────────────────────────
export data_dir,
       load_matrix,
       load_managed_portfolios,
       n_choose_k_mve_sr,
       compute_mve_sr_decomposition,
       simulate_mve_sr,
       calibrate_factor_model,
       calibrate_factor_model_from_data,
       compute_simulation_results

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Numerical knobs
# ──────────────────────────────────────────────────────────────────────────────

"""Small ridge used to stabilize covariance matrices when requested."""
const EPS_RIDGE = 1e-6

# ──────────────────────────────────────────────────────────────────────────────
# Paths & I/O
# ──────────────────────────────────────────────────────────────────────────────

"""
    data_dir() -> String

Return the directory used to load serialized matrices. Respects the environment
variable `SPARSEMAXSR_DATA` if set; otherwise defaults to `<pkg>/data`.
"""
@inline data_dir() = get(ENV, "SPARSEMAXSR_DATA", joinpath(dirname(dirname(@__DIR__)), "data"))

"""
    load_matrix(name; dir=data_dir(), freq=nothing) -> Matrix{Float64}

Load a numeric matrix from a `.jls` file saved with `Serialization.serialize`.

`name` can be:
- a base name without extension (e.g. `"factors_ff5_monthly"`),
- a managed subfolder path (e.g. `"managed_portfolios_monthly/factors_ff5_monthly"`),
- or a full path with extension.

If `freq` is `:monthly` or `:daily`, the function first probes the corresponding
managed-portfolios subfolder before falling back to the base `dir`.

Throws an informative `ArgumentError` if no candidate path is readable, or the
deserialized object is not interpretable as a matrix.
"""
function load_matrix(
    name::AbstractString;
    dir::AbstractString = data_dir(),
    freq::Union{Nothing,Symbol,String} = nothing,
)
    dir  = normpath(dir)
    base = endswith(name, ".jls") ? name[1:end-4] : name

    # Build candidate paths (priority may be altered by `freq`)
    candidates = String[]
    if freq !== nothing
        fstr = freq isa Symbol ? String(freq) : String(freq)
        push!(candidates, joinpath(dir, "managed_portfolios_" * fstr, base * ".jls"))
    end
    push!(candidates, joinpath(dir, base * ".jls"))      # e.g. data/factors_ff5_monthly.jls
    push!(candidates, joinpath(dir, base))                 # tolerate full subpaths

    tried = String[]
    for raw in candidates
        p = normpath(raw)
        try
            X = open(p, "r") do io
                deserialize(io)
            end
            M = _ensure_matrix(X, p)
            return Matrix{Float64}(M)
        catch err
            push!(tried, "$(p)  [$(typeof(err))]")
        end
    end
    throw(ArgumentError("File not found or unreadable. Tried:\n" * join(("  " .* tried), "\n")))
end

# Accept matrix directly, or (dates, X), or Dict/NamedTuple with :data
@inline function _ensure_matrix(X, path::AbstractString)
    if X isa AbstractMatrix
        return X
    elseif X isa Tuple && length(X) == 2 && (X[2] isa AbstractMatrix)
        return X[2]
    elseif X isa NamedTuple && haskey(X, :data)
        return X.data
    elseif X isa Dict
        for (_, v) in X
            v isa AbstractMatrix && return v
        end
    end
    throw(ArgumentError("Deserialized object at $path is not a matrix (got $(typeof(X)))."))
end

# ──────────────────────────────────────────────────────────────────────────────
# Linear-algebra helpers (internal)
# ──────────────────────────────────────────────────────────────────────────────

"""Prepare a symmetric (and optionally ridge-stabilized) covariance matrix.

If `stabilize` is true: `Σ_eff = Sym((Σ+Σ')/2 + ε·mean(diag(Σ))·I)`; otherwise
no ridge is added. Always returns a `Symmetric{Float64}` wrapper over a dense
matrix for downstream routines.
"""
@inline function _prep_S(Σ::AbstractMatrix{T}, epsilon::Real, stabilize::Bool) where {T<:Real}
    M = Matrix{Float64}(Σ)          # promote once
    A = (M .+ M') ./ T(2)           # symmetrize without assuming Symmetric wrapper
    n = size(A, 1)
    if stabilize && epsilon > 0
        ss = T(epsilon) * T(tr(A) / n)
        return Symmetric(Matrix(A) .+ ss .* I(n))
    else
        return Symmetric(Matrix(A))
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Managed Portfolios loader (monthly / daily)
# ──────────────────────────────────────────────────────────────────────────────

"""
    load_managed_portfolios(; data_root=data_dir(),
                              freq=:monthly,
                              type=:US,
                              handling_missing=:Skip,
                              get_dates=false)

Load and horizontally concatenate managed-portfolio panels (serialized `.jls`
matrices) for the requested frequency and dataset type into a single `T×N`
matrix (without the date column). Dates must match across all panels.

Arguments:
- `freq`  :: `:monthly` (default) or `:daily`.
- `type`  :: `:US` (default) or `:International` (international currently implemented for `:daily`).
- `handling_missing` :: `:Median` (row-wise cross-sectional mean imputation) or `:Skip` (drop any row with ≥1 NaN). Rows that are all-NaN are always dropped.
- `get_dates=true` additionally returns the vector of integer dates.

Returns:
- `R::Matrix{Float64}` or `(R, dates::Vector{Int})` if `get_dates=true`.
"""
function load_managed_portfolios(; data_root::AbstractString=data_dir(),
                                   freq::Symbol=:monthly,
                                   type::Symbol=:US,
                                   handling_missing::Symbol=:Skip,
                                   get_dates::Bool=false)

    # --------------------------
    # Choose subdir and base names
    # --------------------------
    subdir = ""                          # initialize to avoid soft-scope/typed-binding issues
    jls_basenames = String[]             # initialize empty list

    if type === :US && freq === :monthly
        subdir = "managed_portfolios_monthly"
        jls_basenames = String[
            "returns_ind17_monthly",
            "returns_bemeinv25_monthly",
            "returns_bemeop25_monthly",
            "returns_meac25_monthly",
            "returns_mebeta25_monthly",
            "returns_meinv25_monthly",
            "returns_meni25_monthly",
            "returns_meop25_monthly",
            "returns_meprior10_monthly",
            "returns_meprior122_monthly",
            "returns_meprior6013_monthly",
            "returns_mevar25_monthly",
            "returns_opinv25_monthly",
            "returns_mebeme25_monthly",
        ]
    elseif type === :US && freq === :daily
        subdir = "managed_portfolios_daily"
        jls_basenames = String[
            "returns_ind49_daily",
            "returns_bemeinv25_daily",
            "returns_bemeop25_daily",
            "returns_meinv25_daily",
            "returns_meop25_daily",
            "returns_meprior10_daily",
            "returns_meprior122_daily",
            "returns_meprior6013_daily",
            "returns_opinv25_daily",
            "returns_mebeme25_daily",
        ]
    elseif type === :International && freq === :daily
        subdir = "managed_portfolios_international_daily"
        jls_basenames = String[
            # Asia Pacific ex Japan
            "returns_apxj_mebeme25_int_daily",
            "returns_apxj_meinv25_int_daily",
            "returns_apxj_meop25_int_daily",
            "returns_apxj_meprior25020_int_daily",
            # Europe
            "returns_eu_mebeme25_int_daily",
            "returns_eu_meinv25_int_daily",
            "returns_eu_meop25_int_daily",
            "returns_eu_meprior25020_int_daily",
            # Japan
            "returns_jp_mebeme25_int_daily",
            "returns_jp_meinv25_int_daily",
            "returns_jp_meop25_int_daily",
            "returns_jp_meprior25020_int_daily",
            # North America
            "returns_na_mebeme25_int_daily",
            "returns_na_meinv25_int_daily",
            "returns_na_meop25_int_daily",
            "returns_na_meprior25020_int_daily",
        ]
    elseif type === :International && freq === :monthly
        error("International monthly panels not configured yet. Provide the monthly .jls list and I’ll add them.")
    else
        error("Unsupported combination: freq=$(freq), type=$(type).")
    end

    isempty(jls_basenames) && error("No panel basenames configured for freq=$(freq), type=$(type).")

    # --------------------------
    # Helper: load a .jls matrix
    # --------------------------
    function _load_jls(name::AbstractString)
        path = joinpath(data_root, subdir, name * ".jls")
        isfile(path) || error("Missing file: ", path)
        open(path, "r") do io
            return deserialize(io)::Matrix{Float64}
        end
    end

    mats = Vector{Matrix{Float64}}(undef, length(jls_basenames))
    dates_ref::Union{Nothing,Vector{Int}} = nothing
    T_ref::Union{Nothing,Int} = nothing

    # --------------------------
    # Load and validate dates
    # --------------------------
    for (i, base) in enumerate(jls_basenames)
        M = _load_jls(base)
        size(M, 2) ≥ 2 || error("Matrix must have at least a date column + one portfolio: ", base)

        dates = Vector{Int}(M[:, 1])
        if dates_ref === nothing
            dates_ref = dates
            T_ref = length(dates)
        else
            length(dates) == T_ref || error("Row count mismatch in ", base,
                                            " (", length(dates), " vs ", T_ref, ")")
            all(dates .== dates_ref) || error("Date vector mismatch in ", base)
        end
        mats[i] = M
    end

    # --------------------------
    # Concatenate columns 2:end
    # --------------------------
    T = T_ref::Int
    total_N = sum(size(M, 2) - 1 for M in mats)

    R = Matrix{Float64}(undef, T, total_N)
    j = 1
    for M in mats
        p = size(M, 2) - 1
        @inbounds R[:, j:(j+p-1)] = M[:, 2:end]
        j += p
    end

    dates_vec = dates_ref::Vector{Int}

    # --------------------------
    # Handle missing values (NaN)
    # --------------------------
    if handling_missing === :Skip
        # Keep rows with no NaNs across any column
        keep = trues(T)
        @inbounds for i in 1:T
            # any isnan? then drop
            for k in 1:total_N
                if isnan(R[i,k])
                    keep[i] = false
                    break
                end
            end
        end
        n_drop = count(!, keep)
        if n_drop > 0
            @warn "Dropping $n_drop rows containing NaNs (handling_missing=:Skip)."
        end
        R = R[keep, :]
        dates_vec = dates_vec[keep]
    elseif handling_missing === :Median
        # Row-wise cross-sectional imputation (ignoring NaNs).
        keep = trues(T)  # may drop rows that are all-NaN
        n_impute = 0
        n_drop = 0
        @inbounds for i in 1:T
            s = 0.0
            c = 0
            # first pass: mean of non-NaN
            for k in 1:total_N
                v = R[i,k]
                if !isnan(v)
                    s += v
                    c += 1
                end
            end
            if c == 0
                # entire row missing: mark to drop
                keep[i] = false
                n_drop += 1
            else
                μ = s / c
                # second pass: fill NaNs with μ
                for k in 1:total_N
                    if isnan(R[i,k])
                        R[i,k] = μ
                        n_impute += 1
                    end
                end
            end
        end
        if n_impute > 0
            @warn "Imputed $n_impute NaN entries with row cross-sectional mean (handling_missing=:Median)."
        end
        if n_drop > 0
            @warn "Dropped $n_drop rows that were entirely NaN across portfolios."
        end
        if any(!, keep)
            R = R[keep, :]
            dates_vec = dates_vec[keep]
        end
    else
        error("handling_missing must be :Median or :Skip (got $(handling_missing))")
    end

    return get_dates ? (R, dates_vec) : R
end

# ──────────────────────────────────────────────────────────────────────────────
# n-choose-k MVE SR (IS/OOS) wrapper
# ──────────────────────────────────────────────────────────────────────────────

"""
    n_choose_k_mve_sr(R, idx_in, idx_out, k; kwargs...) -> NamedTuple

Select a `k`-asset subset **in-sample** from `R[idx_in, :]` using LASSO and MIQP
(vanilla/refit). If `idx_out` is empty, report **in-sample** SRs from the search
methods. If `idx_out` is nonempty, compute weights in-sample and evaluate **out-of-
sample** Sharpe on a separately stabilized covariance.

Returns a named tuple with fields:
- `sr_lasso_vanilla`, `sr_lasso_refit`, `sr_miqp_vanilla`, `sr_miqp_refit`
- `status_lasso_vanilla`, `status_lasso_refit`, `status_miqp_vanilla`, `status_miqp_refit`
- `type ∈ (:in_sample, :out_of_sample)`
"""
function n_choose_k_mve_sr(
    R::AbstractMatrix{<:Real},
    idx_in::AbstractVector{<:Integer},
    idx_out::AbstractVector{<:Integer},
    k::Integer;
    lasso_params::NamedTuple = (;),
    miqp_params::NamedTuple  = (;),
    use_refit_lasso::Bool    = true,
    use_refit_miqp::Bool     = true,
    epsilon_in::Real         = EPS_RIDGE,
    epsilon_out::Real        = EPS_RIDGE,
    stabilize_Σ::Bool        = true,
    do_checks::Bool          = true,
)
    # -----------------------
    # Basic checks
    # -----------------------
    T, N = size(R)
    if do_checks
        N > 0 || throw(ArgumentError("R has zero columns (no assets)."))
        1 ≤ k ≤ N || throw(ArgumentError("k must be in 1..$N"))
        all(1 .≤ idx_in  .≤ T) || throw(ArgumentError("idx_in out of bounds 1..$T"))
        all(1 .≤ idx_out .≤ T) || throw(ArgumentError("idx_out out of bounds 1..$T"))
        length(idx_in) > 1      || throw(ArgumentError("idx_in must have at least 2 observations."))
        if !isempty(idx_out) && length(idx_out) ≤ 1
            throw(ArgumentError("idx_out must have at least 2 observations when provided."))
        end
    end

    # -----------------------
    # In-sample moments (+ stabilized covariance for selection/weights)
    # -----------------------
    Rin   = @view R[idx_in, :]
    Tin   = length(idx_in)
    μ_in  = vec(mean(Rin; dims = 1))
    Σ_in  = cov(Rin; dims = 1)
    Σ_in_prep = _prep_S(Σ_in, epsilon_in, stabilize_Σ)

    # Compute weights only if we will do OOS evaluation
    compute_weights = !isempty(idx_out)

    # -----------------------
    # Searches on in-sample moments (named-tuple results)
    # -----------------------
    res_lasso_vanilla = mve_lasso_relaxation_search(
        μ_in, Σ_in_prep, Tin;
        k = k,
        compute_weights = compute_weights,
        use_refit = false,
        do_checks = false,
        lasso_params...
    )

    res_lasso_refit = mve_lasso_relaxation_search(
        μ_in, Σ_in_prep, Tin;
        k = k,
        compute_weights = compute_weights,
        use_refit = use_refit_lasso,
        do_checks = false,
        lasso_params...
    )

    res_miqp_vanilla = mve_miqp_heuristic_search(
        μ_in, Σ_in_prep;
        k = k,
        compute_weights = compute_weights,
        use_refit = false,
        do_checks = false,
        miqp_params...
    )

    res_miqp_refit = mve_miqp_heuristic_search(
        μ_in, Σ_in_prep;
        k = k,
        compute_weights = compute_weights,
        use_refit = use_refit_miqp,
        do_checks = false,
        miqp_params...
    )

    # -----------------------
    # Evaluation window (+ stabilized covariance for SR eval)
    # -----------------------
    if isempty(idx_out)
        μ_eval, Σ_eval = μ_in, Σ_in_prep    # IS: evaluate on the same stabilized Σ
        typ = :in_sample
    else
        Rout   = @view R[idx_out, :]
        μ_out  = vec(mean(Rout; dims = 1))
        Σ_out  = cov(Rout; dims = 1)
        μ_eval = μ_out
        Σ_eval = _prep_S(Σ_out, epsilon_out, stabilize_Σ)
        typ = :out_of_sample
    end

    # -----------------------
    # Sharpe ratios
    # -----------------------
    if typ === :in_sample
        # Use the SRs computed by the searches (they used Σ_in_prep)
        sr_lasso_vanilla = float(res_lasso_vanilla.sr)
        sr_lasso_refit   = float(res_lasso_refit.sr)
        sr_miqp_vanilla  = float(res_miqp_vanilla.sr)
        sr_miqp_refit    = float(res_miqp_refit.sr)
    else
        # OOS: compute SR on (μ_out, Σ_out_prep) using learned weights
        sr_lasso_vanilla = compute_sr(res_lasso_vanilla.weights, μ_eval, Σ_eval;
                                      selection = res_lasso_vanilla.selection, do_checks = false)
        sr_lasso_refit   = compute_sr(res_lasso_refit.weights, μ_eval, Σ_eval;
                                      selection = res_lasso_refit.selection,   do_checks = false)
        sr_miqp_vanilla  = compute_sr(res_miqp_vanilla.weights, μ_eval, Σ_eval;
                                      selection = res_miqp_vanilla.selection,  do_checks = false)
        sr_miqp_refit    = compute_sr(res_miqp_refit.weights, μ_eval, Σ_eval;
                                      selection = res_miqp_refit.selection,    do_checks = false)
    end

    # -----------------------
    # Collect statuses (pass-through as returned by the solvers)
    # -----------------------
    status_lasso_vanilla = res_lasso_vanilla.status
    status_lasso_refit   = res_lasso_refit.status
    status_miqp_vanilla  = res_miqp_vanilla.status
    status_miqp_refit    = res_miqp_refit.status

    return (
        # SRs
        sr_lasso_vanilla = sr_lasso_vanilla,
        sr_lasso_refit   = sr_lasso_refit,
        sr_miqp_vanilla  = sr_miqp_vanilla,
        sr_miqp_refit    = sr_miqp_refit,
        # Statuses
        status_lasso_vanilla = status_lasso_vanilla,
        status_lasso_refit   = status_lasso_refit,
        status_miqp_vanilla  = status_miqp_vanilla,
        status_miqp_refit    = status_miqp_refit,
        # Type
        type = typ,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# MVE SR decomposition (selection vs estimation)
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_mve_sr_decomposition(μ, Σ, μ̂, Σ̂, k; do_checks=false, return_selection=false)

Decompose the card-`k` Sharpe ratio achieved by a *sample*-selected subset into:

- `mve_sr_cardk_est_term`: Sharpe of the **sample MVE weights** evaluated on the
  **population** moments `(μ, Σ)` over the chosen subset (estimation term).
- `mve_sr_cardk_sel_term`: **Population** MVE Sharpe achievable on the **same**
  chosen subset (selection term).

Returns a named tuple. If `return_selection=true`, also returns the chosen
`selection::Vector{Int}` and `weights::Vector{Float64}` (sample MVE weights).
"""
function compute_mve_sr_decomposition(
    μ ::AbstractVector{<:Real},
    Σ ::AbstractMatrix{<:Real},
    μ̂::AbstractVector{<:Real},
    Σ̂::AbstractMatrix{<:Real},
    k ::Integer;
    do_checks::Bool=false,
    return_selection::Bool=false,
)
    n = length(μ)
    do_checks && begin
        length(μ̂) == n           || throw(ArgumentError("μ̂ length must equal length(μ)"))
        size(Σ)  == (n,n)        || throw(ArgumentError("Σ must be $n×$n"))
        size(Σ̂)  == (n,n)        || throw(ArgumentError("Σ̂ must be $n×$n"))
        1 ≤ k ≤ n                || throw(ArgumentError("k must be in 1..$n"))
    end

    # 1) pick subset of size k from sample moments using the public search API
    res = mve_exhaustive_search(μ̂, Σ̂, k,
        compute_weights=true, do_checks=false)

    A = res.selection
    w = Vector{Float64}(res.weights)   # full-length; zeros outside A

    # 2) estimation term: Sharpe of those weights on population moments
    est_term = compute_sr(w, Vector{Float64}(μ), Matrix{Float64}(Σ); selection=A)

    # 3) selection term: population MVE Sharpe on the same subset
    sel_term = compute_mve_sr(Vector{Float64}(μ), Matrix{Float64}(Σ); selection=A)

    nt = (mve_sr_cardk_est_term = est_term,
          mve_sr_cardk_sel_term = sel_term)

    return return_selection ? merge(nt, (selection=A, weights=w)) : nt
end

# ──────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
    simulate_mve_sr(μ, Σ, T, k; rng=Random.default_rng(), do_checks=false, return_selection=false)

Simulate `T` i.i.d. draws from `N(μ, Σ)`, estimate `(μ̂, Σ̂)` from the sample
(treating the matrix as `T×N`: rows are time, columns are assets), and compute the
MVE-SR decomposition on the subset chosen from sample moments.

Returns a named tuple with estimation and selection terms; optionally include the
chosen `selection` and `weights` when `return_selection=true`.
"""
function simulate_mve_sr(
    μ ::AbstractVector{<:Real},
    Σ ::AbstractMatrix{<:Real},
    T ::Integer,
    k ::Integer;
    rng                     = Random.default_rng(),
    do_checks::Bool         = false,
    return_selection::Bool  = false,
)
    N = length(μ)
    do_checks && begin
        T > 0                 || throw(ArgumentError("T must be positive"))
        size(Σ) == (N,N)     || throw(ArgumentError("Σ must be $N×$N"))
        1 ≤ k ≤ N            || throw(ArgumentError("k must be in 1..$N"))
    end

    # Draw T×N returns with given mean/cov
    L = cholesky(Symmetric(Matrix{Float64}(Σ)), check=true).L
    Z = randn(rng, T, N)
    R = Z * transpose(L) .+ ones(T) * transpose(Vector{Float64}(μ))  # T×N

    # Sample moments (columns=assets as variables)
    μ̂ = vec(mean(R; dims=1))                     # N
    Σ̂ = cov(R; dims=1)                           # N×N (unbiased, 1/(T-1))

    return compute_mve_sr_decomposition(
        Vector{Float64}(μ), Matrix{Float64}(Σ),
        μ̂, Σ̂, k;
        do_checks=do_checks,
        return_selection=return_selection,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Factor-model calibration
# ──────────────────────────────────────────────────────────────────────────────

"""
    calibrate_factor_model(returns, factors; weak_coeff=0.0, idiosy_vol_type=0, do_checks=false)

Calibrate an approximate linear factor model on *time-by-variable* inputs:

- `returns :: T×N` excess returns
- `factors :: T×K` factor returns

Steps (all sample stats treat columns as variables):

1. `μ_f  = mean(factors)`
2. `Σ_f  = cov(factors)`
3. `C_fr = cov(factors, returns)` (K×N)
4. `β    = (Σ_f \\ C_fr)' / N^(weak_coeff/2)` (N×K)
5. `μ    = β * μ_f` (N)
6. Residuals: `R - μ' - F * β'` (T×N)
7. `Σ₀   = σ² I` (homoskedastic) if `idiosy_vol_type==0`, or diag of residual variances if `==1`
8. `Σ    = β Σ_f β' + Σ₀`

Returns `(mu=μ::Vector{Float64}, Sigma=Σ::Matrix{Float64})`.
"""
function calibrate_factor_model(
    returns::AbstractMatrix{<:Real},
    factors::AbstractMatrix{<:Real};
    weak_coeff::Real      = 0.0,
    idiosy_vol_type::Int  = 0,
    do_checks::Bool       = false,
)
    T_r, N = size(returns)
    T_f, K = size(factors)

    do_checks && begin
        T_r == T_f                    || throw(ArgumentError("returns and factors must have same T"))
        0.0 ≤ weak_coeff ≤ 1.0        || throw(ArgumentError("weak_coeff must be in [0,1]"))
        idiosy_vol_type in (0,1)      || throw(ArgumentError("idiosy_vol_type must be 0 or 1"))
    end

    R = Matrix{Float64}(returns)
    F = Matrix{Float64}(factors)

    # Factor moments (columns=variables)
    μf = vec(mean(F; dims=1))         # K
    Σf = cov(F; dims=1)               # K×K

    # Cross-covariance factors→returns: C_fr = cov(F, R) = (F_c' * R_c)/(T-1)
    F_c = F .- ones(T_f) * transpose(vec(mean(F; dims=1)))
    R_c = R .- ones(T_r) * transpose(vec(mean(R; dims=1)))
    C_fr = (transpose(F_c) * R_c) / (T_r - 1)    # K×N

    # Betas N×K; weak factor scaling by N^(weak_coeff/2)
    β = transpose(Σf \ C_fr) ./ (N^(weak_coeff/2))

    # Model-implied means
    μ = β * μf                                 # N

    # Residuals: R - μ' - F β'
    Resid = R .- (ones(T_r) * transpose(μ)) .- (F * transpose(β))  # T×N

    # Idiosyncratic covariance
    resvar = vec(var(Resid; dims=1))          # N
    Σ0 = idiosy_vol_type == 0 ? (mean(resvar) * I(N)) : Diagonal(resvar)

    # Total covariance
    Σ = β * Σf * transpose(β) + Matrix{Float64}(Σ0)

    return (mu = μ, Sigma = Σ)
end

"""
    calibrate_factor_model_from_data(returns_name, factors_name; weak_coeff=0.0,
                                     idiosy_vol_type=0, do_checks=false, dir=data_dir())

Convenience loader: reads `dir/<name>.jls` for both inputs and calls
`calibrate_factor_model`.
"""
function calibrate_factor_model_from_data(
    returns_name::AbstractString,
    factors_name::AbstractString;
    weak_coeff::Real      = 0.0,
    idiosy_vol_type::Int  = 0,
    do_checks::Bool       = false,
    dir::AbstractString   = data_dir(),
)
    R = load_matrix(returns_name; dir)
    F = load_matrix(factors_name; dir)
    return calibrate_factor_model(
        R, F;
        weak_coeff=weak_coeff,
        idiosy_vol_type=idiosy_vol_type,
        do_checks=do_checks,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# Batch simulation driver
# ──────────────────────────────────────────────────────────────────────────────

"""
    compute_simulation_results(n_obs, n_sim, μ, Σ, max_card; rng=Random.default_rng(),
                               save_results=true, file_name="results_portfolios.jls")

Run `n_sim` repetitions for each `T ∈ n_obs` and `k ∈ max_card`. Each repetition:

1. Simulate `T×N` from `N(μ, Σ)`.
2. Select `k` assets via sample MVE.
3. Record:
   - column 1: estimation term (weights on population)
   - column 2: selection term  (population MVE on chosen subset)

Returns a nested `Dict{Int, Dict{Int, Matrix{Float64}}}` keyed by `T` then `k`.
Optionally saves via `Serialization.serialize`.
"""
function compute_simulation_results(
    n_obs   ::AbstractVector{<:Integer},
    n_sim   ::Integer,
    μ       ::AbstractVector{<:Real},
    Σ       ::AbstractMatrix{<:Real},
    max_card::AbstractVector{<:Integer};
    rng                      = Random.default_rng(),
    save_results::Bool       = true,
    file_name::AbstractString = "results_portfolios.jls",
)
    results = Dict{Int, Dict{Int, Matrix{Float64}}}()
    for T in n_obs
        inner = Dict{Int, Matrix{Float64}}()
        for k in max_card
            sims = Matrix{Float64}(undef, n_sim, 2)
            @inbounds for i in 1:n_sim
                out = simulate_mve_sr(μ, Σ, T, k; rng=rng, do_checks=false)
                sims[i, 1] = out.mve_sr_cardk_est_term
                sims[i, 2] = out.mve_sr_cardk_sel_term
            end
            inner[k] = sims
        end
        results[T] = inner
    end

    if save_results
        # ensure parent directory exists (if any)
        dir = dirname(file_name)
        if !isempty(dir)
            mkpath(dir)
        end
        open(file_name, "w") do io
            serialize(io, results)
        end
    end

    return results
end

end # module Utils

