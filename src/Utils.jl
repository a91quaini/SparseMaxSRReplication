module Utils

using Serialization           # stdlib: deserialize
using Random                  # for randn
using LinearAlgebra           # for cholesky, Symmetric
using Statistics              # for mean, cov
import SparseMaxSR:           # pull in the core functionality
    compute_mve_selection,
    compute_mve_weights,
    compute_sr,
    compute_mve_sr

export load_matrix,
       compute_mve_sr_decomposition,
       simulate_mve_sr,
       calibrate_factor_model,
       calibrate_factor_model_from_data

const DATADIR = joinpath(@__DIR__, "..", "data")

##################################
#### load_matrix
##################################

"""
    load_matrix(name::AbstractString) -> AbstractMatrix{Float64}

Load a serialized matrix from `data/<name>.jls`.
"""
function load_matrix(name::AbstractString)
    path = joinpath(DATADIR, name * ".jls")
    open(path, "r") do io
        return deserialize(io)
    end
end

##################################
#### compute_mve_sr_decomposition
##################################

"""
    compute_mve_sr_decomposition(
        μ, Σ, μ_sample, Σ_sample, k; do_checks=false
    ) -> NamedTuple

Decompose the population‐Sharpe ratio into estimation and selection terms.
"""
function compute_mve_sr_decomposition(
        μ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64},
        μ_sample::AbstractVector{Float64}, Σ_sample::AbstractMatrix{Float64},
        k::Int; do_checks::Bool=false
    )
    n = length(μ)
    if do_checks
        @assert length(μ_sample) == n "μ_sample must match length of μ"
        @assert size(Σ) == (n,n)  "Σ must be n×n"
        @assert size(Σ_sample) == (n,n) "Σ_sample must be n×n"
        @assert 1 ≤ k ≤ n       "k must be between 1 and n"
    end

    # 1) pick your k assets on the *sample* data
    selection = compute_mve_selection(μ_sample, Σ_sample, k)
    @info "Selection idxs:" selection

    # 2) compute sample‐MVE weights on those assets
    weights   = compute_mve_weights(μ_sample, Σ_sample; selection=selection)
    @info "Weights:" weights

    # 3) estimation term: Sharpe of those weights evaluated on (μ,Σ)
    est_term  = compute_sr(weights, μ, Σ; selection=selection)
    @info "Estimation term:" est_term

    # 4) selection term: ideal‐MVE Sharpe on the selected subset
    sel_term  = compute_mve_sr(μ, Σ; selection=selection)
    @info "Selection term:" sel_term

    return (
      mve_sr_cardk_est_term = est_term,
      mve_sr_cardk_sel_term = sel_term,
    )
end

#############################
#### simulate_mve_sr
#############################

"""
    simulate_mve_sr(
        μ, Σ, n_obs, k; do_checks=false
    ) -> NamedTuple

Draw n_obs samples from N(μ,Σ), compute the MVE‐SR decomposition.
"""
function simulate_mve_sr(
        μ::AbstractVector{Float64}, Σ::AbstractMatrix{Float64},
        n_obs::Int, k::Int; do_checks::Bool=false
    )
    if do_checks
        @assert n_obs > 0          "n_obs must be positive"
        @assert 1 ≤ k ≤ length(μ)  "k must be between 1 and n"
    end

    # simulate returns: n×n_obs
    Z = randn(length(μ), n_obs)
    L = cholesky(Symmetric(Σ)).L
    sample = (L * Z) .+ μ

    # sample estimates
    μ_sample = vec(mean(sample; dims=2))
    Σ_sample = cov(eachcol(sample))

    # reuse decomposition
    return compute_mve_sr_decomposition(
        μ, Σ, μ_sample, Σ_sample, k;
        do_checks=do_checks
    )
end

#############################
#### calibrate_factor_model
#############################

"""
    calibrate_factor_model(
      returns::AbstractMatrix{Float64},
      factors::AbstractMatrix{Float64};
      weak_coeff::Real=0.0,
      idiosy_vol_type::Int=0,
      do_checks::Bool=false
    ) -> NamedTuple{(:mu,:sigma),Tuple{AbstractVector{Float64},AbstractMatrix{Float64}}}

Given T×N asset‐return matrix `returns` and T×K factor‐return matrix `factors`,
fits the linear factor model

1.  μ_f = mean of factor returns  
2.  Σ_f = covariance of factor returns  
3.  β  = (Σ_f^{-1} cov(factors,returns))ᵀ ÷ N^(weak_coeff/2) 
4.  μ  = β * μ_f  
5.  residuals = returns .- mean(returns,dims=1) .- factors * βᵀ  
6.  Σ₀ = either homoskedastic (mean var · I) or heteroskedastic (diag(var_i))  
7.  Σ = β Σ_f βᵀ + Σ₀  

Optional inputs:

- `weak_coeff ∈ [0,1]`: scales β by N^(–weak_coeff/2).  
- `idiosy_vol_type ∈ (0,1)`: 0 ⇒ homoskedastic Σ₀, 1 ⇒ diagonal Σ₀.  
- `do_checks`: run dimension/value assertions.  

Returns `(mu=AbstractVector{Float64}, sigma=AbstractMatrix{Float64})`.
"""
function calibrate_factor_model(
    returns::AbstractMatrix{Float64},
    factors::AbstractMatrix{Float64};
    weak_coeff::Real       = 0.0,
    idiosy_vol_type::Int   = 0,
    do_checks::Bool        = false,
)
    T_r, N = size(returns)
    T_f, K = size(factors)
    if do_checks
        @assert T_r == T_f            "row‐counts of returns/factors must match"
        @assert 0.0 ≤ weak_coeff ≤ 1.0 "weak_coeff must be in [0,1]"
        @assert idiosy_vol_type in (0,1) "idiosy_vol_type must be 0 or 1"
    end

    # 1) factor moments
    μ_f     = vec(mean(factors; dims=1))      # K-vector
    Σ_f     = cov(factors; dims=1)            # K×K matrix

    # 2) betas: solve Σ_f * X = C_fr → X = Σ_f \ C_fr, 
    # where C_fr is the covariance between factors and = returns, then transpose
    β       = transpose(Σ_f \ cov(factors, returns)) ./ (N^(weak_coeff/2))

    # 3) model‐implied means
    μ       = β * μ_f                          # N-vector

    # 4) residuals: subtract time‐series means and factor model fit
    μ_r  = vec(mean(returns; dims=1))       # N-vector
    Rcen    = returns .- (ones(T_r,1) * transpose(μ_r))
    Resid   = Rcen .- factors * transpose(β)   # T×N

    # 6) idiosyncratic variances
    resvar1 = vec(var(Resid; dims=1))          # N-vector
    Σ₀      = idiosy_vol_type == 0 ?
                mean(resvar1) * I(N) :
                Diagonal(resvar1)

    # 7) total covariance
    Σ       = β * Σ_f * transpose(β) + Σ₀

    return (μ=μ, Σ=Σ)
end

######################################
#### calibrate_factor_model_from_data
######################################

"""
    calibrate_factor_model_from_data(
      returns_name::String,
      factors_name::String;
      weak_coeff=0.0,
      idiosy_vol_type=0,
      do_checks=false
    )

Loads two matrices from `data/<returns_name>.jls` and `data/<factors_name>.jls`,
then calls `calibrate_factor_model` on them.
"""
function calibrate_factor_model_from_data(
    returns_name::AbstractString,
    factors_name::AbstractString;
    weak_coeff::Real     = 0.0,
    idiosy_vol_type::Int = 0,
    do_checks::Bool      = false,
)
    R, F = load_matrix(returns_name), load_matrix(factors_name)
    return calibrate_factor_model(
      R, F;
      weak_coeff=weak_coeff,
      idiosy_vol_type=idiosy_vol_type,
      do_checks=do_checks,
    )
end

end # module Utils
