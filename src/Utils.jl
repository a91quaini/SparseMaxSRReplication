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
       simulate_mve_sr

const DATADIR = joinpath(@__DIR__, "..", "data")

"""
    load_matrix(name::AbstractString) -> Matrix{Float64}

Load a serialized matrix from `data/<name>.jls`.
"""
function load_matrix(name::AbstractString)
    path = joinpath(DATADIR, name * ".jls")
    open(path, "r") do io
        return deserialize(io)
    end
end

"""
    compute_mve_sr_decomposition(
        μ, Σ, μ_sample, Σ_sample, k; do_checks=false
    ) -> NamedTuple

Decompose the population‐Sharpe ratio into estimation and selection terms.
"""
function compute_mve_sr_decomposition(
        μ::Vector{Float64}, Σ::Matrix{Float64},
        μ_sample::Vector{Float64}, Σ_sample::Matrix{Float64},
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

    # 2) compute sample‐MVE weights on those assets
    weights   = compute_mve_weights(μ_sample, Σ_sample; selection=selection)

    # 3) estimation term: Sharpe of those weights evaluated on (μ,Σ)
    est_term  = compute_sr(weights, μ, Σ; selection=selection)

    # 4) selection term: ideal‐MVE Sharpe on the selected subset
    sel_term  = compute_mve_sr(μ, Σ; selection=selection)

    return (
      mve_sr_cardk_est_term = est_term,
      mve_sr_cardk_sel_term = sel_term,
    )
end

"""
    simulate_mve_sr(
        μ, Σ, n_obs, k; max_comb=0, do_checks=false
    ) -> NamedTuple

Draw n_obs samples from N(μ,Σ), compute the MVE‐SR decomposition.
"""
function simulate_mve_sr(
        μ::Vector{Float64}, Σ::Matrix{Float64},
        n_obs::Int, k::Int; max_comb::Int=0, do_checks::Bool=false
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

end # module Utils
