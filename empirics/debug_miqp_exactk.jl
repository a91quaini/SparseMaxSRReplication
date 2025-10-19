#!/usr/bin/env julia

using SparseMaxSR
using SparseMaxSR.MIQPHeuristicSearch
using LinearAlgebra, Random, Printf

# Tiny problem to see if solution changes with k when exactly_k=true
Random.seed!(2025)
N = 10
A = randn(N,N)
Σ = Symmetric(A*A' + 0.10I)         # well-conditioned, positive definite
μ = 0.01 .+ 0.04 .* rand(N)

function run_one(k; exactly_k=true)
    r = mve_miqp_heuristic_search(
        μ, Σ;
        k = k,
        γ = 1.0,
        exactly_k = true,
        fmin = zeros(N),
        fmax = ones(N),
        compute_weights = true,
        use_refit = false,
        epsilon = 1e-8,
        stabilize_Σ = false,
        mipgap = 1e-6,
        time_limit = 10.0,
        threads = max(Threads.nthreads()-1,1),
        verbose = false,
        do_checks = true,
        exactly_k = exactly_k,     # <<<<<<<<<<<<<<<<<<<<<<
    )
    sel = r.selection
    s   = sum(abs.(r.weights) .> 1e-10)
    return (status=r.status, sr=r.sr, k=k, S=length(sel), s_active=s, sel=sel, w=r.weights)
end

println("=== EXACTLY_K = true ===")
for k in 2:6
    out = run_one(k; exactly_k=true)
    @printf "k=%d  status=%s  |S|=%d  s_active=%d  SR=%.4f\n" out.k string(out.status) out.S out.s_active out.sr
end

println("\n=== EXACTLY_K = false (<=k) ===")
for k in 2:6
    out = run_one(k; exactly_k=false)
    @printf "k=%d  status=%s  |S|=%d  s_active=%d  SR=%.4f\n" out.k string(out.status) out.S out.s_active out.sr
end
