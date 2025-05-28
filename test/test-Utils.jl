using Test
using SparseMaxSRReplication

@testset "Utils.load_matrix" begin
    mat = load_matrix("factors_ff5")
    @test isa(mat, Matrix{Float64})
    @test size(mat,1) > 0
    @test size(mat,2) ≥ 2  # date + at least one series
end

@testset "compute_mve_sr_decomposition" begin
    # a trivial 2-asset example where μ_sample=μ and Σ_sample=Σ
    μ = [0.10, 0.20]
    Σ = [1.0 0.3; 0.3 0.5]
    result = compute_mve_sr_decomposition(μ, Σ, μ, Σ, 1)
    @test haskey(result, :mve_sr_cardk_est_term)
    @test haskey(result, :mve_sr_cardk_sel_term)
    # with identical sample and pop, est_term == sel_term
    @test isapprox(result.mve_sr_cardk_est_term,
                   result.mve_sr_cardk_sel_term;
                   atol=1e-8)
end

@testset "simulate_mve_sr" begin
    μ = [0.05, 0.10]
    Σ = [1.0 0.0; 0.0 1.0]
    sim = simulate_mve_sr(μ, Σ, 50, 1)
    @test haskey(sim, :mve_sr_cardk_est_term)
    @test haskey(sim, :mve_sr_cardk_sel_term)
    @test sim.mve_sr_cardk_est_term ≥ 0
    @test sim.mve_sr_cardk_sel_term ≥ 0
end
