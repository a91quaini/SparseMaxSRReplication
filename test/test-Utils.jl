ENV["SPARSEMAXSR_DATA"] = joinpath(@__DIR__, "..", "data")

using Test
using Random
using Statistics
using LinearAlgebra
using Serialization
using SparseMaxSRReplication

const DATA_ROOT = joinpath(pkgdir(SparseMaxSRReplication), "data")

# Helper: tiny PD covariance builder
pd2x2(ρ=0.3) = [1.0 ρ; ρ 0.5]

@testset "Utils.load_matrix" begin
    @test isfile(joinpath(DATA_ROOT, "managed_portfolios_monthly", "factors_ff5_monthly.jls"))
    M1 = load_matrix("managed_portfolios_monthly/factors_ff5_monthly"; dir=DATA_ROOT)
    @test isfile(joinpath(DATA_ROOT, "managed_portfolios_monthly", "returns_ind17_monthly.jls"))
    M2 = load_matrix("managed_portfolios_monthly/returns_ind17_monthly"; dir=DATA_ROOT)

    # factors via alias
    @test M1 isa Matrix{Float64}
    @test size(M1,2) ≥ 2

    # explicit monthly file
    @test M2 isa Matrix{Float64}

    # explicit daily file (if present). If not present, skip.
    daily_path = joinpath(data_dir(), "managed_portfolios_daily", "returns_ind49_daily.jls")
    if isfile(daily_path)
        M3 = load_matrix("returns_ind49_daily"; freq=:daily)
        @test M3 isa Matrix{Float64}
    else
        @info "Skipping daily load_matrix test (no daily data present in this build)."
    end

    # error path
    @test_throws ArgumentError load_matrix("this_file_does_not_exist_anywhere")
end

@testset "compute_mve_sr_decomposition" begin
    # Trivial 2-asset case, identical sample/pop => terms must match
    μ  = [0.10, 0.20]
    Σ  = pd2x2(0.3)
    out = compute_mve_sr_decomposition(μ, Σ, μ, Σ, 1; do_checks=true)
    @test haskey(out, :mve_sr_cardk_est_term)
    @test haskey(out, :mve_sr_cardk_sel_term)
    @test isapprox(out.mve_sr_cardk_est_term, out.mve_sr_cardk_sel_term; atol=1e-10)

    # Ask for selection and weights back
    out2 = compute_mve_sr_decomposition(μ, Σ, μ, Σ, 2;
                                        do_checks=true, return_selection=true)
    @test haskey(out2, :selection)
    @test haskey(out2, :weights)
    @test length(out2.selection) == 2
    @test length(out2.weights)   == 2

    # Error paths
    μbad  = [0.1, 0.2, 0.3]
    Σbad  = Matrix(I, 3, 3)
    @test_throws ArgumentError compute_mve_sr_decomposition(μ, Σ, μbad, Σ, 1; do_checks=true)
    @test_throws ArgumentError compute_mve_sr_decomposition(μ, Σ, μ, Σbad, 1; do_checks=true)
    @test_throws ArgumentError compute_mve_sr_decomposition(μ, Σ, μ, Σ, 0; do_checks=true)
    @test_throws ArgumentError compute_mve_sr_decomposition(μ, Σ, μ, Σ, 3; do_checks=true)
end

@testset "simulate_mve_sr" begin
    rng = MersenneTwister(123)
    μ = [0.05, 0.10]
    Σ = Matrix(I, 2, 2)
    sim = simulate_mve_sr(μ, Σ, 50, 1; rng=rng, do_checks=true)
    @test haskey(sim, :mve_sr_cardk_est_term)
    @test isfinite(sim.mve_sr_cardk_est_term)

    # Determinism with same RNG
    rng1 = MersenneTwister(2025)
    rng2 = MersenneTwister(2025)
    a = simulate_mve_sr(μ, Σ, 80, 1; rng=rng1)
    b = simulate_mve_sr(μ, Σ, 80, 1; rng=rng2)
    @test a.mve_sr_cardk_est_term == b.mve_sr_cardk_est_term
    @test a.mve_sr_cardk_sel_term == b.mve_sr_cardk_sel_term

    # With return_selection=true
    sim2 = simulate_mve_sr(μ, Σ, 60, 2; rng=MersenneTwister(7), return_selection=true)
    @test length(sim2.selection) == 2
    @test length(sim2.weights)   == 2

    # Error paths
    @test_throws ArgumentError simulate_mve_sr(μ, Σ, 0, 1; do_checks=true)
    @test_throws ArgumentError simulate_mve_sr(μ, Σ[:,1:1], 10, 2; do_checks=true)
end

@testset "calibrate_factor_model (synthetic)" begin
    rng = MersenneTwister(321)
    T, N, K = 200, 6, 3
    F = randn(rng, T, K)
    βtrue = randn(rng, N, K)

    # Add small idiosyncratic noise
    σϵ = 0.02
    R = F * βtrue' .+ σϵ * randn(rng, T, N)

    # Baseline: homoskedastic Σ0 = σ² I
    res0 = calibrate_factor_model(R, F; weak_coeff=0.0, idiosy_vol_type=0, do_checks=true)
    @test isa(res0.mu, Vector{Float64})
    @test isa(res0.Sigma, Matrix{Float64})
    @test length(res0.mu) == N
    @test size(res0.Sigma) == (N, N)

    # μ ≈ β μ_f
    μ_f = vec(mean(F; dims=1))
    μ_expected = βtrue * μ_f
    @test isapprox(res0.mu, μ_expected; atol=2e-2, rtol=0.15)

    # Σ ≈ β Σ_f β' + σ² I
    Σ_f = cov(F; dims=1)
    Σ_expected = βtrue * Σ_f * βtrue' + (σϵ^2) * I(N)
    # Not exact (finite sample), but close in Frobenius norm
    @test sqrt(sum(abs2, res0.Sigma - Σ_expected)) /
        sqrt(sum(abs2, Σ_expected)) ≤ 0.35

    # Heteroskedastic version: diagonal close to residual variances
    res1 = calibrate_factor_model(R, F; idiosy_vol_type=1)
    @test isapprox(LinearAlgebra.tr(res1.Sigma), LinearAlgebra.tr(res0.Sigma); rtol=0.25)

    # weak_coeff scaling: larger weak_coeff => smaller ||μ||
    res_w0 = calibrate_factor_model(R, F; weak_coeff=0.0)
    res_w1 = calibrate_factor_model(R, F; weak_coeff=1.0)
    @test norm(res_w1.mu) ≤ norm(res_w0.mu) + 1e-12

    # Error paths
    @test_throws ArgumentError calibrate_factor_model(R[1:end-1, :], F; do_checks=true)
    @test_throws ArgumentError calibrate_factor_model(R, F; weak_coeff=-0.1, do_checks=true)
    @test_throws ArgumentError calibrate_factor_model(R, F; weak_coeff=1.1, do_checks=true)
    @test_throws ArgumentError calibrate_factor_model(R, F; idiosy_vol_type=2, do_checks=true)
end

@testset "calibrate_factor_model_from_data (types only)" begin
    mdl = calibrate_factor_model_from_data("managed_portfolios_monthly/returns_ind17_monthly",
                                       "managed_portfolios_monthly/factors_ff5_monthly";
                                       do_checks=true, dir=DATA_ROOT)
    @test isa(mdl.mu, Vector{Float64})
    @test isa(mdl.Sigma, Matrix{Float64})
    @test length(mdl.mu) == size(mdl.Sigma, 1)
end

@testset "compute_simulation_results (small run)" begin
    rng = MersenneTwister(42)
    μ = [0.05, 0.08, 0.10]
    # Mildly correlated PD Σ
    Σ = [1.0 0.2 0.1;
         0.2 0.8 0.3;
         0.1 0.3 0.6]

    n_obs   = [40, 60]
    n_sim   = 5
    max_card = [1, 2]
    tmpfile = joinpath(pwd(), "test_tmp_results.jls")

    results = compute_simulation_results(n_obs, n_sim, μ, Σ, max_card;
                                         rng=rng, save_results=true, file_name=tmpfile)

    # Structure checks
    @test sort(collect(keys(results))) == sort(n_obs)
    for T in n_obs
        inner = results[T]
        @test sort(collect(keys(inner))) == sort(max_card)
        for k in max_card
            sims = inner[k]
            @test size(sims) == (n_sim, 2)
            @test all(!isnan, sims)
        end
    end

    # File written
    @test isfile(tmpfile)

    # Cleanup best-effort
    try; rm(tmpfile; force=true); catch; end
end

@testset "Utils.load_managed_portfolios" begin
    # Helper to deserialize a panel for cross-checks
    _load_panel = function(data_root::AbstractString, subdir::AbstractString, name::AbstractString)
        path = joinpath(data_root, subdir, name * ".jls")
        @test isfile(path) |> identity  # assert exists to make failures clear
        open(path, "r") do io
            return deserialize(io)::Matrix{Float64}
        end
    end

    # Base data dir (defaults to your package's data_dir(); change if needed)
    base_data = data_dir()

    # ======================
    # 1) MONTHLY PANELS
    # ======================
    Rm, dates_m = load_managed_portfolios(; data_root=base_data, freq=:monthly, get_dates=true)

    @test isa(Rm, Matrix{Float64})
    @test isa(dates_m, Vector{Int})
    @test size(Rm, 1) == length(dates_m) > 0
    @test size(Rm, 2) > 0
    @test all(isfinite, Rm)

    # Dates strictly increasing & unique
    @test all(diff(dates_m) .> 0)
    @test allunique(dates_m)

    # Cross-check first column of one constituent file (e.g., ind17)
    ind17m = _load_panel(base_data, "managed_portfolios_monthly", "returns_ind17_monthly")
    @test all(ind17m[:, 1] .== dates_m)
    @test all(isfinite, ind17m[:, 2:end])

    # Column count equals sum across all monthly panels (excluding date col)
    monthly_names = String[
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
    monthly_counts = map(name -> size(_load_panel(base_data, "managed_portfolios_monthly", name), 2) - 1,
                         monthly_names)
    @test sum(monthly_counts) == size(Rm, 2)

    # ======================
    # 2) DAILY PANELS
    # ======================
    Rd, dates_d = load_managed_portfolios(; data_root=base_data, freq=:daily, get_dates=true)

    @test isa(Rd, Matrix{Float64})
    @test isa(dates_d, Vector{Int})
    @test size(Rd, 1) == length(dates_d) > 0
    @test size(Rd, 2) > 0
    @test all(isfinite, Rd)

    # Dates strictly increasing & unique (no assumptions on business days)
    @test all(diff(dates_d) .> 0)
    @test allunique(dates_d)

    # Cross-check first column of one daily panel (e.g., ind49)
    ind49d = _load_panel(base_data, "managed_portfolios_daily", "returns_ind49_daily")
    @test all(ind49d[:, 1] .== dates_d)
    @test all(isfinite, ind49d[:, 2:end])

    # Column count equals sum across all daily panels (excluding date col)
    daily_names = String[
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
    daily_counts = map(name -> size(_load_panel(base_data, "managed_portfolios_daily", name), 2) - 1,
                       daily_names)
    @test sum(daily_counts) == size(Rd, 2)

    # ======================
    # 3) ERROR PATH
    # ======================
    bad_root = joinpath(@__DIR__, "non_existing_path_xyz")
    @test_throws ErrorException load_managed_portfolios(; data_root=bad_root, freq=:monthly)

    println("✅ load_managed_portfolios: monthly and daily passed structural + consistency checks.")
end

@testset "Utils.n_choose_k_mve_sr" begin
    using LinearAlgebra
    Random.seed!(20251017)
    T, N = 120, 8
    μtrue = collect(range(0.02, 0.05; length=N))
    Σtrue = 0.2 .* I(N) .+ 0.1 .* ones(N, N)
    Z = randn(T, N)
    R = Z * cholesky(Symmetric(Σtrue)).L .+ ones(T) * μtrue'

    idx_in  = 1:80
    idx_out = 81:120
    k = 4

    lasso_cfg = (alpha=0.9, nlambda=100, lambda_min_ratio=1e-4, standardize=false)
    miqp_cfg  = (γ=1.0, mipgap=1e-3, time_limit=10.0, threads=0, verbose=false)

    # ---- 1) Out-of-sample evaluation ----
    res_oos = n_choose_k_mve_sr(
        R, collect(idx_in), collect(idx_out), k;
        lasso_params=lasso_cfg, miqp_params=miqp_cfg,
        use_refit_lasso=true, use_refit_miqp=true,
        epsilon_in=1e-6, epsilon_out=1e-6, stabilize_Σ=true,
        do_checks=true,
    )
    @test res_oos.type == :out_of_sample
    @test isfinite(res_oos.sr_lasso_vanilla)
    @test isfinite(res_oos.sr_lasso_refit)
    @test isfinite(res_oos.sr_miqp_vanilla)
    @test isfinite(res_oos.sr_miqp_refit)

    # ---- 2) In-sample evaluation (SRs from search) ----
    res_ins = n_choose_k_mve_sr(
        R, collect(idx_in), Int[], k;
        lasso_params=lasso_cfg, miqp_params=miqp_cfg,
        use_refit_lasso=false, use_refit_miqp=false,
        epsilon_in=1e-6, epsilon_out=1e-6, stabilize_Σ=true,
        do_checks=true,
    )
    @test res_ins.type == :in_sample
    @test isfinite(res_ins.sr_lasso_vanilla)
    @test isfinite(res_ins.sr_lasso_refit)
    @test isfinite(res_ins.sr_miqp_vanilla)
    @test isfinite(res_ins.sr_miqp_refit)

    # ---- 3) Monotonicity ONLY in-sample (refit ≥ vanilla) ----
    @test res_ins.sr_lasso_refit  ≥ res_ins.sr_lasso_vanilla - 1e-8
    @test res_ins.sr_miqp_refit   ≥ res_ins.sr_miqp_vanilla  - 1e-8
    # (No OOS monotonicity assertions.)

    # ---- 4) Consistency when OOS==IS (allow small solver/normalization drift) ----
    res_is_eval_as_oos = n_choose_k_mve_sr(
        R, collect(idx_in), collect(idx_in), k;
        lasso_params=lasso_cfg, miqp_params=miqp_cfg,
        use_refit_lasso=true, use_refit_miqp=true,
        epsilon_in=1e-6, epsilon_out=1e-6, stabilize_Σ=true,
        do_checks=true,
    )
    @test isapprox(res_is_eval_as_oos.sr_lasso_vanilla, res_ins.sr_lasso_vanilla; rtol=0.10, atol=1e-3)
    @test isapprox(res_is_eval_as_oos.sr_lasso_refit,   res_ins.sr_lasso_refit;   rtol=0.10, atol=1e-3)
    @test isapprox(res_is_eval_as_oos.sr_miqp_vanilla,  res_ins.sr_miqp_vanilla;  rtol=0.10, atol=1e-3)
    @test isapprox(res_is_eval_as_oos.sr_miqp_refit,    res_ins.sr_miqp_refit;    rtol=0.10, atol=1e-3)

    # ---- 5) Stabilization toggle still yields finite results ----
    res_oos_nostab = n_choose_k_mve_sr(
        R, collect(idx_in), collect(idx_out), k;
        lasso_params=lasso_cfg, miqp_params=miqp_cfg,
        use_refit_lasso=true, use_refit_miqp=true,
        epsilon_in=0.0, epsilon_out=0.0, stabilize_Σ=false,
        do_checks=true,
    )
    @test isfinite(res_oos_nostab.sr_lasso_vanilla)
    @test isfinite(res_oos_nostab.sr_lasso_refit)
    @test isfinite(res_oos_nostab.sr_miqp_vanilla)
    @test isfinite(res_oos_nostab.sr_miqp_refit)

    # ---- 6) Error paths ----
    @test_throws ArgumentError n_choose_k_mve_sr(R, Int[], Int[], k)
    @test_throws ArgumentError n_choose_k_mve_sr(R, [1], Int[], k)
    @test_throws ArgumentError n_choose_k_mve_sr(R, collect(idx_in), [T], k)
    @test_throws ArgumentError n_choose_k_mve_sr(R, collect(idx_in), collect(idx_out), 0)
    @test_throws ArgumentError n_choose_k_mve_sr(R, collect(idx_in), collect(idx_out), N+1)
    @test_throws ArgumentError n_choose_k_mve_sr(R, [0, 1, 2], Int[], k)
    @test_throws ArgumentError n_choose_k_mve_sr(R, collect(idx_in), [T+1, T+2], k)

    println("✅ n_choose_k_mve_sr passed IS/OOS, stabilization, and error-path tests.")
end



