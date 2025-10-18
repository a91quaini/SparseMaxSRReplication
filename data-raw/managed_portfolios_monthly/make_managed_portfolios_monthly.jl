# data-raw/managed_portfolios_monthly/make_data_monthly.jl
# ────────────────────────────────────────────────────────
# Read MONTHLY FF5 factors + managed portfolio CSVs,
# convert %→decimals, subtract RF, align by date (YYYYMM),
# and serialize to data/managed_portfolios_monthly/*.jls

using DelimitedFiles   # Base
using Serialization    # Base
using Printf

# -------- paths (script sits WITH the CSVs) --------
scriptdir = @__DIR__   # .../data-raw/managed_portfolios_monthly
rawdir    = scriptdir  # CSVs live here
datadir   = joinpath(scriptdir, "..", "..", "data", "managed_portfolios_monthly")
mkpath(datadir)

# -------- date window (YYYYMM integers) --------
const START = 197101
const STOP  = 202312

# -------- helpers --------
is_yyyymm(s::AbstractString) = occursin(r"^\s*\d{6}\s*$", s)

# Robust monthly reader: tolerate headers/footers and nonnumeric cells
function read_monthly_table(path::AbstractString)
    A, hdr = readdlm(path, ',', Any; header=true)
    header = String.(vec(hdr))

    # Keep only rows whose first cell is YYYYMM (string or number)
    keep = Int[]
    for i in 1:size(A,1)
        x = A[i,1]
        if x isa Number
            d = Int(round(x))
            if 101 ≤ d ≤ 999912
                push!(keep, i)
            end
        elseif x isa AbstractString
            s = strip(x)
            if is_yyyymm(s)
                A[i,1] = parse(Int, s)
                push!(keep, i)
            end
        end
    end
    A = A[keep, :]

    # Build numeric matrix: col1 DATE as Float64, others Float64 (non-parsable → NaN)
    R = Array{Float64}(undef, size(A,1), size(A,2))
    @inbounds for i in 1:size(A,1)
        R[i,1] = Float64(A[i,1])
        for j in 2:size(A,2)
            v = A[i,j]
            if v isa Number
                R[i,j] = float(v)
            else
                s = strip(String(v))
                x = tryparse(Float64, replace(s, '%' => ' '))
                R[i,j] = x === nothing ? NaN : x
            end
        end
    end
    return R, header
end

filter_window!(M::Matrix{Float64}) = begin
    M = M[M[:,1] .>= START, :]
    M = M[M[:,1] .<= STOP, :]
    M
end

percent_to_decimal!(A; from_col=2) = (A[:, from_col:end] .*= 1/100; A)

function rf_from_factors(A::Matrix{Float64}, header::Vector{String})
    idx = findfirst(i -> strip(header[i]) == "RF", eachindex(header))
    idx === nothing && error("RF column not found in factors header: $(join(header, ", "))")
    return A[:, [1, idx]]
end

function align_and_excess(M::Matrix{Float64}, rf::Matrix{Float64})
    rf_dict = Dict{Int,Float64}()
    @inbounds for i in 1:size(rf,1)
        rf_dict[Int(rf[i,1])] = rf[i,2]
    end
    M = filter_window!(M)
    # keep dates present in rf
    keep = map(i -> haskey(rf_dict, Int(M[i,1])), 1:size(M,1))
    M = M[keep, :]
    percent_to_decimal!(M)  # raw % → decimals
    E = similar(M)
    @inbounds for i in 1:size(M,1)
        d = Int(M[i,1]); E[i,1] = d
        rfi = rf_dict[d]
        @views E[i,2:end] .= M[i,2:end] .- rfi
    end
    return E
end

# -------- read monthly FF5 (for RF) --------
ff5_file = joinpath(rawdir, "F-F_Research_Data_5_Factors_2x3.csv")
f_factors, f_header = read_monthly_table(ff5_file)
f_factors = filter_window!(f_factors)
percent_to_decimal!(f_factors)  # convert all factor columns (incl. RF) to decimals
rf = rf_from_factors(f_factors, f_header)

# persist factors + rf (often handy)
open(joinpath(datadir, "factors_ff5_monthly.jls"), "w") do io
    serialize(io, f_factors)
end
open(joinpath(datadir, "rf_monthly.jls"), "w") do io
    serialize(io, rf)
end

# -------- the 17- & 25-portfolio MONTHLY tables --------
portfolio_files = [
    "17_Industry_Portfolios.csv"        => "returns_ind17_monthly",
    "25_Portfolios_BEME_INV_5x5.csv"    => "returns_bemeinv25_monthly",
    "25_Portfolios_BEME_OP_5x5.csv"     => "returns_bemeop25_monthly",
    "25_Portfolios_ME_AC_5x5.csv"       => "returns_meac25_monthly",
    "25_Portfolios_ME_BETA_5x5.csv"     => "returns_mebeta25_monthly",
    "25_Portfolios_ME_INV_5x5.csv"      => "returns_meinv25_monthly",
    "25_Portfolios_ME_NI_5x5.csv"       => "returns_meni25_monthly",
    "25_Portfolios_ME_OP_5x5.csv"       => "returns_meop25_monthly",
    "25_Portfolios_ME_Prior_1_0.csv"    => "returns_meprior10_monthly",
    "25_Portfolios_ME_Prior_12_2.csv"   => "returns_meprior122_monthly",
    "25_Portfolios_ME_Prior_60_13.csv"  => "returns_meprior6013_monthly",
    "25_Portfolios_ME_VAR_5x5.csv"      => "returns_mevar25_monthly",
    "25_Portfolios_OP_INV_5x5.csv"      => "returns_opinv25_monthly",
    "25_Portfolios_5x5.csv"             => "returns_mebeme25_monthly",
]

for (infile, outslug) in portfolio_files
    path = joinpath(rawdir, infile)
    if !isfile(path)
        @warn "Skipping missing file" infile=infile
        continue
    end
    M, _ = read_monthly_table(path)     # DATE + raw % returns (with NaNs tolerated)
    E = align_and_excess(M, rf)         # DATE + EXCESS (decimals), aligned to RF dates
    open(joinpath(datadir, outslug * ".jls"), "w") do io
        serialize(io, E)
    end
    @printf("✓ wrote %s.jls  (rows=%d, cols=%d)\n", outslug, size(E,1), size(E,2))
end

println("\n→ All monthly datasets written under data/managed_portfolios_monthly/*.jls")