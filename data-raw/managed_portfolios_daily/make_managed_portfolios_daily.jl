# data-raw/make_managed_portfolios_daily.jl
# ───────────────────────────
# Read daily FF5 factors + daily managed portfolios,
# check for sentinel missings (< -90.0) and set them to NaN,
# convert % → decimals, subtract RF, date-align, and
# serialize to data/managed_portfolios_daily/*.jls

using DelimitedFiles       # Base
using Serialization        # Base
using Printf

# -------- paths --------
scriptdir = @__DIR__                                # .../data-raw
rawdir    = scriptdir
datadir   = joinpath(scriptdir, "..", "..", "data", "managed_portfolios_daily")
mkpath(datadir)

# -------- optional window (YYYYMMDD integers); set to nothing for full range --------
const START_DATE = 19910101 # nothing  :: Union{Nothing,Int}
const STOP_DATE  = 20241231 # nothing  :: Union{Nothing,Int}

# -------- helpers --------
"""
    read_raw_csv(path) -> (A::Matrix{Float64}, header::Vector{String})

Read a CSV with a single header row and numeric body using only Base.
Assumes first column is DATE (YYYYMMDD).
"""
function read_raw_csv(path::AbstractString)
    A, hdr = readdlm(path, ',', Float64; header=true)
    header = String.(vec(hdr))   # ensure Vector{String}
    return (A, header)
end

"""
    check_no_missing_sentinels!(A; from_col=2, threshold=-90.0, replacement=NaN)

Scan columns `from_col:end` for any value `< threshold` (e.g. -99.99, -999).
Replace those entries with `replacement` (default: `NaN`) instead of throwing.
Logs the number of replacements (up to 10 sample positions).
"""
function check_no_missing_sentinels!(
    A::Matrix{Float64};
    from_col::Int = 2,
    threshold::Float64 = -90.0,
    replacement::Float64 = NaN
)
    bad_pos = Vector{Tuple{Int,Int}}()
    n_bad = 0
    @inbounds for i in 1:size(A,1), j in from_col:size(A,2)
        val = A[i,j]
        if !isnan(val) && val < threshold
            A[i,j] = replacement
            n_bad += 1
            if length(bad_pos) < 10
                push!(bad_pos, (i,j))
            end
        end
    end
    if n_bad > 0
        sample_str = join(["(row=$(p[1]), col=$(p[2]))" for p in bad_pos], ", ")
        @warn "Replaced $n_bad sentinel values (< $threshold) with $replacement. Sample: $sample_str"
    end
    return A
end

"""
    filter_window!(A)

Filter matrix `A` (first column is date as YYYYMMDD) to START_DATE..STOP_DATE if set.
"""
function filter_window!(A::Matrix{Float64})
    if START_DATE !== nothing
        A = A[A[:,1] .>= START_DATE, :]
    end
    if STOP_DATE !== nothing
        A = A[A[:,1] .<= STOP_DATE, :]
    end
    return A
end

"""
    percent_to_decimal!(A; from_col=2)

In-place: divide columns from `from_col` to end by 100.
"""
function percent_to_decimal!(A::Matrix{Float64}; from_col::Int=2)
    A[:, from_col:end] .*= 1/100
    return A
end

"""
    rf_from_factors(A, header) -> rf::Matrix{Float64}

Extract DATE and RF columns from a factors matrix using header names.
Returns a 2-column matrix [DATE RF].
"""
function rf_from_factors(A::Matrix{Float64}, header::Vector{String})
    rf_idx = findfirst(i -> strip(header[i]) == "RF", eachindex(header))
    rf_idx === nothing && error("RF column not found in factors header: $(join(header, ", "))")
    rf = A[:, [1, rf_idx]]
    return rf
end

"""
    align_and_excess(M, rf) -> E

Given a returns table `M` (DATE + raw % returns) and `rf` (DATE + RF in decimals),
return aligned excess returns matrix `E` where:
- Column 1: DATE
- Columns 2:end: (M%/100 - RF) in decimals
Dates not present in `rf` are dropped.
"""
function align_and_excess(M::Matrix{Float64}, rf::Matrix{Float64})
    # build RF dictionary (date -> RF)
    rf_dict = Dict{Int,Float64}()
    @inbounds for i in 1:size(rf,1)
        rf_dict[Int(rf[i,1])] = rf[i,2]
    end

    # (1) optionally filter M by global window
    M = filter_window!(M)

    # (2) keep only rows where rf exists
    keep = map(i -> haskey(rf_dict, Int(M[i,1])), 1:size(M,1))
    M = M[keep, :]

    # (3) convert % → decimals
    percent_to_decimal!(M)

    # (4) subtract RF (broadcast by row using dict)
    E = similar(M)
    @inbounds for i in 1:size(M,1)
        d = Int(M[i,1])
        E[i,1] = d
        rf_i = rf_dict[d]
        @views E[i,2:end] .= M[i,2:end] .- rf_i
    end
    return E
end

# -------- read FF5 daily (for RF) --------
ff5_file = joinpath(rawdir, "F-F_Research_Data_5_Factors_2x3_daily.csv")
f_factors, f_header = read_raw_csv(ff5_file)
# Clean sentinels BEFORE any transformation
check_no_missing_sentinels!(f_factors; from_col=2, threshold=-90.0, replacement=NaN)
f_factors = filter_window!(f_factors)
percent_to_decimal!(f_factors)                      # includes RF column now in decimals
rf = rf_from_factors(f_factors, f_header)

# persist factors + rf (useful to have both)
open(joinpath(datadir, "factors_ff5_daily.jls"), "w") do io
    serialize(io, f_factors)
end
open(joinpath(datadir, "rf_daily.jls"), "w") do io
    serialize(io, rf)
end

# -------- portfolio files mapping (input → output slug) --------
# All of these are raw % returns; we write excess returns (decimals) after RF subtraction.
portfolio_files = [
    "49_Industry_Portfolios_Daily.csv"          => "returns_ind49_daily",
    "25_Portfolios_5x5_Daily.csv"               => "returns_mebeme25_daily",
    "25_Portfolios_BEME_INV_5x5_daily.csv"      => "returns_bemeinv25_daily",
    "25_Portfolios_BEME_OP_5x5_daily.csv"       => "returns_bemeop25_daily",
    "25_Portfolios_ME_INV_5x5_daily.csv"        => "returns_meinv25_daily",
    "25_Portfolios_ME_OP_5x5_Daily.csv"         => "returns_meop25_daily",
    "25_Portfolios_ME_Prior_1_0_Daily.csv"      => "returns_meprior10_daily",
    "25_Portfolios_ME_Prior_12_2_Daily.csv"     => "returns_meprior122_daily",
    "25_Portfolios_ME_Prior_60_13_Daily.csv"    => "returns_meprior6013_daily",
    "25_Portfolios_OP_INV_5x5_daily.csv"        => "returns_opinv25_daily",
]

# -------- process each portfolio table --------
for (infile, outslug) in portfolio_files
    path = joinpath(rawdir, infile)
    M, _ = read_raw_csv(path)                   # DATE + portfolio % returns

    # Clean sentinels BEFORE any transformation
    check_no_missing_sentinels!(M; from_col=2, threshold=-90.0, replacement=NaN)

    # Align to RF and compute excess (decimals)
    E = align_and_excess(M, rf)

    open(joinpath(datadir, outslug * ".jls"), "w") do io
        serialize(io, E)
    end
    @printf("✓ wrote %s.jls  (rows=%d, cols=%d)\n", outslug, size(E,1), size(E,2))
end

println("\n→ All daily datasets written under data/managed_portfolios_daily/*.jls")
