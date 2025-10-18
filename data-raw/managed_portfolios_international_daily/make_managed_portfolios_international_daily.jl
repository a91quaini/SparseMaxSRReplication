# data-raw/managed_portfolios_international_daily/make_managed_portfolios_international_daily.jl
# ──────────────────────────────────────────────────────────────────────────────
# Read daily international managed-portfolio CSVs (raw % returns), sanity-check
# for missing-value sentinels (< -90.0), convert % → decimals, align all panels
# to the common dates, and serialize to data/managed_portfolios_international_daily/*.jls
#
# Output matrices keep DATE in column 1 and decimal (not excess) returns in 2:end.

using DelimitedFiles   # Base
using Serialization    # Base
using Printf

# ---------------- paths ----------------
scriptdir = @__DIR__   # .../data-raw/managed_portfolios_international_daily
rawdir    = scriptdir
datadir   = joinpath(scriptdir, "..", "..", "data", "managed_portfolios_international_daily")
mkpath(datadir)

# ---------------- optional date window (YYYYMMDD; set to `nothing` for full range) ----------------
const START_DATE = 19910101  # e.g., 19900101
const STOP_DATE  = 20241231  # e.g., 20231231

# ---------------- helpers ----------------
"""
    read_raw_csv(path) -> (A::Matrix{Float64}, header::Vector{String})

Read a CSV with a single header row and numeric body using Base only.
Assumes first column is DATE as YYYYMMDD (integer-like).
"""
function read_raw_csv(path::AbstractString)
    A, hdr = readdlm(path, ',', Float64; header=true)
    header = String.(vec(hdr))
    return (A, header)
end

"""
    check_no_missing_sentinels!(A; from_col=2, threshold=-90.0, replacement=NaN)

Scan columns `from_col:end` for any value `< threshold` (e.g. -99.99, -999).
Replace those entries with `replacement` (default: `NaN`) instead of throwing an error.
Logs the number of replacements and shows up to 10 sample positions.
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

Filter by global START_DATE..STOP_DATE if provided.
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

Divide columns `from_col:end` by 100 in-place.
"""
function percent_to_decimal!(A::Matrix{Float64}; from_col::Int=2)
    A[:, from_col:end] .*= 1/100
    return A
end

"""
    extract_dates(A) -> Vector{Int}

Return DATE column as integers.
"""
extract_dates(A::Matrix{Float64}) = Vector{Int}(round.(Int, A[:,1]))

"""
    align_to_dates(M, keep_dates::Vector{Int}) -> Matrix{Float64}

Filter matrix `M` to the rows whose DATE is in `keep_dates` (must match exactly).
"""
function align_to_dates(M::Matrix{Float64}, keep_dates::Vector{Int})
    # quick index: date -> row
    dict = Dict{Int,Int}()
    @inbounds for i in 1:size(M,1)
        dict[Int(round(M[i,1]))] = i
    end
    # build aligned matrix
    E = Matrix{Float64}(undef, length(keep_dates), size(M,2))
    @inbounds for (i, d) in enumerate(keep_dates)
        r = dict[d]
        E[i, :] = M[r, :]
    end
    return E
end

# ---------------- file mapping (input → output slug) ----------------
# Slugs encode region & sort, and end with _daily. We keep everything lowercase.
const portfolio_files = [
    # Asia Pacific ex Japan
    "Asia_Pacific_ex_Japan_25_Portfolios_ME_BE-ME_Daily.csv"      => "returns_apxj_mebeme25_int_daily",
    "Asia_Pacific_ex_Japan_25_Portfolios_ME_INV_Daily.csv"        => "returns_apxj_meinv25_int_daily",
    "Asia_Pacific_ex_Japan_25_Portfolios_ME_OP_Daily.csv"         => "returns_apxj_meop25_int_daily",
    "Asia_Pacific_ex_Japan_25_Portfolios_ME_Prior_250_20_Daily.csv" => "returns_apxj_meprior25020_int_daily",

    # Europe
    "Europe_25_Portfolios_ME_BE-ME_Daily.csv"                     => "returns_eu_mebeme25_int_daily",
    "Europe_25_Portfolios_ME_INV_Daily.csv"                       => "returns_eu_meinv25_int_daily",
    "Europe_25_Portfolios_ME_OP_Daily.csv"                        => "returns_eu_meop25_int_daily",
    "Europe_25_Portfolios_ME_Prior_250_20_Daily.csv"              => "returns_eu_meprior25020_int_daily",

    # Japan
    "Japan_25_Portfolios_ME_BE-ME_Daily.csv"                      => "returns_jp_mebeme25_int_daily",
    "Japan_25_Portfolios_ME_INV_Daily.csv"                        => "returns_jp_meinv25_int_daily",
    "Japan_25_Portfolios_ME_OP_Daily.csv"                         => "returns_jp_meop25_int_daily",
    "Japan_25_Portfolios_ME_Prior_250_20_Daily.csv"               => "returns_jp_meprior25020_int_daily",

    # North America
    "North_America_25_Portfolios_ME_BE-ME_Daily.csv"              => "returns_na_mebeme25_int_daily",
    "North_America_25_Portfolios_ME_INV_Daily.csv"                => "returns_na_meinv25_int_daily",
    "North_America_25_Portfolios_ME_OP_Daily.csv"                 => "returns_na_meop25_int_daily",
    "North_America_25_Portfolios_ME_Prior_250_20_Daily.csv"       => "returns_na_meprior25020_int_daily",
]

# ---------------- ingest all, check sentinels, and collect common dates ----------------
raw_tables = Dict{String, Matrix{Float64}}()
dates_list = Vector{Vector{Int}}()

for (infile, _slug) in portfolio_files
    path = joinpath(rawdir, infile)
    if !isfile(path)
        @warn "Skipping missing file" infile=infile
        continue
    end

    M, _ = read_raw_csv(path)

    # (0) window filter only on dates (does not touch returns)
    M = filter_window!(M)

    # (1) sentinel check BEFORE any transformation
    check_no_missing_sentinels!(M; from_col=2, threshold=-90.0)

    # keep for later alignment
    raw_tables[infile] = M
    push!(dates_list, extract_dates(M))
end

if isempty(raw_tables)
    error("No input CSVs found. Expected them under: $rawdir")
end

# Intersection of all available dates across the loaded panels
common_dates = reduce(intersect, dates_list)
common_dates = sort(common_dates)

@info "Common date intersection determined" n_panels=length(raw_tables) n_dates=length(common_dates)

# ---------------- convert %→decimals, align by common dates, and write ----------------
n_written = Ref(0)
for (infile, outslug) in portfolio_files
    haskey(raw_tables, infile) || continue
    M = raw_tables[infile]

    # Align rows to the common dates first (still raw %)
    M = align_to_dates(M, common_dates)

    # Convert % → decimals (in-place)
    percent_to_decimal!(M; from_col=2)

    # Persist (DATE in col1, decimals in cols 2:end)
    outpath = joinpath(datadir, outslug * ".jls")
    open(outpath, "w") do io
        serialize(io, M)
    end
    n_written[] += 1
    @printf("✓ wrote %s.jls  (rows=%d, cols=%d)\n", outslug, size(M,1), size(M,2))
end

println("\n→ Wrote $n_written international daily datasets under data/managed_portfolios_international_daily/")
