# data-raw/make_data.jl
# ─────────────────────
# Reads the 200805–202212 window of each CSV, converts
# percentages→decimals, subtracts RF where appropriate,
# and serializes matrices into ../data/*.jls

using DelimitedFiles    # in Base
using Serialization     # in Base

#–– configure paths and date‐filter
scriptdir = @__DIR__                  # .../data-raw
rawdir    = scriptdir
datadir   = joinpath(scriptdir, "..", "data")
mkpath(datadir)                       # ensure it exists

const START = 200805
const STOP  = 202212

read_raw(fn) = readdlm(joinpath(rawdir, fn), ',',
                       Float64; header=true)  # → (matrix, header_names)

filter!(A) = A[(A[:,1] .>= START) .& (A[:,1] .<= STOP), :]

#–– Section A: FF5 & Momentum
raw_ff5, _ = read_raw("F-F_Research_Data_5_Factors_2x3.csv")
factors_ff5 = filter!(raw_ff5)
factors_ff5[:,2:end] .*= 1/100                     # perc→dec
open(joinpath(datadir,"factors_ff5.jls"),"w") do io
    serialize(io, factors_ff5)
end

raw_mom, _ = read_raw("F-F_Momentum_Factor.csv")
factor_mom = filter!(raw_mom)
factor_mom[:,2:end] .*= 1/100
open(joinpath(datadir,"factor_mom.jls"),"w") do io
    serialize(io, factor_mom)
end

# extract RF (Date + RF‐column is 7th in the FF5 file)
rf = factors_ff5[:, [1,7]]
open(joinpath(datadir,"rf.jls"),"w") do io
    serialize(io, rf)
end

#–– Section B: CRSP excess returns
raw_crsp, _ = read_raw("CRSP_Returns.csv")
returns_crsp = filter!(raw_crsp)
# perc→dec and subtract RF
returns_crsp[:,2:end] .= returns_crsp[:,2:end]./100 .- rf[:,2]
open(joinpath(datadir,"returns_crsp.jls"),"w") do io
    serialize(io, returns_crsp)
end

#–– Section C: the 17‐ and 25‐portfolio tables
portfolio_files = [
    "17_Industry_Portfolios.csv"        => "returns_ind17",
    "25_Portfolios_BEME_INV_5x5.csv"     => "returns_bemeinv25",
    "25_Portfolios_BEME_OP_5x5.csv"      => "returns_bemeop25",
    "25_Portfolios_ME_AC_5x5.csv"        => "returns_meac25",
    "25_Portfolios_ME_BETA_5x5.csv"      => "returns_mebeta25",
    "25_Portfolios_ME_INV_5x5.csv"       => "returns_meinv25",
    "25_Portfolios_ME_NI_5x5.csv"        => "returns_meni25",
    "25_Portfolios_ME_OP_5x5.csv"        => "returns_meop25",
    "25_Portfolios_ME_Prior_1_0.csv"     => "returns_meprior10",
    "25_Portfolios_ME_Prior_12_2.csv"    => "returns_meprior122",
    "25_Portfolios_ME_Prior_60_13.csv"   => "returns_meprior6013",
    "25_Portfolios_ME_VAR_5x5.csv"       => "returns_mevar25",
    "25_Portfolios_OP_INV_5x5.csv"       => "returns_opinv25",
    "25_Portfolios_5x5.csv"              => "returns_mebeme25",
]

for (infile, outname) in portfolio_files
    raw, _ = read_raw(infile)
    M = filter!(raw)
    # percentage→decimal and subtract risk‐free
    M[:,2:end] .= M[:,2:end]./100 .- rf[:,2]
    open(joinpath(datadir, outname*".jls"),"w") do io
        serialize(io, M)
    end
end

println("→ All data written under data/*.jls")
