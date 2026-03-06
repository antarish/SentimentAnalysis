"""
=============================================================================
STEP 4: Fama-French Factor Merge & FF5-Adjusted Abnormal Returns — v2
=============================================================================
Paper: Hedged Headlines and Slow Markets

Optimized for local CPU with multiprocessing.
Ryzen 7 3700x (8 cores / 16 threads): estimated ~8-12 minutes.

Inputs:
  ticker_day_with_returns.parquet  — from Script 03

Outputs:
  ticker_day_ff5.parquet           — final modeling dataset with
                                     FF5-adjusted abnormal returns,
                                     FF5-adjusted CAR and DA

Look-ahead bias controls:
  - Factor loadings use ONLY trailing 252-day window ending at t-1
  - Rolling regression uses expanding minimum of 63 days (one quarter)
  - Factor returns on day t are known at end of day t — safe to use
=============================================================================
"""

import io
import os
import zipfile
import logging
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION — local paths for Windows
# ---------------------------------------------------------------------------

INPUT_PATH  = r"C:\Users\cooln\Documents\ECG 755\ticker_day_with_returns.parquet"
FF_CACHE    = r"C:\Users\cooln\Documents\ECG 755\ff5_daily_factors.parquet"
OUTPUT_PATH = r"C:\Users\cooln\Documents\ECG 755\ticker_day_ff5.parquet"

ROLLING_WINDOW = 252
MIN_WINDOW     = 63
FACTORS        = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
N_CORES        = cpu_count()   # auto-detects all 16 threads on 3700x


# ---------------------------------------------------------------------------
# DOWNLOAD FAMA-FRENCH FACTORS
# ---------------------------------------------------------------------------

def download_ff5_factors() -> pd.DataFrame:
    """
    Download FF5 + Momentum daily factors from Kenneth French's website.
    Returns DataFrame with columns: date, mkt_rf, smb, hml, rmw, cma, mom, rf
    All values in decimal form (divided by 100).
    """
    log.info("Downloading Fama-French 5 factors (daily)...")

    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def fetch_french_zip(url: str) -> pd.DataFrame:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = [n for n in z.namelist() if n.upper().endswith(".CSV")][0]
            with z.open(csv_name) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()

        # Find where the actual data rows start (8-digit date)
        data_start = 0
        for i, line in enumerate(lines):
            first_field = line.split(",")[0].strip()
            if first_field.isdigit() and len(first_field) == 8:
                data_start = i
                break

        # Find where data ends (non-numeric line after data)
        data_end = len(lines)
        for i in range(data_start + 1, len(lines)):
            first_field = lines[i].split(",")[0].strip()
            if not first_field.isdigit():
                data_end = i
                break

        from io import StringIO
        df = pd.read_csv(StringIO("\n".join(lines[data_start:data_end])), header=None)
        return df

    # FF5
    ff5_raw = fetch_french_zip(ff5_url)
    ff5_raw.columns = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    ff5_raw["date"] = pd.to_datetime(ff5_raw["date"].astype(str), format="%Y%m%d")
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]:
        ff5_raw[col] = pd.to_numeric(ff5_raw[col], errors="coerce") / 100

    log.info(f"  FF5: {len(ff5_raw):,} days, "
             f"{ff5_raw['date'].min().date()} to {ff5_raw['date'].max().date()}")

    # Momentum
    log.info("Downloading Momentum factor (daily)...")
    mom_raw = fetch_french_zip(mom_url)
    mom_raw.columns = ["date", "mom"]
    mom_raw["date"] = pd.to_datetime(mom_raw["date"].astype(str), format="%Y%m%d")
    mom_raw["mom"]  = pd.to_numeric(mom_raw["mom"], errors="coerce") / 100
    

    factors = ff5_raw.merge(mom_raw, on="date", how="left")
    factors = factors.sort_values("date").reset_index(drop=True)
    log.info(f"  Combined factor dataset: {factors.shape}")

    return factors


def load_or_download_factors() -> pd.DataFrame:
    if os.path.exists(FF_CACHE):
        log.info(f"Loading cached FF factors from {FF_CACHE}")
        return pd.read_parquet(FF_CACHE)
    factors = download_ff5_factors()
    factors.to_parquet(FF_CACHE, index=False)
    log.info(f"FF factors cached to {FF_CACHE}")
    return factors


# ---------------------------------------------------------------------------
# PER-TICKER ROLLING REGRESSION (runs in parallel worker)
# ---------------------------------------------------------------------------

def process_ticker(args):
    """
    Worker function — estimates rolling FF5 loadings for a single ticker
    and returns FF5-adjusted abnormal returns.

    Designed to be called via multiprocessing.Pool.map()
    """
    ticker, t_df = args

    t_df = t_df.sort_values("effective_day0").copy()
    n    = len(t_df)
    ar_ff5 = np.full(n, np.nan)

    for j in range(MIN_WINDOW, n):
        window_start = max(0, j - ROLLING_WINDOW)
        train = t_df.iloc[window_start:j]
        train_clean = train[["excess_return"] + FACTORS].dropna()

        if len(train_clean) < MIN_WINDOW:
            continue

        X_train = train_clean[FACTORS].values
        y_train = train_clean["excess_return"].values

        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_train, y_train)

            row_j       = t_df.iloc[j]
            factor_vals = [row_j.get(f, np.nan) for f in FACTORS]

            if any(np.isnan(v) for v in factor_vals):
                continue

            predicted  = reg.intercept_ + np.dot(reg.coef_, factor_vals)
            ar_ff5[j]  = row_j["excess_return"] - predicted

        except Exception:
            continue

    t_df["AR_ff5"] = ar_ff5
    return t_df[["ticker_clean", "effective_day0", "AR_ff5"]]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    log.info(f"Detected {N_CORES} CPU threads on this machine")

    # --- Load data ---
    log.info(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    df["effective_day0"] = pd.to_datetime(df["effective_day0"])
    df["trading_day"]    = pd.to_datetime(df["trading_day"])
    log.info(f"Loaded {len(df):,} rows, {df['ticker_clean'].nunique()} tickers")

    # --- Load or download FF factors ---
    factors = load_or_download_factors()
    factors["date"] = pd.to_datetime(factors["date"])

    # --- Merge factor returns into main df ---
    log.info("Merging factor returns into ticker-day panel...")
    df = df.merge(
        factors[["date"] + FACTORS + ["rf"]],
        left_on="effective_day0",
        right_on="date",
        how="left"
    ).drop(columns=["date"], errors="ignore")

    log.info(f"Factor merge coverage: {df[FACTORS[0]].notna().mean():.1%}")

    # Excess return = log return - risk free rate
    df["excess_return"] = df["log_return"] - df["rf"].fillna(0)

    # --- Split by ticker for parallel processing ---
    log.info(f"Starting parallel FF5 estimation on {N_CORES} cores...")
    ticker_groups = [
        (ticker, group.copy())
        for ticker, group in df.groupby("ticker_clean")
    ]

    # Use all available cores
    with Pool(processes=N_CORES) as pool:
        results = pool.map(process_ticker, ticker_groups)

    log.info("Parallel estimation complete. Merging results...")

    # --- Merge AR_ff5 back into main df ---
    ar_df = pd.concat(results, ignore_index=True)
    df = df.merge(ar_df, on=["ticker_clean", "effective_day0"], how="left")

    # --- Recompute CAR and DA using FF5-adjusted returns ---
    log.info("Recomputing CAR windows and DA using FF5-adjusted returns...")
    df = df.sort_values(["ticker_clean", "effective_day0"])

    df["CAR_t_ff5"] = df["AR_ff5"]

    df["CAR_t1_t5_ff5"] = (
        df.groupby("ticker_clean")["AR_ff5"]
          .transform(lambda x: x.shift(-1).rolling(5, min_periods=5).sum())
    )

    df["DA_ff5"] = df["CAR_t1_t5_ff5"] - df["CAR_t_ff5"]

    # Robustness horizons (R2 in proposal)
    df["CAR_t1_t3_ff5"] = (
        df.groupby("ticker_clean")["AR_ff5"]
          .transform(lambda x: x.shift(-1).rolling(3, min_periods=3).sum())
    )
    df["CAR_t1_t10_ff5"] = (
        df.groupby("ticker_clean")["AR_ff5"]
          .transform(lambda x: x.shift(-1).rolling(10, min_periods=10).sum())
    )
    df["CAR_t2_t5_ff5"] = (
        df.groupby("ticker_clean")["AR_ff5"]
          .transform(lambda x: x.shift(-2).rolling(4, min_periods=4).sum())
    )

    # --- Summary ---
    log.info(f"\n=== RESULTS SUMMARY ===")
    log.info(f"Total rows:          {len(df):,}")
    log.info(f"Non-null DA_ff5:     {df['DA_ff5'].notna().sum():,} "
             f"({df['DA_ff5'].notna().mean():.1%})")
    log.info(f"\nMarket-adj DA:  mean={df['DA'].mean():.4f}, std={df['DA'].std():.4f}")
    log.info(f"FF5-adj DA:     mean={df['DA_ff5'].mean():.4f}, std={df['DA_ff5'].std():.4f}")
    log.info(f"\nCorrelation DA vs DA_ff5: "
             f"{df[['DA','DA_ff5']].dropna().corr().iloc[0,1]:.4f}")
    log.info(f"\nDA_ff5 summary:\n{df['DA_ff5'].describe().round(4)}")

        # --- Remove duplicates created by factor merge ---
    before = len(df)
    df = df.sort_values(["ticker_clean", "effective_day0", "DA_ff5"],
                        na_position="last")
    df = df.drop_duplicates(subset=["ticker_clean", "effective_day0"], keep="first")
    after = len(df)
    log.info(f"Removed {before - after:,} duplicate rows. Final: {after:,}")

    # --- Save ---
    log.info(f"\nSaving to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Done. Final shape: {df.shape}")


if __name__ == "__main__":
    main()
