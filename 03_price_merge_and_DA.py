"""
=============================================================================
STEP 3: Price Merge & Target Variable Construction — v2
=============================================================================
Paper: Hedged Headlines and Slow Markets

Inputs:
  ticker_day_panel.parquet         — from Script 02
  pricing_data_close.csv           — Bloomberg wide format, close prices
  price_data_open.1.csv            — Bloomberg wide format, open prices

Outputs:
  ticker_day_with_returns.parquet  — ticker-day panel with returns + DA

Target variable:
  DA(i,t) = CAR[t+1:t+5] - CAR[t]

LOOK-AHEAD BIAS CONTROLS (critical):
  1. Time-bin adjusted Day 0 assignment:
       Pre-market    → Day 0 = same trading day
       Market hours  → Day 0 = same trading day
       After-market  → Day 0 = NEXT trading day
     This ensures no headline ever sees the return it could not have
     influenced. After-market headlines physically cannot affect same-day
     close prices.

  2. All rolling features (momentum, volatility) use shift(1) so today's
     return is never included in its own predictor.

  3. CAR[t+1:t+5] is strictly forward-looking — it is the TARGET variable
     only and never appears as a feature anywhere in the pipeline.

  4. Factor loadings (Script 04) estimated on trailing 252-day window only.

  5. Train/test split uses strictly expanding window — never random split
     on time series data.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

TICKER_DAY_INPUT  = "/content/drive/MyDrive/(H)edging Paper/ticker_day_panel.parquet"
CLOSE_PATH        = "/content/drive/MyDrive/(H)edging Paper/Price Data/pricing_data_close.csv"
OPEN_PATH         = "/content/drive/MyDrive/(H)edging Paper/Price Data/price_data_open.1.csv"
OUTPUT_PATH       = "/content/drive/MyDrive/(H)edging Paper/ticker_day_with_returns.parquet"

MIN_HISTORY_DAYS  = 60   # minimum price history before ticker-day is included


# ---------------------------------------------------------------------------
# PARSE BLOOMBERG WIDE FORMAT
# ---------------------------------------------------------------------------

def parse_bloomberg_wide(path: str, price_type: str = "close") -> pd.DataFrame:
    """
    Parse Bloomberg-style wide CSV export into long format.

    Bloomberg export structure:
      Row 0: Start Date metadata
      Row 1: End Date metadata
      Row 2: empty
      Row 3: [blank], AAPL US Equity, ABBV US Equity, ...
      Row 4: [blank], Last Price, Last Price, ...
      Row 5+: [date], [price], [price], ...

    Returns DataFrame with columns: date, ticker, {price_type}
    """
    log.info(f"Parsing {price_type} prices from {path}...")

    raw = pd.read_csv(path, header=None, skiprows=0, low_memory=False)

    # Extract ticker names from row 3, strip Bloomberg suffix
    tickers = raw.iloc[3, 1:].tolist()
    tickers = [str(t).replace(" US Equity", "").strip() for t in tickers]

    # Price data starts at row 5
    price_data = raw.iloc[5:, :].copy()
    price_data.columns = ["date"] + tickers
    price_data = price_data.dropna(subset=["date"])

    # Parse dates
    price_data["date"] = pd.to_datetime(
        price_data["date"], format="mixed", dayfirst=False
    )
    price_data = price_data.dropna(subset=["date"])
    price_data = price_data.sort_values("date").reset_index(drop=True)

    # Melt wide → long
    price_long = price_data.melt(
        id_vars="date",
        var_name="ticker",
        value_name=price_type
    )

    price_long[price_type] = pd.to_numeric(price_long[price_type], errors="coerce")
    price_long = price_long.dropna(subset=[price_type])
    price_long = price_long[price_long[price_type] > 0]

    log.info(f"  {price_type}: {price_long.shape[0]:,} rows, "
             f"{price_long['ticker'].nunique()} tickers, "
             f"{price_long['date'].min().date()} to {price_long['date'].max().date()}")

    return price_long


# ---------------------------------------------------------------------------
# LOOK-AHEAD SAFE DAY 0 ASSIGNMENT
# ---------------------------------------------------------------------------

def build_trading_day_map(dates: pd.Series) -> pd.Series:
    """
    Build a map from each trading date to the NEXT trading date.
    Used to shift after-market headlines forward by one trading day.
    Non-trading days (weekends, holidays) are skipped automatically
    because the price data only contains actual trading days.
    """
    sorted_dates = sorted(dates.unique())
    next_day_map = {
        d: sorted_dates[i + 1]
        for i, d in enumerate(sorted_dates[:-1])
    }
    return next_day_map


def adjust_trading_day_for_timebin(ticker_day: pd.DataFrame,
                                    next_day_map: dict) -> pd.DataFrame:
    """
    CRITICAL LOOK-AHEAD BIAS PREVENTION:

    Adjust the effective Day 0 based on the dominant time bin of headlines
    for each ticker-day observation.

    Rule:
      - Pre-market or Market hours → Day 0 = trading_day (unchanged)
      - After-market               → Day 0 = next trading day

    Rationale: An after-market headline published at 5pm cannot affect
    that day's closing price (market already closed). Assigning it to
    the same day would leak future return information into the feature set.

    The 'time_bin' column in ticker_day reflects the DOMINANT bin across
    all headlines for that ticker-day (most common bin by headline count).
    """
    log.info("Adjusting Day 0 assignment for time-bin look-ahead safety...")

    # Map after-market rows forward by one trading day
    def safe_day0(row):
        bin_val = str(row.get("time_bin", "")).lower()
        if "after" in bin_val:
            # Shift to next trading day
            return next_day_map.get(row["trading_day"], row["trading_day"])
        else:
            # Pre-market or market hours: same day is correct
            return row["trading_day"]

    ticker_day["effective_day0"] = ticker_day.apply(safe_day0, axis=1)

    n_shifted = (ticker_day["effective_day0"] != ticker_day["trading_day"]).sum()
    log.info(f"  After-market rows shifted to next trading day: {n_shifted:,} "
             f"({n_shifted / len(ticker_day):.1%} of ticker-day obs)")

    return ticker_day


# ---------------------------------------------------------------------------
# COMPUTE RETURNS
# ---------------------------------------------------------------------------

def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns per ticker.
    r_it = log(P_it / P_i,t-1)

    Log returns are additive: CAR[t1:t2] = sum(r_t1 ... r_t2)
    """
    log.info("Computing daily log returns...")
    close = close.sort_values(["ticker", "date"])

    close["log_return"] = (
        close.groupby("ticker")["close"]
             .transform(lambda x: np.log(x / x.shift(1)))
    )

    close = close.dropna(subset=["log_return"])

    # Quality flags
    close["is_penny"]    = close["close"] < 1.0
    close["return_flag"] = close["log_return"].abs() > 0.5

    log.info(f"  Penny stock obs:      {close['is_penny'].sum():,}")
    log.info(f"  Extreme return flags: {close['return_flag'].sum():,}")

    return close


def compute_market_return(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weighted market return = cross-sectional mean of clean returns.
    Excludes penny stocks and extreme return days from the market benchmark.
    """
    log.info("Computing equal-weighted market return...")
    mkt = (
        returns[~returns["is_penny"] & ~returns["return_flag"]]
        .groupby("date")["log_return"]
        .mean()
        .reset_index()
        .rename(columns={"log_return": "market_return"})
    )
    return mkt


def compute_abnormal_returns(returns: pd.DataFrame,
                              market: pd.DataFrame) -> pd.DataFrame:
    """
    Market-adjusted abnormal return:
    AR_it = r_it - r_mt

    Note: FF5-adjusted abnormal returns computed in Script 04.
    Factor loadings estimated on trailing 252-day rolling window (no look-ahead).
    """
    log.info("Computing market-adjusted abnormal returns...")
    df = returns.merge(market, on="date", how="left")
    df["abnormal_return"] = df["log_return"] - df["market_return"]
    return df


# ---------------------------------------------------------------------------
# COMPUTE CAR AND DA
# ---------------------------------------------------------------------------

def compute_car_and_da(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CAR windows and DA target variable.

    LOOK-AHEAD SAFETY:
      - CAR[t] uses same-day abnormal return — this is the OUTCOME on Day 0,
        not a predictor. It is only used as target variable Target 3.
      - CAR[t+1:t+5] uses shift(-1) to look strictly forward.
        This is Target variable 2 and part of DA (Target 1).
      - Prior return controls use shift(1) — strictly backward looking.
      - Realized volatility uses shift(1) rolling std — strictly backward.

    DA(i,t) = CAR[t+1:t+5] - CAR[t]  ← PRIMARY TARGET VARIABLE
    """
    log.info("Computing CAR windows and DA...")

    df = df.sort_values(["ticker", "date"]).copy()

    # --- Target variables (forward-looking by design) ---
    df["CAR_t"]     = df["abnormal_return"]   # Day 0 reaction

    df["CAR_t1_t5"] = (                        # Post-news drift [t+1, t+5]
        df.groupby("ticker")["abnormal_return"]
          .transform(lambda x: x.shift(-1).rolling(5, min_periods=5).sum())
    )

    df["DA"] = df["CAR_t1_t5"] - df["CAR_t"]  # PRIMARY TARGET

    # --- Robustness horizons (R2 in proposal) ---
    df["CAR_t1_t3"] = (
        df.groupby("ticker")["abnormal_return"]
          .transform(lambda x: x.shift(-1).rolling(3, min_periods=3).sum())
    )
    df["CAR_t1_t10"] = (
        df.groupby("ticker")["abnormal_return"]
          .transform(lambda x: x.shift(-1).rolling(10, min_periods=10).sum())
    )
    df["CAR_t2_t5"] = (
        df.groupby("ticker")["abnormal_return"]
          .transform(lambda x: x.shift(-2).rolling(4, min_periods=4).sum())
    )

    # --- Control variables (strictly backward-looking, shift(1) enforced) ---
    df["prior_return_1d"] = (
        df.groupby("ticker")["log_return"]
          .transform(lambda x: x.shift(1))          # yesterday's return
    )
    df["prior_return_5d"] = (
        df.groupby("ticker")["log_return"]
          .transform(lambda x: x.shift(1).rolling(5, min_periods=3).sum())
    )
    df["prior_return_20d"] = (
        df.groupby("ticker")["log_return"]
          .transform(lambda x: x.shift(1).rolling(20, min_periods=10).sum())
    )
    df["realized_vol_20d"] = (
        df.groupby("ticker")["log_return"]
          .transform(lambda x: x.shift(1).rolling(20, min_periods=10).std())
    )

    log.info(f"  Non-null DA rows: {df['DA'].notna().sum():,}")
    log.info(f"\n  DA summary:\n{df['DA'].describe().round(4)}")
    log.info(f"\n  CAR_t summary:\n{df['CAR_t'].describe().round(4)}")
    log.info(f"\n  CAR_t1_t5 summary:\n{df['CAR_t1_t5'].describe().round(4)}")

    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    # --- Load ticker-day panel ---
    log.info(f"Loading ticker-day panel...")
    ticker_day = pd.read_parquet(TICKER_DAY_INPUT)
    ticker_day["trading_day"] = pd.to_datetime(ticker_day["trading_day"])
    log.info(f"Ticker-day panel: {ticker_day.shape}")

    # --- Parse price files ---
    close  = parse_bloomberg_wide(CLOSE_PATH, price_type="close")
    open_  = parse_bloomberg_wide(OPEN_PATH,  price_type="open")

    # --- Compute returns ---
    prices = compute_returns(close)
    market = compute_market_return(prices)
    prices = compute_abnormal_returns(prices, market)
    prices = compute_car_and_da(prices)

    # --- Build next-trading-day map for time-bin adjustment ---
    next_day_map = build_trading_day_map(prices["date"])

    # --- Adjust Day 0 for after-market headlines ---
    ticker_day = adjust_trading_day_for_timebin(ticker_day, next_day_map)

    # --- Merge on effective_day0 (look-ahead safe) ---
    log.info("Merging returns into ticker-day panel on effective_day0...")
    price_cols = [
        "date", "ticker",
        "close", "open",
        "log_return", "abnormal_return", "market_return",
        "CAR_t", "CAR_t1_t5", "CAR_t1_t3", "CAR_t1_t10", "CAR_t2_t5",
        "DA",
        "prior_return_1d", "prior_return_5d", "prior_return_20d",
        "realized_vol_20d",
        "is_penny", "return_flag"
    ]
    prices_clean = prices[[c for c in price_cols if c in prices.columns]].copy()

    ticker_day = ticker_day.merge(
        prices_clean,
        left_on  = ["ticker_clean", "effective_day0"],  # ← time-bin adjusted
        right_on = ["ticker", "date"],
        how      = "left"
    )
    ticker_day.drop(columns=["ticker", "date"], errors="ignore", inplace=True)

    # --- Coverage report ---
    n_total   = len(ticker_day)
    n_with_da = ticker_day["DA"].notna().sum()
    n_shifted = (ticker_day["effective_day0"] != ticker_day["trading_day"]).sum()

    log.info(f"\n=== MERGE COVERAGE ===")
    log.info(f"Total ticker-day rows:          {n_total:,}")
    log.info(f"Rows with DA:                   {n_with_da:,} ({n_with_da/n_total:.1%})")
    log.info(f"After-market shifted rows:      {n_shifted:,} ({n_shifted/n_total:.1%})")
    log.info(f"\nDA summary:\n{ticker_day['DA'].describe().round(4)}")

    # --- Save ---
    log.info(f"\nSaving to {OUTPUT_PATH}...")
    ticker_day.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Done. Final shape: {ticker_day.shape}")


if __name__ == "__main__":
    main()