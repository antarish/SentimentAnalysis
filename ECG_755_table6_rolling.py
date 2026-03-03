import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score


@dataclass
class Cols:
    date: str = "date"
    ticker: str = "ticker"
    text: str = "text"
    y: str = "y"
    r_trade: str = "r_trade"
    mcap: str | None = None  # optional, for value-weighting


def _maybe_autodetect_mcap(df: pd.DataFrame) -> str | None:
    candidates = ["mcap", "me", "market_cap", "marketcap", "size", "mktcap", "mkvalt", "cap"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _as_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _year_window_ok(years_present: set[int], train_years: list[int], val_years: list[int], test_year: int) -> bool:
    needed = set(train_years + val_years + [test_year])
    return needed.issubset(years_present)


def _fit_score_year(
    df: pd.DataFrame,
    cols: Cols,
    vec: HashingVectorizer,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    alpha_grid: list[float],
) -> tuple[np.ndarray, dict]:
    X_train = vec.transform(df.loc[train_mask, cols.text].fillna(""))
    y_train = df.loc[train_mask, cols.y].astype(int).to_numpy()

    X_val = vec.transform(df.loc[val_mask, cols.text].fillna(""))
    y_val = df.loc[val_mask, cols.y].astype(int).to_numpy()

    best_alpha = None
    best_val_loss = None

    for alpha in alpha_grid:
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            fit_intercept=True,
            random_state=0,
        )
        clf.fit(X_train, y_train)

        p_val = clf.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, p_val, labels=[0, 1])

        if (best_val_loss is None) or (ll < best_val_loss):
            best_val_loss = ll
            best_alpha = alpha

    # Refit on train+val (paper: tune on validation, then use in-sample to score OOS)
    in_mask = train_mask | val_mask
    X_in = vec.transform(df.loc[in_mask, cols.text].fillna(""))
    y_in = df.loc[in_mask, cols.y].astype(int).to_numpy()

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(best_alpha),
        fit_intercept=True,
        random_state=0,
    )
    clf.fit(X_in, y_in)

    X_test = vec.transform(df.loc[test_mask, cols.text].fillna(""))
    p_test = clf.predict_proba(X_test)[:, 1]

    y_test = df.loc[test_mask, cols.y].astype(int).to_numpy()
    acc_test = accuracy_score(y_test, (p_test >= 0.5).astype(int))

    info = {
        "best_alpha": float(best_alpha),
        "val_logloss": float(best_val_loss),
        "test_accuracy": float(acc_test),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    return p_test, info


def score_sentiment_rolling_6_2_1(
    df: pd.DataFrame,
    cols: Cols,
    n_features: int,
    alpha_grid: list[float],
    start_test_year: int | None,
    end_test_year: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[cols.date] = _as_date(df[cols.date])
    df = df.dropna(subset=[cols.date, cols.text, cols.y, cols.r_trade, cols.ticker]).copy()

    df[cols.y] = pd.to_numeric(df[cols.y], errors="coerce")
    df[cols.r_trade] = pd.to_numeric(df[cols.r_trade], errors="coerce")
    df = df.dropna(subset=[cols.y, cols.r_trade]).copy()
    df[cols.y] = df[cols.y].astype(int)

    df["year"] = df[cols.date].dt.year.astype(int)

    years = np.sort(df["year"].unique())
    years_present = set(int(y) for y in years)

    if start_test_year is None:
        start_test_year = int(years.min()) + 8  # need 8 prior years for 6 train + 2 val
    if end_test_year is None:
        end_test_year = int(years.max())

    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    sent = np.full(len(df), np.nan, dtype=float)
    rows = []

    for test_year in range(int(start_test_year), int(end_test_year) + 1):
        train_years = list(range(test_year - 8, test_year - 2))  # 6 years
        val_years = [test_year - 2, test_year - 1]               # 2 years

        if not _year_window_ok(years_present, train_years, val_years, test_year):
            continue

        train_mask = df["year"].isin(train_years).to_numpy()
        val_mask = df["year"].isin(val_years).to_numpy()
        test_mask = (df["year"] == test_year).to_numpy()

        if test_mask.sum() == 0:
            continue

        p_test, info = _fit_score_year(df, cols, vec, train_mask, val_mask, test_mask, alpha_grid)
        sent[test_mask] = p_test

        rows.append(
            {
                "test_year": int(test_year),
                "best_alpha": info["best_alpha"],
                "val_logloss": info["val_logloss"],
                "test_accuracy": info["test_accuracy"],
                "n_train": info["n_train"],
                "n_val": info["n_val"],
                "n_test": info["n_test"],
            }
        )

    df["sent_score"] = sent
    yearly = pd.DataFrame(rows).sort_values("test_year").reset_index(drop=True)
    return df, yearly


def aggregate_to_ticker_day(df_scored: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    # Paper concept: sort stocks; avoid overweighting tickers with many same-day articles.
    agg_dict = {
        "sent_score": "mean",
        cols.r_trade: "first",
    }
    if cols.mcap and cols.mcap in df_scored.columns:
        agg_dict[cols.mcap] = "first"

    td = (
        df_scored.dropna(subset=["sent_score"])
        .groupby([cols.date, cols.ticker], as_index=False)
        .agg(agg_dict)
    )
    return td


def daily_portfolios_quintile(td: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    td = td.dropna(subset=[cols.r_trade, "sent_score"]).copy()

    def assign_quintiles(s: pd.Series) -> pd.Series:
        n = len(s)
        if n < 5:
            return pd.Series(np.nan, index=s.index)
        r = s.rank(method="first")
        return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5])

    td["q"] = td.groupby(cols.date)["sent_score"].transform(assign_quintiles)
    td = td.dropna(subset=["q"]).copy()
    td["q"] = td["q"].astype(int)

    def wavg(x: pd.Series, w: pd.Series) -> float:
        w = pd.to_numeric(w, errors="coerce")
        x = pd.to_numeric(x, errors="coerce")
        m = (~x.isna()) & (~w.isna()) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float((x[m] * w[m]).sum() / w[m].sum())

    out_rows = []
    has_vw = cols.mcap is not None and cols.mcap in td.columns

    for d, g in td.groupby(cols.date):
        gL = g[g["q"] == 5]
        gS = g[g["q"] == 1]

        if len(gL) == 0 or len(gS) == 0:
            continue

        long_ew = float(gL[cols.r_trade].mean())
        short_ew = float(gS[cols.r_trade].mean())
        ls_ew = long_ew - short_ew

        row = {
            cols.date: d,
            "n_names": int(len(g)),
            "long_n": int(len(gL)),
            "short_n": int(len(gS)),
            "long_ew": long_ew,
            "short_ew": short_ew,
            "ls_ew": ls_ew,
        }

        if has_vw:
            long_vw = wavg(gL[cols.r_trade], gL[cols.mcap])
            short_vw = wavg(gS[cols.r_trade], gS[cols.mcap])
            row.update(
                {
                    "long_vw": long_vw,
                    "short_vw": short_vw,
                    "ls_vw": (long_vw - short_vw) if pd.notna(long_vw) and pd.notna(short_vw) else np.nan,
                }
            )

        out_rows.append(row)

    daily = pd.DataFrame(out_rows).sort_values(cols.date).reset_index(drop=True)
    return daily


def perf_stats(daily: pd.DataFrame, col: str) -> dict:
    x = daily[col].dropna()
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    sr = float((mean / sd) * np.sqrt(252)) if sd and sd > 0 else np.nan
    return {"mean": mean, "sd": sd, "sharpe": sr, "n_days": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_ml_v2.csv")
    ap.add_argument("--out_prefix", default="table6_like")
    ap.add_argument("--date_col", default="date")
    ap.add_argument("--ticker_col", default="ticker")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--ret_col", default="r_trade")
    ap.add_argument("--mcap_col", default="auto", help="column name for value-weighting, or 'auto' or ''")
    ap.add_argument("--n_features", type=int, default=2**18)
    ap.add_argument("--alpha_grid", default="1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--start_test_year", type=int, default=0, help="0 => auto (min_year+8)")
    ap.add_argument("--end_test_year", type=int, default=0, help="0 => auto (max_year)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    cols = Cols(
        date=args.date_col,
        ticker=args.ticker_col,
        text=args.text_col,
        y=args.y_col,
        r_trade=args.ret_col,
        mcap=None,
    )

    if args.mcap_col and args.mcap_col != "auto":
        cols.mcap = args.mcap_col
    elif args.mcap_col == "auto":
        cols.mcap = _maybe_autodetect_mcap(df)

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    start_test_year = None if args.start_test_year == 0 else int(args.start_test_year)
    end_test_year = None if args.end_test_year == 0 else int(args.end_test_year)

    df_scored, yearly = score_sentiment_rolling_6_2_1(
        df=df,
        cols=cols,
        n_features=int(args.n_features),
        alpha_grid=alpha_grid,
        start_test_year=start_test_year,
        end_test_year=end_test_year,
    )

    td = aggregate_to_ticker_day(df_scored, cols)
    daily = daily_portfolios_quintile(td, cols)

    stats = {
        "EW Long": perf_stats(daily, "long_ew"),
        "EW Short": perf_stats(daily, "short_ew"),
        "EW L-S": perf_stats(daily, "ls_ew"),
    }
    if "ls_vw" in daily.columns:
        stats.update(
            {
                "VW Long": perf_stats(daily, "long_vw"),
                "VW Short": perf_stats(daily, "short_vw"),
                "VW L-S": perf_stats(daily, "ls_vw"),
            }
        )

    print("\nYearly OOS accuracy (paper reports this by year):")
    if len(yearly):
        print(yearly.to_string(index=False))
    else:
        print("No rolling windows formed (check years / start_test_year).")

    print("\nTable-6-like portfolio stats (daily, annualized Sharpe):")
    for k, v in stats.items():
        print(f"{k}: mean={v['mean']:.6f} sd={v['sd']:.6f} sharpe={v['sharpe']:.3f} n_days={v['n_days']}")

    df_scored.to_csv(f"{args.out_prefix}_article_scored.csv", index=False)
    td.to_csv(f"{args.out_prefix}_ticker_day.csv", index=False)
    daily.to_csv(f"{args.out_prefix}_daily_portfolios.csv", index=False)
    yearly.to_csv(f"{args.out_prefix}_yearly_accuracy.csv", index=False)

    print(f"\nWrote: {args.out_prefix}_article_scored.csv")
    print(f"Wrote: {args.out_prefix}_ticker_day.csv")
    print(f"Wrote: {args.out_prefix}_daily_portfolios.csv")
    print(f"Wrote: {args.out_prefix}_yearly_accuracy.csv")


if __name__ == "__main__":
    main()