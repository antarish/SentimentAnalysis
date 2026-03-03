import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


def fit_predict_expanding_no_lookahead(df: pd.DataFrame, n_features: int = 2**18) -> pd.DataFrame:
    """
    No-lookahead expanding scheme by year:
      - first year: train only
      - for each subsequent year yr:
          predict on yr using model trained on < yr
          then update model with yr data
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year

    vec = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        fit_intercept=True,
        random_state=0,
    )

    proba = np.full(len(df), np.nan, dtype=float)
    classes = np.array([0, 1], dtype=int)

    years = np.sort(df["year"].unique())
    first_year = True

    for yr in years:
        idx = df["year"].values == yr
        X = vec.transform(df.loc[idx, "text"].fillna(""))
        y = df.loc[idx, "y"].astype(int).values

        if first_year:
            clf.partial_fit(X, y, classes=classes)
            first_year = False
            continue

        proba[idx] = clf.predict_proba(X)[:, 1]
        clf.partial_fit(X, y)

    df["sent_score"] = proba
    return df


def add_quintiles_and_daily_ls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["sent_score", "r_trade", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # ensure pure daily grouping

    def quintile_group(s: pd.Series) -> pd.Series:
        n = len(s)
        if n < 5:
            return pd.Series(np.nan, index=s.index)  # can't form quintiles
        r = s.rank(method="first")  # breaks ties deterministically
        return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5])

    df["q"] = df.groupby("date")["sent_score"].transform(quintile_group)
    df = df.dropna(subset=["q"]).copy()
    df["q"] = df["q"].astype(int)

    tmp = df.groupby(["date", "q"])["r_trade"].mean().unstack("q")

    daily = pd.DataFrame(
        {
            "long_ret": tmp.get(5),
            "short_ret": tmp.get(1),
            "n_obs": df.groupby("date").size(),
        }
    ).reset_index()

    daily["ls_ret"] = daily["long_ret"] - daily["short_ret"]
    daily = daily.dropna(subset=["long_ret", "short_ret"])
    return daily


def table6_style_stats(daily: pd.DataFrame) -> dict:
    ls = daily["ls_ret"].dropna()
    mean = ls.mean()
    sd = ls.std(ddof=1)
    sr = (mean / sd) * np.sqrt(252) if sd and sd > 0 else np.nan
    return {"ls_mean": mean, "ls_sd": sd, "ls_sharpe": sr}


if __name__ == "__main__":
    df = pd.read_csv("data_ml_v2.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").astype("Int64")
    df["r_trade"] = pd.to_numeric(df["r_trade"], errors="coerce")

    df = df.dropna(subset=["text", "y", "r_trade"]).copy()
    df["y"] = df["y"].astype(int)

    df_scored = fit_predict_expanding_no_lookahead(df, n_features=2**18)
    daily_ls = add_quintiles_and_daily_ls(df_scored)
    stats = table6_style_stats(daily_ls)

    print(stats)

    df_scored.to_csv("data_with_scores.csv", index=False)
    daily_ls.to_csv("daily_ls.csv", index=False)