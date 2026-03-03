import argparse
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.linear_model import SGDClassifier


def clean_text(s: str, max_chars: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").strip()
    if not s:
        s = "."
    if max_chars and len(s) > max_chars:
        s = s[:max_chars]
    return s


def ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def build_or_load_unique_texts(df: pd.DataFrame, text_col: str, cache_dir: Path, max_chars: int):
    uniq_path = cache_dir / "unique_texts.parquet"
    cleaned = df[text_col].fillna("").astype(str).map(lambda x: clean_text(x, max_chars))

    if uniq_path.exists():
        uniq = pd.read_parquet(uniq_path)["text"].astype(str).tolist()
        idx = pd.Index(uniq)
        codes = idx.get_indexer(cleaned)
        if (codes < 0).any():
            missing = int((codes < 0).sum())
            raise RuntimeError(
                f"{missing} texts not found in cached unique_texts. "
                f"Delete {cache_dir} to rebuild, or keep cleaning/max_chars identical."
            )
        return cleaned, codes.astype(np.int32), uniq

    codes, uniq = pd.factorize(cleaned, sort=False)
    codes = codes.astype(np.int32)
    pd.DataFrame({"text": uniq.astype(str)}).to_parquet(uniq_path, index=False)
    return cleaned, codes, uniq.astype(str).tolist()


def open_or_create_memmap(cache_dir: Path, n_unique: int, dim: int):
    mm_path = cache_dir / "embeddings.f32"
    done_path = cache_dir / "done.npy"

    if mm_path.exists() and done_path.exists():
        emb = np.memmap(mm_path, dtype=np.float32, mode="r+", shape=(n_unique, dim))
        done = np.load(done_path)
        if done.shape[0] != n_unique:
            raise RuntimeError("Cache size mismatch (unique_texts changed). Delete cache dir and rerun.")
        return emb, done

    emb = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_unique, dim))
    done = np.zeros(n_unique, dtype=bool)
    np.save(done_path, done)
    return emb, done


def embed_texts_with_backoff(client: OpenAI, model: str, batch_texts: list[str], max_retries: int = 8):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=batch_texts)
            return [d.embedding for d in resp.data]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)


def build_embeddings_cache(uniq_texts: list[str], cache_dir: Path, model: str, embed_batch: int):
    client = OpenAI()

    probe = embed_texts_with_backoff(client, model, [uniq_texts[0]])[0]
    dim = len(probe)

    emb, done = open_or_create_memmap(cache_dir, len(uniq_texts), dim)

    if not done[0]:
        emb[0, :] = np.asarray(probe, dtype=np.float32)
        done[0] = True
        np.save(cache_dir / "done.npy", done)

    remaining = np.where(~done)[0]
    if remaining.size == 0:
        return dim

    for start in range(0, remaining.size, embed_batch):
        idxs = remaining[start : start + embed_batch]
        batch = [uniq_texts[i] for i in idxs.tolist()]
        vecs = embed_texts_with_backoff(client, model, batch)

        for i, v in zip(idxs, vecs):
            emb[i, :] = np.asarray(v, dtype=np.float32)

        done[idxs] = True
        emb.flush()
        np.save(cache_dir / "done.npy", done)

        done_ct = int(done.sum())
        if done_ct % 5000 < embed_batch:
            print(f"Embedded {done_ct}/{len(done)} unique texts")

    return dim


def iter_batches(row_idx: np.ndarray, batch_size: int):
    for s in range(0, row_idx.size, batch_size):
        yield row_idx[s : s + batch_size]


def stream_fit_sgd(
    clf: SGDClassifier,
    X_mm: np.memmap,
    emb_idx: np.ndarray,
    y: np.ndarray,
    row_idx: np.ndarray,
    batch_size: int,
):
    classes = np.array([0, 1], dtype=int)
    first = True
    for b in iter_batches(row_idx, batch_size):
        Xb = X_mm[emb_idx[b], :]
        yb = y[b]
        if first:
            clf.partial_fit(Xb, yb, classes=classes)
            first = False
        else:
            clf.partial_fit(Xb, yb)


def stream_logloss_and_acc(
    clf: SGDClassifier,
    X_mm: np.memmap,
    emb_idx: np.ndarray,
    y: np.ndarray,
    row_idx: np.ndarray,
    batch_size: int,
):
    eps = 1e-12
    loss_sum = 0.0
    n = 0
    correct = 0

    for b in iter_batches(row_idx, batch_size):
        Xb = X_mm[emb_idx[b], :]
        yb = y[b]
        pb = clf.predict_proba(Xb)[:, 1]
        pb = np.clip(pb, eps, 1.0 - eps)

        loss_sum += float((-yb * np.log(pb) - (1 - yb) * np.log(1 - pb)).sum())
        correct += int(((pb >= 0.5).astype(int) == yb).sum())
        n += yb.size

    return (loss_sum / n) if n else np.nan, (correct / n) if n else np.nan


def rolling_6_2_1_sentiment(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    emb_idx: np.ndarray,
    X_mm: np.memmap,
    alpha_grid: list[float],
    train_batch: int,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col, y_col]).copy()
    df["year"] = df[date_col].dt.year.astype(int)

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    keep = ~np.isnan(y)
    df = df.loc[keep].copy()
    y = y[keep].astype(int)
    emb_idx = emb_idx[keep]

    years = np.sort(df["year"].unique())
    years_set = set(int(v) for v in years)
    start_test_year = int(years.min()) + 8
    end_test_year = int(years.max())

    sent = np.full(len(df), np.nan, dtype=float)
    yearly_rows = []

    idx_all = np.arange(len(df), dtype=np.int32)

    for test_year in range(start_test_year, end_test_year + 1):
        train_years = list(range(test_year - 8, test_year - 2))
        val_years = [test_year - 2, test_year - 1]

        needed = set(train_years + val_years + [test_year])
        if not needed.issubset(years_set):
            continue

        train_idx = idx_all[df["year"].isin(train_years).to_numpy()]
        val_idx = idx_all[df["year"].isin(val_years).to_numpy()]
        test_idx = idx_all[(df["year"].to_numpy() == test_year)]

        if test_idx.size == 0:
            continue

        best_alpha = None
        best_val_loss = None

        for alpha in alpha_grid:
            clf = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=float(alpha),
                fit_intercept=True,
                random_state=0,
            )
            stream_fit_sgd(clf, X_mm, emb_idx, y, train_idx, train_batch)
            val_loss, _ = stream_logloss_and_acc(clf, X_mm, emb_idx, y, val_idx, train_batch)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alpha = float(alpha)

        in_idx = np.concatenate([train_idx, val_idx])
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=float(best_alpha),
            fit_intercept=True,
            random_state=0,
        )
        stream_fit_sgd(clf, X_mm, emb_idx, y, in_idx, train_batch)

        test_loss, test_acc = stream_logloss_and_acc(clf, X_mm, emb_idx, y, test_idx, train_batch)

        probs = []
        for b in iter_batches(test_idx, train_batch):
            Xb = X_mm[emb_idx[b], :]
            probs.append(clf.predict_proba(Xb)[:, 1])
        sent[test_idx] = np.concatenate(probs)

        yearly_rows.append(
            {
                "test_year": int(test_year),
                "best_alpha": float(best_alpha),
                "val_logloss": float(best_val_loss),
                "test_logloss": float(test_loss),
                "test_accuracy": float(test_acc),
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "n_test": int(test_idx.size),
            }
        )
        print(f"Finished OOS year {test_year} (alpha={best_alpha}, acc={test_acc:.4f})")

    df["sent_score"] = sent
    yearly = pd.DataFrame(yearly_rows).sort_values("test_year").reset_index(drop=True)
    return df, yearly


def aggregate_ticker_day(df_scored: pd.DataFrame, date_col: str, ticker_col: str, ret_col: str):
    return (
        df_scored.dropna(subset=["sent_score", ret_col, date_col, ticker_col])
        .groupby([date_col, ticker_col], as_index=False)
        .agg({"sent_score": "mean", ret_col: "first"})
    )


def daily_quintile_portfolios(td: pd.DataFrame, date_col: str, ret_col: str):
    td = td.copy()

    def assign_quintiles(s: pd.Series) -> pd.Series:
        if len(s) < 5:
            return pd.Series(np.nan, index=s.index)
        r = s.rank(method="first")
        return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5])

    td["q"] = td.groupby(date_col)["sent_score"].transform(assign_quintiles)
    td = td.dropna(subset=["q"]).copy()
    td["q"] = td["q"].astype(int)

    tmp = td.groupby([date_col, "q"])[ret_col].mean().unstack("q")
    daily = pd.DataFrame({date_col: tmp.index, "long_ew": tmp.get(5), "short_ew": tmp.get(1)}).reset_index(drop=True)
    daily["ls_ew"] = daily["long_ew"] - daily["short_ew"]
    return daily.dropna(subset=["long_ew", "short_ew"])


def perf_stats(x: pd.Series):
    x = x.dropna()
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    sharpe = float((mean / sd) * np.sqrt(252)) if sd and sd > 0 else np.nan
    return mean, sd, sharpe, int(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_ml_v2.csv")
    ap.add_argument("--out_prefix", default="chatgpt_like")
    ap.add_argument("--model", default="text-embedding-3-large")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--date_col", default="signal_date")
    ap.add_argument("--ticker_col", default="ticker")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--ret_col", default="r_trade")
    ap.add_argument("--max_chars", type=int, default=20000)
    ap.add_argument("--embed_batch", type=int, default=128)
    ap.add_argument("--train_batch", type=int, default=4096)
    ap.add_argument("--alpha_grid", default="1e-5,1e-4,1e-3,1e-2")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY not set. Run: setx OPENAI_API_KEY "..." then reopen terminal.')

    df = pd.read_csv(args.input)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(f"emb_cache_{args.model}")
    ensure_cache_dir(cache_dir)

    cleaned, emb_codes, uniq_texts = build_or_load_unique_texts(df, args.text_col, cache_dir, args.max_chars)
    df["_emb_idx"] = emb_codes

    print(f"Rows: {len(df)} | Unique texts: {len(uniq_texts)}")
    print("Building/Loading embeddings cache...")
    dim = build_embeddings_cache(uniq_texts, cache_dir, args.model, args.embed_batch)
    print(f"Embedding dim: {dim}")

    X_mm = np.memmap(cache_dir / "embeddings.f32", dtype=np.float32, mode="r", shape=(len(uniq_texts), dim))

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    df_scored, yearly = rolling_6_2_1_sentiment(
        df=df,
        date_col=args.date_col,
        y_col=args.y_col,
        emb_idx=df["_emb_idx"].to_numpy().astype(np.int32),
        X_mm=X_mm,
        alpha_grid=alpha_grid,
        train_batch=args.train_batch,
    )

    td = aggregate_ticker_day(df_scored, args.date_col, args.ticker_col, args.ret_col)
    daily = daily_quintile_portfolios(td, args.date_col, args.ret_col)

    print("\nYearly OOS accuracy:")
    print(yearly.to_string(index=False) if len(yearly) else "No rolling windows formed (check years).")

    print("\nTable-6-like portfolio stats (EW):")
    mL, sdL, srL, nL = perf_stats(daily["long_ew"])
    mS, sdS, srS, nS = perf_stats(daily["short_ew"])
    m, sd, sr, n = perf_stats(daily["ls_ew"])
    print(f"EW Long:  mean={mL:.6f} sd={sdL:.6f} sharpe={srL:.3f} n_days={nL}")
    print(f"EW Short: mean={mS:.6f} sd={sdS:.6f} sharpe={srS:.3f} n_days={nS}")
    print(f"EW L-S:   mean={m:.6f}  sd={sd:.6f}  sharpe={sr:.3f} n_days={n}")

    out = args.out_prefix
    df_scored.to_csv(f"{out}_article_scored.csv", index=False)
    td.to_csv(f"{out}_ticker_day.csv", index=False)
    daily.to_csv(f"{out}_daily_portfolios.csv", index=False)
    yearly.to_csv(f"{out}_yearly_accuracy.csv", index=False)

    print(f"\nWrote: {out}_article_scored.csv")
    print(f"Wrote: {out}_ticker_day.csv")
    print(f"Wrote: {out}_daily_portfolios.csv")
    print(f"Wrote: {out}_yearly_accuracy.csv")


if __name__ == "__main__":
    main()
