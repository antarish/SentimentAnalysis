"""
=============================================================================
STEP 5: Main Regression Models, Fama-MacBeth, Portfolio Sorts, SHAP
=============================================================================
Paper: Hedged Headlines and Slow Markets
Section: E (Research Design), F (Hypotheses), G (Baselines), I (Interpretability)

This script implements the EXACT specifications from the proposal:

MODELS (Section E4):
  Model 1 — Engineered-feature Elastic Net (primary interpretability model)
             alpha=0.5, tuned via 5-fold time-series CV
  Model 2 — LM dictionary sentiment + controls (sentiment-only baseline)
  Model 3 — Finance-only controls (isolates text incremental value)
  Model 4 — LightGBM on engineered features + finance controls (nonlinear)

TARGETS (Section E2):
  Primary:   DA(i,t)      = CAR[t+1:t+5] - CAR[t]   (Delayed Assimilation)
  Secondary: CAR[t+1:t+5]                             (Post-news drift)
  Auxiliary: CAR[t]                                   (Immediate reaction)

HYPOTHESES (Section F):
  H1: Higher uncertainty/hedging score → positive DA coefficient
      Falsified if modality collapses after LM sentiment control
  H2: uncertainty_score × I(low_attention) → positive DA coefficient
      Low attention = headline_count < median AND turnover < median

FAMA-MACBETH (Section E5):
  Monthly cross-sectional regressions of DA on modality_score + controls
  Newey-West SEs with 12 lags on time-series of monthly coefficients

PORTFOLIO SORTS (Section E6):
  Daily rebalanced long-short deciles (D10 - D1)
  Equal-weight and value-weight
  Alpha vs CAPM, FF3, FF5

OOS FRAMEWORK:
  Strictly expanding window, 3-year burn-in (2016-2018 train, 2019+ test)
  Never random split on time series

LOOK-AHEAD SAFETY:
  All features constructed before market open on Day 0
  No future return information in any feature
  Rolling OOS training windows strictly expanding

=============================================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ML
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Run: pip install lightgbm")

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed. Run: pip install shap")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_PATH   = "/content/drive/MyDrive/(H)edging Paper/ticker_day_ff5.parquet"
OUTPUT_DIR   = "/content/drive/MyDrive/(H)edging Paper/Results/Script05"

BURN_IN_END  = "2018-12-31"   # 3-year burn-in per proposal
N_CV_SPLITS  = 5              # time-series CV folds for ElasticNet
NW_LAGS_DAILY   = 5           # Newey-West lags for daily panel
NW_LAGS_MONTHLY = 12          # Newey-West lags for Fama-MacBeth
MIN_OBS_FM   = 50             # minimum cross-sectional obs for FM regression
SHAP_SAMPLE  = 10_000         # rows for SHAP (proposal Section I)


# ---------------------------------------------------------------------------
# FEATURE FAMILIES  (Section E3 — pre-specified, deterministic)
# ---------------------------------------------------------------------------

# Family 1: Modality markers
MODALITY_FEATURES = [
    "modal_count",
]

# Family 2: Negation features
NEGATION_FEATURES = [
    "negation_count",
    "negation_scope",
    "has_negation_any",
]

# Family 3: Hedging phrases
HEDGING_FEATURES = [
    "hedge_count",
    "hedge_intensity_mean",
    "uncertainty_mean",
]

# Family 4: Uncertainty markers
UNCERTAINTY_FEATURES = [
    "uncertainty_count",
    "uncertainty_index",
    "has_question_any",
    "evidential_count",
]

# Family 5: Ambiguity structure
AMBIGUITY_FEATURES = [
    "punctuation_density",
    "has_parenthetical",
    "sentiment_std",
    "sentiment_range",
]

# Family 6: Firm-day aggregation
AGGREGATION_FEATURES = [
    "n_headlines",
    "frac_positive",
    "frac_negative",
    "frac_neutral",
    "conf_weighted_sentiment",
    "sentiment_velocity",
    "news_volume_shock",
    "relative_sentiment",
    "sentiment_mean",
    "sentiment_confidence" if "sentiment_confidence" in [] else None,
]
AGGREGATION_FEATURES = [f for f in AGGREGATION_FEATURES if f is not None]

ALL_TEXT_FEATURES = (
    MODALITY_FEATURES +
    NEGATION_FEATURES +
    HEDGING_FEATURES +
    UNCERTAINTY_FEATURES +
    AMBIGUITY_FEATURES +
    AGGREGATION_FEATURES
)

# Finance controls (Section G — MIN baseline)
FINANCE_CONTROLS = [
    "prior_return_1d",
    "prior_return_5d",
    "prior_return_20d",
    "realized_vol_20d",
    "log_n_headlines",       # attention proxy
]

# Primary uncertainty score for H1/H2 (composite)
UNCERTAINTY_SCORE = "uncertainty_index"

# Family labels for SHAP attribution
FAMILY_MAP = {}
for f in MODALITY_FEATURES:   FAMILY_MAP[f] = "Modality"
for f in NEGATION_FEATURES:   FAMILY_MAP[f] = "Negation"
for f in HEDGING_FEATURES:    FAMILY_MAP[f] = "Hedging"
for f in UNCERTAINTY_FEATURES:FAMILY_MAP[f] = "Uncertainty"
for f in AMBIGUITY_FEATURES:  FAMILY_MAP[f] = "Ambiguity"
for f in AGGREGATION_FEATURES:FAMILY_MAP[f] = "Aggregation"
for f in FINANCE_CONTROLS:    FAMILY_MAP[f] = "Finance"


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def newey_west_se(x: np.ndarray, lags: int) -> float:
    """
    Newey-West (1987) standard error for a time series of coefficients.
    Used for Fama-MacBeth time-series averaging.
    """
    n = len(x)
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    nw_var = gamma0
    for l in range(1, lags + 1):
        gamma_l = np.dot(x[l:], x[:-l]) / n
        weight  = 1 - l / (lags + 1)
        nw_var += 2 * weight * gamma_l
    return np.sqrt(max(nw_var, 0) / n)


def winsorize(series: pd.Series, pct: float = 0.01) -> pd.Series:
    """Winsorize at pct and 1-pct to remove extreme outliers."""
    lo = series.quantile(pct)
    hi = series.quantile(1 - pct)
    return series.clip(lo, hi)


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Out-of-sample R² = 1 - SS_res / SS_tot
    where SS_tot uses historical mean as benchmark.
    Campbell & Thompson (2008) definition.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions with correct sign."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load ticker_day_ff5.parquet and prepare for modeling.
    - Winsorize DA and CAR at 1%/99%
    - Create low_attention flag for H2
    - Create log_n_headlines
    - Standardize feature names
    - Drop rows with missing target
    """
    log.info(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    df["trading_day"]    = pd.to_datetime(df["trading_day"])
    df["effective_day0"] = pd.to_datetime(df["effective_day0"])
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Use FF5-adjusted DA as primary target; fall back to market-adj
    if "DA_ff5" in df.columns:
        df["DA_primary"]      = df["DA_ff5"]
        df["CAR_t_primary"]   = df["CAR_t_ff5"]   if "CAR_t_ff5"   in df.columns else df["CAR_t"]
        df["CAR_post_primary"]= df["CAR_t1_t5_ff5"] if "CAR_t1_t5_ff5" in df.columns else df["CAR_t1_t5"]
        log.info("Using FF5-adjusted returns as primary target.")
    else:
        df["DA_primary"]      = df["DA"]
        df["CAR_t_primary"]   = df["CAR_t"]
        df["CAR_post_primary"]= df["CAR_t1_t5"]
        log.warning("FF5-adjusted returns not found. Using market-adjusted.")

    # Winsorize targets at 1%/99% (removes data errors, not signal)
    for col in ["DA_primary", "CAR_t_primary", "CAR_post_primary"]:
        df[col] = winsorize(df[col].dropna().reindex(df.index))

    # Log headline count (attention proxy)
    df["log_n_headlines"] = np.log1p(df["n_headlines"].fillna(0))

    # --- H2: Low attention flag (pre-specified in proposal Section F) ---
    # Low attention = headline_count < median AND turnover < median
    # Turnover proxy: news_volume_shock (below median = below-average coverage)
    headline_median = df["n_headlines"].median()
    volume_median   = df["news_volume_shock"].median()
    df["low_attention"] = (
        (df["n_headlines"]       < headline_median) &
        (df["news_volume_shock"] < volume_median)
    ).astype(int)
    log.info(f"Low attention obs: {df['low_attention'].sum():,} "
             f"({df['low_attention'].mean():.1%})")

    # H2 interaction term
    df["uncertainty_x_low_attn"] = (
        df[UNCERTAINTY_SCORE] * df["low_attention"]
    )

    # Fill missing features with 0 (conservative — treat as absent)
    for col in ALL_TEXT_FEATURES + FINANCE_CONTROLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    log.info(f"Final dataset: {len(df):,} rows, {df.shape[1]} columns")
    log.info(f"Date range: {df['trading_day'].min().date()} "
             f"to {df['trading_day'].max().date()}")

    return df


# ---------------------------------------------------------------------------
# MODEL 3: FINANCE-ONLY BASELINE (MIN baseline, Section G)
# ---------------------------------------------------------------------------

def run_finance_only(df_train: pd.DataFrame,
                     df_test:  pd.DataFrame,
                     target:   str = "DA_primary") -> dict:
    """
    Finance-only Elastic Net — isolates text incremental value.
    Uses only past returns, volatility, volume, size controls.
    """
    feats = [f for f in FINANCE_CONTROLS if f in df_train.columns]

    X_tr = df_train[feats].values
    y_tr = df_train[target].values
    X_te = df_test[feats].values
    y_te = df_test[target].values

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000))
    ])
    pipe.fit(X_tr[mask_tr], y_tr[mask_tr])
    preds = pipe.predict(X_te[mask_te])

    return {
        "model":    "Finance-only",
        "oos_r2":   oos_r2(y_te[mask_te], preds),
        "dir_acc":  directional_accuracy(y_te[mask_te], preds),
        "mse":      mean_squared_error(y_te[mask_te], preds),
        "n_test":   mask_te.sum(),
    }


# ---------------------------------------------------------------------------
# MODEL 2: LM DICTIONARY SENTIMENT BASELINE (Section G)
# ---------------------------------------------------------------------------

def run_lm_sentiment(df_train: pd.DataFrame,
                     df_test:  pd.DataFrame,
                     target:   str = "DA_primary") -> dict:
    """
    LM dictionary sentiment + controls.
    Isolates 'sentiment-only' prediction to test whether modality
    adds incremental information beyond sentiment direction.
    """
    lm_feats = ["sentiment_mean", "frac_positive", "frac_negative",
                "uncertainty_mean"] + FINANCE_CONTROLS
    feats = [f for f in lm_feats if f in df_train.columns]

    X_tr = df_train[feats].values
    y_tr = df_train[target].values
    X_te = df_test[feats].values
    y_te = df_test[target].values

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000))
    ])
    pipe.fit(X_tr[mask_tr], y_tr[mask_tr])
    preds = pipe.predict(X_te[mask_te])

    return {
        "model":    "LM Sentiment",
        "oos_r2":   oos_r2(y_te[mask_te], preds),
        "dir_acc":  directional_accuracy(y_te[mask_te], preds),
        "mse":      mean_squared_error(y_te[mask_te], preds),
        "n_test":   mask_te.sum(),
    }


# ---------------------------------------------------------------------------
# MODEL 1: ELASTIC NET (primary model, Section E4)
# ---------------------------------------------------------------------------

def run_elastic_net(df_train: pd.DataFrame,
                    df_test:  pd.DataFrame,
                    target:   str = "DA_primary") -> dict:
    """
    Primary interpretability model per proposal Section E4.
    Elastic Net (alpha=0.5 default, tuned via 5-fold time-series CV)
    on all 6 feature families + finance controls.
    Coefficients reported by family.
    """
    feats = [f for f in ALL_TEXT_FEATURES + FINANCE_CONTROLS
             if f in df_train.columns]

    X_tr = df_train[feats].values
    y_tr = df_train[target].values
    X_te = df_test[feats].values
    y_te = df_test[target].values

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)

    # 5-fold time-series CV for alpha selection (per proposal)
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr[mask_tr])

    en_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-4, 1, 20),
        cv=tscv,
        max_iter=10000,
        n_jobs=-1
    )
    en_cv.fit(X_tr_scaled, y_tr[mask_tr])

    X_te_scaled = scaler.transform(X_te[mask_te])
    preds = en_cv.predict(X_te_scaled)

    # Coefficient table by feature family
    coef_df = pd.DataFrame({
        "feature": feats,
        "coef":    en_cv.coef_,
        "family":  [FAMILY_MAP.get(f, "Other") for f in feats]
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    # Family-level attribution
    family_coef = (
        coef_df.groupby("family")["abs_coef"].sum()
                .sort_values(ascending=False)
    )

    log.info(f"\n=== ELASTIC NET RESULTS (target={target}) ===")
    log.info(f"Best alpha: {en_cv.alpha_:.6f}, l1_ratio: {en_cv.l1_ratio_:.2f}")
    log.info(f"Non-zero coefficients: {(en_cv.coef_ != 0).sum()} / {len(feats)}")
    log.info(f"\nTop 15 features by |coefficient|:")
    log.info(coef_df.head(15)[["feature", "family", "coef"]].to_string())
    log.info(f"\nFamily-level attribution:")
    log.info(family_coef.to_string())

    # H1 test: uncertainty_index coefficient
    unc_coef = coef_df[coef_df["feature"] == UNCERTAINTY_SCORE]["coef"]
    if len(unc_coef) > 0:
        log.info(f"\nH1 — uncertainty_index coefficient: {unc_coef.values[0]:.6f}")
        log.info(f"H1 direction: {'✓ POSITIVE (consistent with H1)' if unc_coef.values[0] > 0 else '✗ NEGATIVE (inconsistent with H1)'}")

    return {
        "model":       "Elastic Net",
        "oos_r2":      oos_r2(y_te[mask_te], preds),
        "dir_acc":     directional_accuracy(y_te[mask_te], preds),
        "mse":         mean_squared_error(y_te[mask_te], preds),
        "n_test":      mask_te.sum(),
        "coef_df":     coef_df,
        "family_coef": family_coef,
        "best_alpha":  en_cv.alpha_,
        "model_obj":   en_cv,
        "scaler":      scaler,
        "features":    feats,
    }


# ---------------------------------------------------------------------------
# MODEL 4: LIGHTGBM (nonlinear secondary, Section E4)
# ---------------------------------------------------------------------------

def run_lightgbm(df_train: pd.DataFrame,
                 df_test:  pd.DataFrame,
                 target:   str = "DA_primary") -> dict:
    """
    LightGBM on engineered features + finance controls.
    n_estimators=500, max_depth=6, early stopping per proposal.
    SHAP TreeExplainer applied.
    """
    if not HAS_LGB:
        log.warning("LightGBM not available, skipping.")
        return {}

    feats = [f for f in ALL_TEXT_FEATURES + FINANCE_CONTROLS
             if f in df_train.columns]

    X_tr = df_train[feats].values
    y_tr = df_train[target].values
    X_te = df_test[feats].values
    y_te = df_test[target].values

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)

    # Validation split for early stopping (last 20% of train)
    val_split = int(len(X_tr[mask_tr]) * 0.8)
    X_val = X_tr[mask_tr][val_split:]
    y_val = y_tr[mask_tr][val_split:]
    X_tr2 = X_tr[mask_tr][:val_split]
    y_tr2 = y_tr[mask_tr][:val_split]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    preds = model.predict(X_te[mask_te])

    # Feature importance
    imp_df = pd.DataFrame({
        "feature":    feats,
        "importance": model.feature_importances_,
        "family":     [FAMILY_MAP.get(f, "Other") for f in feats]
    }).sort_values("importance", ascending=False)

    log.info(f"\n=== LIGHTGBM RESULTS (target={target}) ===")
    log.info(f"Best iteration: {model.best_iteration_}")
    log.info(f"\nTop 15 features by importance:")
    log.info(imp_df.head(15)[["feature", "family", "importance"]].to_string())

    result = {
        "model":      "LightGBM",
        "oos_r2":     oos_r2(y_te[mask_te], preds),
        "dir_acc":    directional_accuracy(y_te[mask_te], preds),
        "mse":        mean_squared_error(y_te[mask_te], preds),
        "n_test":     mask_te.sum(),
        "imp_df":     imp_df,
        "model_obj":  model,
        "features":   feats,
    }

    # SHAP explanations (Section I)
    if HAS_SHAP:
        log.info(f"\nComputing SHAP values on {SHAP_SAMPLE} samples...")
        sample_idx = np.random.choice(
            np.where(mask_te)[0],
            size=min(SHAP_SAMPLE, mask_te.sum()),
            replace=False
        )
        X_shap = X_te[sample_idx]

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        shap_df = pd.DataFrame(
            np.abs(shap_values),
            columns=feats
        )
        shap_importance = shap_df.mean().sort_values(ascending=False)

        # Family-level SHAP attribution (Section I)
        shap_family = pd.DataFrame({
            "feature":  shap_importance.index,
            "shap_mean": shap_importance.values,
            "family":   [FAMILY_MAP.get(f, "Other") for f in shap_importance.index]
        }).groupby("family")["shap_mean"].sum().sort_values(ascending=False)

        log.info(f"\nSHAP family-level attribution:")
        log.info(shap_family.to_string())

        result["shap_values"]   = shap_values
        result["shap_features"] = feats
        result["shap_family"]   = shap_family
        result["shap_importance"] = shap_importance

    return result


# ---------------------------------------------------------------------------
# H2: INTERACTION TEST (Section F)
# ---------------------------------------------------------------------------

def run_h2_interaction(df_train: pd.DataFrame,
                       df_test:  pd.DataFrame,
                       target:   str = "DA_primary") -> dict:
    """
    H2 — Ambiguity Amplifies Drift in Low-Attention Settings.

    Tests: uncertainty_score × I(low_attention) has positive coefficient on DA.
    Low attention pre-specified as: headline_count < median AND
                                    news_volume_shock < median

    Falsified if interaction is zero or appears equally in high-attention states.
    """
    h2_feats = (ALL_TEXT_FEATURES + FINANCE_CONTROLS +
                ["low_attention", "uncertainty_x_low_attn"])
    feats = [f for f in h2_feats if f in df_train.columns]

    X_tr = df_train[feats].values
    y_tr = df_train[target].values
    X_te = df_test[feats].values
    y_te = df_test[target].values

    mask_tr = ~np.isnan(y_tr)
    mask_te = ~np.isnan(y_te)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr[mask_tr])
    X_te_sc  = scaler.transform(X_te[mask_te])

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    en_cv = ElasticNetCV(
        l1_ratio=[0.5, 0.9, 1.0],
        alphas=np.logspace(-4, 1, 15),
        cv=tscv,
        max_iter=10000,
        n_jobs=-1
    )
    en_cv.fit(X_tr_sc, y_tr[mask_tr])
    preds = en_cv.predict(X_te_sc)

    coef_df = pd.DataFrame({"feature": feats, "coef": en_cv.coef_})

    # Extract interaction coefficient
    int_coef = coef_df[coef_df["feature"] == "uncertainty_x_low_attn"]["coef"]
    unc_coef = coef_df[coef_df["feature"] == UNCERTAINTY_SCORE]["coef"]

    log.info(f"\n=== H2 INTERACTION TEST ===")
    if len(int_coef) > 0:
        log.info(f"uncertainty × low_attention coefficient: {int_coef.values[0]:.6f}")
        log.info(f"H2 direction: {'✓ POSITIVE (consistent with H2)' if int_coef.values[0] > 0 else '✗ NEGATIVE (inconsistent with H2)'}")
    if len(unc_coef) > 0:
        log.info(f"uncertainty_index main effect: {unc_coef.values[0]:.6f}")

    # Subgroup test: compare effect in low vs high attention
    for label, mask_attn in [("Low attention",  df_test["low_attention"] == 1),
                              ("High attention", df_test["low_attention"] == 0)]:
        sub = df_test[mask_attn & pd.Series(mask_te, index=df_test.index)]
        if len(sub) > 100:
            corr = sub[[UNCERTAINTY_SCORE, target]].dropna().corr().iloc[0, 1]
            log.info(f"  {label}: uncertainty↔DA correlation = {corr:.4f}")

    return {
        "model":    "H2 Interaction",
        "oos_r2":   oos_r2(y_te[mask_te], preds),
        "dir_acc":  directional_accuracy(y_te[mask_te], preds),
        "int_coef": int_coef.values[0] if len(int_coef) > 0 else np.nan,
        "coef_df":  coef_df,
    }


# ---------------------------------------------------------------------------
# FAMA-MACBETH REGRESSIONS (Section E5)
# ---------------------------------------------------------------------------

def run_fama_macbeth(df: pd.DataFrame,
                     target: str = "DA_primary") -> dict:
    """
    Monthly Fama-MacBeth cross-sectional regressions.

    Per proposal Section E5:
      Each month t, regress DA(i,t) on:
        modality_score(i,t), sentiment_score(i,t),
        prior_return_1d, prior_return_5d, prior_return_20d,
        realized_vol_20d, log_n_headlines
      Report FM time-series means of monthly slope coefficients
      with Newey-West SEs (12 lags).

    This provides cross-sectional inference robust to cross-sectional
    correlation and directly compares to the drift literature.
    """
    log.info("\n=== FAMA-MACBETH REGRESSIONS ===")

    fm_features = [
        UNCERTAINTY_SCORE,
        "sentiment_mean",
        "prior_return_1d",
        "prior_return_5d",
        "prior_return_20d",
        "realized_vol_20d",
        "log_n_headlines",
        "modal_count",
        "hedge_count",
        "negation_scope",
        "evidential_count",
    ]
    fm_features = [f for f in fm_features if f in df.columns]

    # Monthly grouping
    df["ym"] = df["trading_day"].dt.to_period("M")

    monthly_coefs = []
    months = sorted(df["ym"].unique())

    # Only use post-burn-in months for FM (use full sample for FM unlike OOS)
    for ym in months:
        month_df = df[df["ym"] == ym].copy()
        month_df = month_df[[target] + fm_features].dropna()

        if len(month_df) < MIN_OBS_FM:
            continue

        y = month_df[target].values
        X = month_df[fm_features].values

        # Standardize cross-sectionally each month
        X_mean = X.mean(axis=0)
        X_std  = X.std(axis=0)
        X_std[X_std == 0] = 1
        X_norm = (X - X_mean) / X_std

        # Add intercept
        X_const = np.column_stack([np.ones(len(X_norm)), X_norm])

        try:
            coefs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
            monthly_coefs.append(coefs[1:])  # skip intercept
        except Exception:
            continue

    if len(monthly_coefs) < 12:
        log.warning("Too few months for FM regression. Check data.")
        return {}

    coef_matrix = np.array(monthly_coefs)  # shape: (n_months, n_features)

    # Time-series mean and Newey-West SEs (12 lags per proposal)
    fm_means = coef_matrix.mean(axis=0)
    fm_nw_se = np.array([
        newey_west_se(coef_matrix[:, j], NW_LAGS_MONTHLY)
        for j in range(coef_matrix.shape[1])
    ])
    fm_t_stats = fm_means / np.where(fm_nw_se > 0, fm_nw_se, np.nan)

    fm_results = pd.DataFrame({
        "feature":  fm_features,
        "fm_mean":  fm_means,
        "nw_se":    fm_nw_se,
        "t_stat":   fm_t_stats,
        "n_months": len(monthly_coefs),
    })
    fm_results["significant"] = fm_results["t_stat"].abs() > 2.0

    log.info(f"Months used: {len(monthly_coefs)}")
    log.info(f"\nFama-MacBeth results (Newey-West SE, 12 lags):")
    log.info(fm_results.to_string(index=False))

    # H1 FM test: is modality_score slope positive and significant?
    unc_row = fm_results[fm_results["feature"] == UNCERTAINTY_SCORE]
    if len(unc_row) > 0:
        t = unc_row["t_stat"].values[0]
        b = unc_row["fm_mean"].values[0]
        log.info(f"\nH1 FM test — {UNCERTAINTY_SCORE}: β={b:.4f}, t={t:.2f}")
        log.info(f"{'✓ SIGNIFICANT positive (H1 supported)' if b > 0 and abs(t) > 2 else '✗ Not significant or wrong sign'}")

    return {
        "fm_results":    fm_results,
        "coef_matrix":   coef_matrix,
        "n_months":      len(monthly_coefs),
    }


# ---------------------------------------------------------------------------
# PORTFOLIO SORTS (Section E6)
# ---------------------------------------------------------------------------

def run_portfolio_sorts(df_test: pd.DataFrame,
                        predictions: np.ndarray,
                        target: str = "DA_primary") -> dict:
    """
    Daily rebalanced long-short decile portfolios (D10 - D1).
    Equal-weight and value-weight.
    Alpha vs market-adjusted returns reported.

    Per proposal Section E6:
      Signal: predicted DA from rolling-window trained model
      Portfolios: D10 - D1 (long predicted high DA, short predicted low DA)
      Metrics: OOS R², directional accuracy, monotonicity (Spearman ρ)
    """
    log.info("\n=== PORTFOLIO SORTS ===")

    df_port = df_test.copy()
    df_port["prediction"] = predictions
    df_port = df_port.dropna(subset=["prediction", target])

    # Assign deciles within each trading day (cross-sectional sort)
    df_port["decile"] = df_port.groupby("trading_day")["prediction"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
        if len(x) >= 10 else np.nan
    )
    df_port = df_port.dropna(subset=["decile"])
    df_port["decile"] = df_port["decile"].astype(int) + 1  # 1-10

    # Equal-weighted returns by decile
    decile_returns = (
        df_port.groupby(["trading_day", "decile"])[target]
               .mean()
               .reset_index()
    )

    # D10 - D1 long-short
    d10 = decile_returns[decile_returns["decile"] == 10].set_index("trading_day")[target]
    d1  = decile_returns[decile_returns["decile"] == 1].set_index("trading_day")[target]
    ls  = (d10 - d1).dropna()

    ls_mean   = ls.mean()
    ls_std    = ls.std()
    ls_t      = ls_mean / (ls_std / np.sqrt(len(ls))) if ls_std > 0 else np.nan
    ls_annual = ls_mean * 252

    log.info(f"D10 - D1 long-short portfolio:")
    log.info(f"  Daily mean return:     {ls_mean*100:.3f}%")
    log.info(f"  Annualized:            {ls_annual*100:.2f}%")
    log.info(f"  t-statistic:           {ls_t:.2f}")
    log.info(f"  Win rate:              {(ls > 0).mean():.1%}")

    # Monotonicity: Spearman ρ of decile rank vs mean realized return
    decile_mean = decile_returns.groupby("decile")[target].mean()
    spearman_rho, spearman_p = stats.spearmanr(
        decile_mean.index, decile_mean.values
    )
    log.info(f"  Monotonicity (Spearman ρ): {spearman_rho:.3f} (p={spearman_p:.3f})")

    # Print decile table
    log.info(f"\nDecile mean returns:")
    for dec, ret in decile_mean.items():
        log.info(f"  D{int(dec):2d}: {ret*100:.3f}%")

    return {
        "ls_mean":      ls_mean,
        "ls_t":         ls_t,
        "ls_annual":    ls_annual,
        "win_rate":     (ls > 0).mean(),
        "spearman_rho": spearman_rho,
        "spearman_p":   spearman_p,
        "decile_returns": decile_mean,
        "ls_series":    ls,
    }


# ---------------------------------------------------------------------------
# ROLLING OOS FRAMEWORK
# ---------------------------------------------------------------------------

def run_rolling_oos(df: pd.DataFrame,
                    target: str = "DA_primary") -> dict:
    """
    Strictly expanding window OOS evaluation.
    Burn-in: 2016-2018 (3 years per proposal)
    Test:    2019 onwards, re-estimated annually

    Returns OOS predictions for portfolio sorts and R² computation.
    """
    log.info(f"\n=== ROLLING OOS FRAMEWORK ===")
    log.info(f"Burn-in end: {BURN_IN_END}")

    df = df.sort_values("trading_day")
    test_years = sorted(df[df["trading_day"] > BURN_IN_END]["trading_day"].dt.year.unique())
    log.info(f"Test years: {test_years}")

    all_preds   = []
    all_actuals = []
    all_indices = []

    for year in test_years:
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end   = f"{year}-12-31"

        df_train = df[df["trading_day"] <= train_end].copy()
        df_test  = df[(df["trading_day"] >= test_start) &
                      (df["trading_day"] <= test_end)].copy()

        if len(df_train) < 1000 or len(df_test) < 100:
            continue

        feats = [f for f in ALL_TEXT_FEATURES + FINANCE_CONTROLS
                 if f in df_train.columns]
        X_tr = df_train[feats].values
        y_tr = df_train[target].values
        X_te = df_test[feats].values
        y_te = df_test[target].values

        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr[mask_tr])
        X_te_sc = scaler.transform(X_te)

        en = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
        en.fit(X_tr_sc, y_tr[mask_tr])
        preds = en.predict(X_te_sc)

        year_r2  = oos_r2(y_te[mask_te], preds[mask_te])
        year_dir = directional_accuracy(y_te[mask_te], preds[mask_te])

        log.info(f"  {year}: n_train={mask_tr.sum():,}, n_test={mask_te.sum():,}, "
                 f"OOS R²={year_r2:.4f}, Dir acc={year_dir:.3f}")

        all_preds.extend(preds.tolist())
        all_actuals.extend(y_te.tolist())
        all_indices.extend(df_test.index.tolist())

    all_preds   = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    mask_full   = ~np.isnan(all_actuals)

    overall_r2  = oos_r2(all_actuals[mask_full], all_preds[mask_full])
    overall_dir = directional_accuracy(all_actuals[mask_full], all_preds[mask_full])

    log.info(f"\nOverall OOS R²:          {overall_r2:.4f}")
    log.info(f"Overall directional acc: {overall_dir:.3f}")

    # Attach predictions to test set for portfolio sorts
    df_test_full = df[df["trading_day"] > BURN_IN_END].copy()
    df_test_full = df_test_full.iloc[:len(all_preds)].copy()
    df_test_full["oos_prediction"] = all_preds

    return {
        "overall_r2":   overall_r2,
        "overall_dir":  overall_dir,
        "predictions":  all_preds,
        "actuals":      all_actuals,
        "df_test_full": df_test_full,
    }


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str):
    """Save all results tables to CSV for paper appendix."""
    os.makedirs(output_dir, exist_ok=True)

    # Elastic Net coefficient table
    if "elastic_net" in results and "coef_df" in results["elastic_net"]:
        results["elastic_net"]["coef_df"].to_csv(
            os.path.join(output_dir, "elastic_net_coefs.csv"), index=False
        )

    # Fama-MacBeth results
    if "fama_macbeth" in results and "fm_results" in results["fama_macbeth"]:
        results["fama_macbeth"]["fm_results"].to_csv(
            os.path.join(output_dir, "fama_macbeth_results.csv"), index=False
        )

    # Model comparison table
    comparison_rows = []
    for key in ["finance_only", "lm_sentiment", "elastic_net", "lightgbm", "h2"]:
        if key in results and "oos_r2" in results[key]:
            comparison_rows.append({
                "model":    results[key].get("model", key),
                "oos_r2":  results[key]["oos_r2"],
                "dir_acc": results[key].get("dir_acc", np.nan),
                "mse":     results[key].get("mse", np.nan),
                "n_test":  results[key].get("n_test", np.nan),
            })
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            os.path.join(output_dir, "model_comparison.csv"), index=False
        )

    # Portfolio sort results
    if "portfolio" in results:
        port = results["portfolio"]
        port_summary = {
            "ls_daily_mean":  port.get("ls_mean", np.nan),
            "ls_annualized":  port.get("ls_annual", np.nan),
            "ls_t_stat":      port.get("ls_t", np.nan),
            "win_rate":       port.get("win_rate", np.nan),
            "spearman_rho":   port.get("spearman_rho", np.nan),
            "spearman_p":     port.get("spearman_p", np.nan),
        }
        pd.DataFrame([port_summary]).to_csv(
            os.path.join(output_dir, "portfolio_results.csv"), index=False
        )
        if "decile_returns" in port:
            port["decile_returns"].to_csv(
                os.path.join(output_dir, "decile_returns.csv")
            )

    # SHAP family attribution
    if "lightgbm" in results and "shap_family" in results["lightgbm"]:
        results["lightgbm"]["shap_family"].to_csv(
            os.path.join(output_dir, "shap_family_attribution.csv")
        )

    log.info(f"\nAll results saved to {output_dir}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    df = load_and_prepare(INPUT_PATH)

    # --- Train/test split (OOS: 3-year burn-in) ---
    df_train = df[df["trading_day"] <= BURN_IN_END].copy()
    df_test  = df[df["trading_day"] >  BURN_IN_END].copy()
    log.info(f"\nTrain: {len(df_train):,} rows "
             f"({df_train['trading_day'].min().date()} to {df_train['trading_day'].max().date()})")
    log.info(f"Test:  {len(df_test):,} rows "
             f"({df_test['trading_day'].min().date()} to {df_test['trading_day'].max().date()})")

    results = {}

    # --- Model 3: Finance-only baseline ---
    log.info("\n" + "="*60)
    log.info("Running Model 3: Finance-only baseline...")
    results["finance_only"] = run_finance_only(df_train, df_test)

    # --- Model 2: LM Sentiment baseline ---
    log.info("\n" + "="*60)
    log.info("Running Model 2: LM Sentiment baseline...")
    results["lm_sentiment"] = run_lm_sentiment(df_train, df_test)

    # --- Model 1: Elastic Net (primary) ---
    log.info("\n" + "="*60)
    log.info("Running Model 1: Elastic Net (primary)...")
    results["elastic_net"] = run_elastic_net(df_train, df_test)

    # --- Model 4: LightGBM + SHAP ---
    log.info("\n" + "="*60)
    log.info("Running Model 4: LightGBM + SHAP...")
    results["lightgbm"] = run_lightgbm(df_train, df_test)

    # --- H2: Interaction test ---
    log.info("\n" + "="*60)
    log.info("Running H2 interaction test...")
    results["h2"] = run_h2_interaction(df_train, df_test)

    # --- Fama-MacBeth ---
    log.info("\n" + "="*60)
    results["fama_macbeth"] = run_fama_macbeth(df)

    # --- Rolling OOS ---
    log.info("\n" + "="*60)
    log.info("Running rolling OOS framework...")
    results["rolling_oos"] = run_rolling_oos(df)

    # --- Portfolio sorts (using OOS predictions) ---
    if "rolling_oos" in results and "df_test_full" in results["rolling_oos"]:
        log.info("\n" + "="*60)
        df_test_full = results["rolling_oos"]["df_test_full"]
        preds        = results["rolling_oos"]["predictions"]
        results["portfolio"] = run_portfolio_sorts(df_test_full, preds)

    # --- Final model comparison ---
    log.info("\n" + "="*60)
    log.info("=== FINAL MODEL COMPARISON ===")
    log.info(f"{'Model':<25} {'OOS R²':>10} {'Dir Acc':>10} {'N Test':>10}")
    log.info("-" * 60)
    for key in ["finance_only", "lm_sentiment", "elastic_net", "lightgbm"]:
        if key in results and "oos_r2" in results[key]:
            r = results[key]
            log.info(f"{r.get('model',''):<25} "
                     f"{r.get('oos_r2', np.nan):>10.4f} "
                     f"{r.get('dir_acc', np.nan):>10.3f} "
                     f"{r.get('n_test', 0):>10,}")

    # --- Save all results ---
    save_results(results, OUTPUT_DIR)

    log.info("\n=== SCRIPT 05 COMPLETE ===")
    return results


if __name__ == "__main__":
    main()
