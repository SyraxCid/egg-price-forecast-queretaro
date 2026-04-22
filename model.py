"""
model.py
VAR model, Granger causality, Impulse Response Functions,
12-month forecast, and stress scenario simulation.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Variables included in the VAR system (order matters for IRF Cholesky)
# egg_producer is placed first so shocks to other variables flow into it.
VAR_COLS = [
    "egg_producer",
    "egg_retail",
    "egg_production_tons",  # supply variable — determined ~5 months prior (biological lag)
    "corn_mxn",
    "soy_mxn",
    "oil_wti",
    "mxn_usd",
]

GRANGER_TARGETS = ["egg_production_tons", "corn_mxn", "soy_mxn", "wheat_mxn", "oil_wti", "mxn_usd"]
MAX_LAGS = 12      # monthly data: up to 12-month lag search
FORECAST_STEPS = 12


# ── Stationarity ─────────────────────────────────────────────────────────────

def adf_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ADF test on each column; returns table with p-value & stationarity."""
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        try:
            stat, pval, _, _, _, _ = adfuller(series, autolag="AIC")
            rows.append({
                "Variable":     col,
                "ADF Stat":     round(stat, 3),
                "p-value":      round(pval, 4),
                "Stationary":   "Yes" if pval < 0.05 else "No",
            })
        except Exception:
            rows.append({"Variable": col, "ADF Stat": None,
                         "p-value": None, "Stationary": "Unknown"})
    return pd.DataFrame(rows)


def make_stationary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Differences non-stationary columns.
    Returns (stationary_df, differencing_map {col: n_diffs}).
    """
    out = df.copy()
    diffs = {}
    for col in df.columns:
        series = df[col].dropna()
        _, pval, *_ = adfuller(series, autolag="AIC")
        if pval >= 0.05:
            out[col] = df[col].diff()
            diffs[col] = 1
        else:
            diffs[col] = 0
    return out.dropna(), diffs


# ── Granger Causality ─────────────────────────────────────────────────────────

def run_granger(df: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """
    Tests whether each GRANGER_TARGET Granger-causes egg_producer.
    Returns DataFrame: Variable × Lag → p-value.
    """
    rows = []
    stat_df, _ = make_stationary(df[["egg_producer"] +
                                    [c for c in GRANGER_TARGETS if c in df.columns]])
    for var in GRANGER_TARGETS:
        if var not in stat_df.columns:
            continue
        pair = stat_df[["egg_producer", var]].dropna()
        row = {"Variable": var}
        try:
            results = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                pval = results[lag][0]["ssr_ftest"][1]
                row[f"Lag {lag}m"] = round(pval, 4)
        except Exception:
            for lag in range(1, max_lag + 1):
                row[f"Lag {lag}m"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variable")


# ── VAR Model ─────────────────────────────────────────────────────────────────

def fit_var(df: pd.DataFrame) -> tuple:
    """
    Fits a VAR model on available VAR_COLS.
    Returns (fitted_result, stationary_df, differencing_map, selected_lag).
    """
    available = [c for c in VAR_COLS if c in df.columns]
    sub = df[available].dropna()

    stat_df, diffs = make_stationary(sub)
    stat_df = stat_df.dropna()

    model = VAR(stat_df)
    n_obs = len(stat_df)
    n_vars = len(available)
    max_feasible = max(1, (n_obs - n_vars) // (n_vars * 2 + 1) - 1)
    safe_maxlags = min(MAX_LAGS, max_feasible, n_obs // 10)
    safe_maxlags = max(1, safe_maxlags)
    try:
        lag_results = model.select_order(maxlags=safe_maxlags)
        selected_lag = lag_results.aic or 3
    except Exception:
        selected_lag = 3
    selected_lag = max(1, min(int(selected_lag), safe_maxlags))

    result = model.fit(selected_lag)
    return result, stat_df, diffs, selected_lag, available


# ── Impulse Response ──────────────────────────────────────────────────────────

def compute_irf(var_result, periods: int = 24) -> pd.DataFrame:
    """
    Computes IRF: response of egg_producer to 1 SD shock in each variable.
    Returns DataFrame indexed by period, columns = shocked variable.
    """
    irf = var_result.irf(periods)
    names = var_result.names
    egg_idx = list(names).index("egg_producer") if "egg_producer" in names else 0

    rows = {}
    for shock_idx, shock_name in enumerate(names):
        # irf.orth_irfs shape: (periods+1, n_vars, n_vars)
        response = irf.orth_irfs[:, egg_idx, shock_idx]
        rows[shock_name] = response

    return pd.DataFrame(rows, index=range(periods + 1))


# ── 12-Month Forecast ─────────────────────────────────────────────────────────

def forecast_12m(
    df: pd.DataFrame,
    scenario_shocks: dict | None = None,
) -> dict:
    """
    Generates a 12-month ahead forecast for egg_producer price.

    scenario_shocks: optional dict of {column: pct_change} applied to the
                     last observed values before forecasting.
                     e.g. {"oil_wti": 0.30, "corn_mxn": 0.20}

    Returns dict with:
        - "forecast_df": DataFrame with mean, lower_80, upper_80, lower_95, upper_95
        - "var_result": fitted VAR result
        - "stat_df": stationary series used
        - "diffs": differencing map
        - "last_date": last observed date
        - "last_producer": last observed producer price
    """
    var_result, stat_df, diffs, lag, available = fit_var(df)

    input_data = stat_df.copy()

    # Apply scenario shocks across the full seed window (last `lag` rows)
    if scenario_shocks:
        original_levels = df[available].iloc[-1]
        seed_window = input_data.iloc[-lag:].copy()
        for col, pct in scenario_shocks.items():
            if col not in input_data.columns:
                continue
            if diffs.get(col, 0) == 1:
                delta = float(original_levels.get(col, 0)) * pct
                seed_window[col] = seed_window[col] + delta
            else:
                seed_window[col] = float(original_levels.get(col, 0)) * (1 + pct)
        input_data = pd.concat([input_data.iloc[:-lag], seed_window])

    # VAR forecast
    fc_mean, lower, upper = var_result.forecast_interval(
        input_data.values[-lag:], steps=FORECAST_STEPS, alpha=0.20
    )
    fc_mean_95, lower_95, upper_95 = var_result.forecast_interval(
        input_data.values[-lag:], steps=FORECAST_STEPS, alpha=0.05
    )

    egg_idx = list(var_result.names).index("egg_producer") \
              if "egg_producer" in var_result.names else 0

    fc   = fc_mean[:, egg_idx]
    lo80 = lower[:, egg_idx]
    hi80 = upper[:, egg_idx]
    lo95 = lower_95[:, egg_idx]
    hi95 = upper_95[:, egg_idx]

    # Undo differencing if applied
    if diffs.get("egg_producer", 0) == 1:
        last_level = df["egg_producer"].dropna().iloc[-1]
        fc   = last_level + np.cumsum(fc)
        lo80 = last_level + np.cumsum(lo80)
        hi80 = last_level + np.cumsum(hi80)
        lo95 = last_level + np.cumsum(lo95)
        hi95 = last_level + np.cumsum(hi95)

    last_date = df.index[-1]
    future_idx = pd.date_range(last_date, periods=FORECAST_STEPS + 1, freq="MS")[1:]

    forecast_df = pd.DataFrame({
        "mean":     fc,
        "lower_80": lo80,
        "upper_80": hi80,
        "lower_95": lo95,
        "upper_95": hi95,
    }, index=future_idx)

    if "egg_retail" in var_result.names:
        ret_idx = list(var_result.names).index("egg_retail")
        fc_ret   = fc_mean[:, ret_idx]
        lo80_ret = lower[:, ret_idx]
        hi80_ret = upper[:, ret_idx]
        if diffs.get("egg_retail", 0) == 1:
            last_retail_level = df["egg_retail"].dropna().iloc[-1]
            fc_ret   = last_retail_level + np.cumsum(fc_ret)
            lo80_ret = last_retail_level + np.cumsum(lo80_ret)
            hi80_ret = last_retail_level + np.cumsum(hi80_ret)
        forecast_df["retail_mean"]     = fc_ret
        forecast_df["retail_lower_80"] = lo80_ret
        forecast_df["retail_upper_80"] = hi80_ret

    return {
        "forecast_df":    forecast_df,
        "var_result":     var_result,
        "stat_df":        stat_df,
        "diffs":          diffs,
        "last_date":      last_date,
        "last_producer":  df["egg_producer"].dropna().iloc[-1],
        "last_retail":    df["egg_retail"].dropna().iloc[-1],
        "lag_order":      lag,
        "available_vars": available,
    }


# ── Stress Scenarios ──────────────────────────────────────────────────────────

SCENARIOS = {
    "Baseline (no shock)": {},
    "Oil surge +30%":      {"oil_wti": 0.30},
    "Oil surge +50%":      {"oil_wti": 0.50},
    "Grain spike +25%":    {"corn_mxn": 0.25, "soy_mxn": 0.25},
    "Grain spike +50%":    {"corn_mxn": 0.50, "soy_mxn": 0.50},
    "Combined energy+grain +30%": {"oil_wti": 0.30, "corn_mxn": 0.30, "soy_mxn": 0.30},
    "MXN depreciation +20%": {"mxn_usd": 0.20},
    "Full stress (oil+grain+FX)": {
        "oil_wti": 0.40, "corn_mxn": 0.35, "soy_mxn": 0.35, "mxn_usd": 0.20
    },
}


def run_all_scenarios(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns {scenario_name: forecast_df} for all predefined scenarios."""
    results = {}
    for name, shocks in SCENARIOS.items():
        try:
            out = forecast_12m(df, scenario_shocks=shocks)
            results[name] = out["forecast_df"]
        except Exception as e:
            logger.warning("Scenario '%s' failed: %s", name, e)
    return results
