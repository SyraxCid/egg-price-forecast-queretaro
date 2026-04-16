"""
hedge_signals.py
Generates actionable hedge signals for egg producers and resellers
based on the 12-month VAR forecast and current market conditions.
"""

import numpy as np
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
PRICE_RISE_STRONG  = 0.20   # >20% forecast rise  → strong accumulate / lock
PRICE_RISE_MILD    = 0.08   # >8%  forecast rise  → mild signal
PRICE_DROP_STRONG  = 0.15   # >15% forecast drop  → strong reduce
PRICE_DROP_MILD    = 0.07   # >7%  forecast drop  → mild reduce

FEED_RISE_THRESHOLD = 0.12  # >12% feed cost rise → lock feed contracts
OIL_RISE_THRESHOLD  = 0.15  # >15% oil rise       → hedge energy exposure


# ── Main signal generator ─────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    forecast_result: dict,
    scenario_shocks: dict | None = None,
) -> dict:
    """
    Parameters
    ----------
    df : full historical DataFrame
    forecast_result : output of model.forecast_12m()
    scenario_shocks : active scenario shocks (for display context)

    Returns dict with:
        - "producer_signals"  : list of signal dicts
        - "reseller_signals"  : list of signal dicts
        - "summary_table"     : pd.DataFrame for display
        - "price_trajectory"  : dict with key price stats
    """
    fc_df        = forecast_result["forecast_df"]
    last_price   = forecast_result["last_producer"]
    last_date    = forecast_result["last_date"]

    fc_3m  = fc_df["mean"].iloc[2]   # price in 3 months
    fc_6m  = fc_df["mean"].iloc[5]   # price in 6 months
    fc_12m = fc_df["mean"].iloc[-1]  # price in 12 months
    fc_peak = fc_df["mean"].max()
    fc_trough = fc_df["mean"].min()

    chg_3m  = (fc_3m  - last_price) / last_price
    chg_6m  = (fc_6m  - last_price) / last_price
    chg_12m = (fc_12m - last_price) / last_price

    # Feed cost change from last 3 months
    feed_chg = _compute_feed_change(df, months=3)
    oil_chg  = _compute_col_change(df, "oil_wti", months=3)

    # Lag estimate to first significant move (>8%)
    lag_to_move = _lag_to_threshold(fc_df["mean"], last_price, PRICE_RISE_MILD)

    producer_signals  = _producer_signals(
        chg_3m, chg_6m, chg_12m, feed_chg, oil_chg, lag_to_move, fc_peak
    )
    reseller_signals  = _reseller_signals(
        chg_3m, chg_6m, chg_12m, last_price, fc_peak, fc_trough, lag_to_move
    )

    summary_table = _build_summary_table(
        last_price, fc_3m, fc_6m, fc_12m,
        chg_3m, chg_6m, chg_12m,
        fc_peak, fc_trough, feed_chg, oil_chg, lag_to_move
    )

    return {
        "producer_signals": producer_signals,
        "reseller_signals": reseller_signals,
        "summary_table":    summary_table,
        "price_trajectory": {
            "last":     last_price,
            "fc_3m":    fc_3m,
            "fc_6m":    fc_6m,
            "fc_12m":   fc_12m,
            "chg_3m":   chg_3m,
            "chg_6m":   chg_6m,
            "chg_12m":  chg_12m,
            "peak":     fc_peak,
            "trough":   fc_trough,
            "feed_chg": feed_chg,
            "oil_chg":  oil_chg,
            "lag_months": lag_to_move,
        }
    }


# ── Producer signals ──────────────────────────────────────────────────────────

def _producer_signals(chg_3m, chg_6m, chg_12m, feed_chg, oil_chg,
                      lag_months, fc_peak):
    signals = []

    # 1. Feed cost hedging
    if feed_chg > FEED_RISE_THRESHOLD:
        signals.append({
            "signal":   "🔴 LOCK FEED COSTS NOW",
            "urgency":  "HIGH",
            "action":   f"Feed input costs up {feed_chg*100:.1f}% in 3 months. "
                        f"Negotiate 3-6 month forward contracts for corn and soy "
                        f"before additional hikes flow through.",
            "horizon":  "Immediate – 60 days",
            "category": "Feed Hedging",
        })
    elif feed_chg > FEED_RISE_THRESHOLD * 0.5:
        signals.append({
            "signal":   "🟡 MONITOR FEED CONTRACTS",
            "urgency":  "MEDIUM",
            "action":   f"Feed costs rising ({feed_chg*100:.1f}%). Begin sourcing "
                        f"forward pricing quotes. Lock in if corn crosses threshold.",
            "horizon":  "30-90 days",
            "category": "Feed Hedging",
        })
    else:
        signals.append({
            "signal":   "🟢 FEED COSTS STABLE",
            "urgency":  "LOW",
            "action":   "No immediate hedging needed. Continue monitoring monthly.",
            "horizon":  "Ongoing",
            "category": "Feed Hedging",
        })

    # 2. Energy cost hedging
    if oil_chg > OIL_RISE_THRESHOLD:
        signals.append({
            "signal":   "🔴 HEDGE ENERGY COSTS",
            "urgency":  "HIGH",
            "action":   f"Oil up {oil_chg*100:.1f}% — transportation and heating "
                        f"costs will follow. Lock in fuel supply contracts or "
                        f"pass-through clauses in buyer agreements.",
            "horizon":  "Immediate",
            "category": "Energy Hedging",
        })

    # 3. Flock size / expansion decision
    if chg_6m > PRICE_RISE_STRONG:
        signals.append({
            "signal":   "🟢 EXPAND FLOCK – PRICES RISING",
            "urgency":  "MEDIUM",
            "action":   f"Model projects +{chg_6m*100:.1f}% price in 6 months "
                        f"(peak ~${fc_peak:.1f} MXN/kg). Purchase laying hen "
                        f"chicks NOW — 18-20 week grow-out means production "
                        f"comes online in time to capture the peak.",
            "horizon":  f"Act within 4 weeks (18-20 week lag to production)",
            "category": "Flock Management",
        })
    elif chg_6m < -PRICE_DROP_STRONG:
        signals.append({
            "signal":   "🔴 DO NOT EXPAND – PRICES DECLINING",
            "urgency":  "HIGH",
            "action":   f"Model projects {chg_6m*100:.1f}% price fall in 6 months. "
                        f"Hold current flock size. Consider reducing by not "
                        f"replacing retiring hens.",
            "horizon":  "6-month window",
            "category": "Flock Management",
        })
    elif chg_6m < -PRICE_DROP_MILD:
        signals.append({
            "signal":   "🟡 HOLD FLOCK SIZE",
            "urgency":  "MEDIUM",
            "action":   f"Moderate price decline expected ({chg_6m*100:.1f}%). "
                        f"Maintain current flock. Avoid new capital expenditure.",
            "horizon":  "3-6 months",
            "category": "Flock Management",
        })

    # 4. Price lock / forward sale
    if chg_3m > PRICE_RISE_MILD and lag_months is not None:
        signals.append({
            "signal":   "🟡 CONSIDER FORWARD SALES",
            "urgency":  "MEDIUM",
            "action":   f"Price increase expected in ~{lag_months} months. "
                        f"Negotiate forward supply agreements at current or "
                        f"slightly above current prices to lock in margins.",
            "horizon":  f"{lag_months} months",
            "category": "Price Locking",
        })

    return signals


# ── Reseller signals ──────────────────────────────────────────────────────────

def _reseller_signals(chg_3m, chg_6m, chg_12m, last_price, fc_peak,
                      fc_trough, lag_months):
    signals = []

    # 1. Inventory accumulation
    if chg_3m > PRICE_RISE_STRONG:
        signals.append({
            "signal":   "🔴 ACCUMULATE INVENTORY URGENTLY",
            "urgency":  "HIGH",
            "action":   f"Strong price surge forecast (+{chg_3m*100:.1f}% in 3 months). "
                        f"Buy and store maximum possible volume at today's "
                        f"~${last_price:.1f} MXN/kg. Peak expected ~${fc_peak:.1f} MXN/kg.",
            "horizon":  f"Immediate – within {lag_months or 2} months",
            "category": "Inventory Strategy",
        })
    elif chg_3m > PRICE_RISE_MILD:
        signals.append({
            "signal":   "🟡 INCREASE INVENTORY MODERATELY",
            "urgency":  "MEDIUM",
            "action":   f"Moderate rise forecast (+{chg_3m*100:.1f}% in 3 months). "
                        f"Build 4-6 week buffer stock above usual levels.",
            "horizon":  f"Next {lag_months or 4} months",
            "category": "Inventory Strategy",
        })
    elif chg_3m < -PRICE_DROP_MILD:
        signals.append({
            "signal":   "🟡 RUN DOWN INVENTORY",
            "urgency":  "MEDIUM",
            "action":   f"Price decline expected ({chg_3m*100:.1f}% in 3 months). "
                        f"Reduce stock to minimum. Avoid overbuying at current prices.",
            "horizon":  "1-3 months",
            "category": "Inventory Strategy",
        })
    else:
        signals.append({
            "signal":   "🟢 MAINTAIN NORMAL INVENTORY",
            "urgency":  "LOW",
            "action":   "Price movement within normal range. Standard inventory levels.",
            "horizon":  "Ongoing",
            "category": "Inventory Strategy",
        })

    # 2. Supplier contract strategy
    if chg_6m > PRICE_RISE_STRONG:
        signals.append({
            "signal":   "🔴 LOCK IN SUPPLIER PRICES NOW",
            "urgency":  "HIGH",
            "action":   f"Prices projected to rise +{chg_6m*100:.1f}% over 6 months. "
                        f"Negotiate fixed-price supply contracts for 3-6 months "
                        f"at current rates. Lock in as much volume as cold-chain allows.",
            "horizon":  "Immediate negotiation",
            "category": "Supplier Contracts",
        })
    elif chg_12m < -PRICE_DROP_STRONG:
        signals.append({
            "signal":   "🟢 AVOID LONG-TERM FIXED CONTRACTS",
            "urgency":  "MEDIUM",
            "action":   f"12-month forecast shows decline ({chg_12m*100:.1f}%). "
                        f"Prefer short-term or spot purchases. Benefit from lower "
                        f"prices as they materialize.",
            "horizon":  "6-12 months",
            "category": "Supplier Contracts",
        })

    # 3. Pricing strategy to end customers
    if chg_3m > PRICE_RISE_MILD:
        signals.append({
            "signal":   "🟡 ADJUST SALE PRICE GRADUALLY",
            "urgency":  "MEDIUM",
            "action":   f"Begin gradual price increases to end customers now "
                        f"({lag_months or 2}-month head start before cost increase arrives). "
                        f"Avoids sudden margin compression.",
            "horizon":  f"Begin adjustments now",
            "category": "Pricing Strategy",
        })

    return signals


# ── Summary table ─────────────────────────────────────────────────────────────

def _build_summary_table(
    last_price, fc_3m, fc_6m, fc_12m,
    chg_3m, chg_6m, chg_12m,
    fc_peak, fc_trough, feed_chg, oil_chg, lag_months
) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Current egg price (producer)",  "Value": f"${last_price:.2f} MXN/kg"},
        {"Metric": "Forecast 3 months",             "Value": f"${fc_3m:.2f} MXN/kg  ({_pct(chg_3m)})"},
        {"Metric": "Forecast 6 months",             "Value": f"${fc_6m:.2f} MXN/kg  ({_pct(chg_6m)})"},
        {"Metric": "Forecast 12 months",            "Value": f"${fc_12m:.2f} MXN/kg ({_pct(chg_12m)})"},
        {"Metric": "Projected peak (12m window)",   "Value": f"${fc_peak:.2f} MXN/kg"},
        {"Metric": "Projected trough (12m window)", "Value": f"${fc_trough:.2f} MXN/kg"},
        {"Metric": "Feed cost trend (3m)",          "Value": _pct(feed_chg)},
        {"Metric": "Oil price trend (3m)",          "Value": _pct(oil_chg)},
        {"Metric": "Lag to first significant move", "Value": f"{lag_months} months" if lag_months else "No move >8% forecast"},
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    arrow = "▲" if v > 0 else "▼"
    return f"{arrow} {abs(v)*100:.1f}%"


def _compute_feed_change(df: pd.DataFrame, months: int = 3) -> float:
    """Average pct change in corn+soy over last N months."""
    changes = []
    for col in ["corn_mxn", "soy_mxn", "corn_usd", "soy_usd"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > months:
                chg = (s.iloc[-1] - s.iloc[-months - 1]) / s.iloc[-months - 1]
                changes.append(chg)
    return float(np.mean(changes)) if changes else 0.0


def _compute_col_change(df: pd.DataFrame, col: str, months: int = 3) -> float:
    if col not in df.columns:
        return 0.0
    s = df[col].dropna()
    if len(s) <= months:
        return 0.0
    return float((s.iloc[-1] - s.iloc[-months - 1]) / s.iloc[-months - 1])


def _lag_to_threshold(mean_series: pd.Series, base: float,
                      threshold: float) -> int | None:
    """Returns number of months until forecast price moves > threshold% from base."""
    for i, val in enumerate(mean_series):
        if abs((val - base) / base) >= threshold:
            return i + 1
    return None
