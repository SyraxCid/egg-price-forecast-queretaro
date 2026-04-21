"""
data_fetcher.py
Fetches commodity + FX data via yfinance and generates synthetic
Querétaro egg price data calibrated to real historical levels.
Falls back entirely to synthetic data if no internet connection.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

START_DATE = "2016-01-01"
TODAY = datetime.now().strftime("%Y-%m-%d")

# ── Commodity tickers ────────────────────────────────────────────────────────
TICKERS = {
    "corn_usd":  "ZC=F",   # Corn futures  (cents/bushel → /100 = USD/bushel)
    "soy_usd":   "ZS=F",   # Soybean futures (cents/bushel)
    "wheat_usd": "ZW=F",   # Wheat futures   (cents/bushel)
    "oil_wti":   "CL=F",   # WTI Crude Oil   (USD/barrel)
    "mxn_usd":   "MXN=X",  # MXN per 1 USD
}


def fetch_commodities() -> tuple[pd.DataFrame, bool]:
    """
    Returns (df_monthly, is_live).
    df_monthly columns: corn_usd, soy_usd, wheat_usd, oil_wti,
                        mxn_usd, corn_mxn, soy_mxn, wheat_mxn, oil_mxn
    """
    dfs = {}
    for name, ticker in TICKERS.items():
        try:
            raw = yf.download(ticker, start=START_DATE, end=TODAY,
                              progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("empty")
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            monthly = close.resample("MS").mean()
            monthly.name = name
            dfs[name] = monthly
        except Exception:
            pass

    if len(dfs) < 3:           # not enough live data → use synthetic
        return _synthetic_commodities(), False

    df = pd.concat(dfs.values(), axis=1)
    df.index = pd.to_datetime(df.index)

    # Convert cents/bushel → USD/bushel for grains
    for col in ["corn_usd", "soy_usd", "wheat_usd"]:
        if col in df.columns:
            df[col] = df[col] / 100.0

    # MXN-denominated feed costs
    if "mxn_usd" in df.columns:
        for grain in ["corn", "soy", "wheat"]:
            usd_col = f"{grain}_usd"
            if usd_col in df.columns:
                df[f"{grain}_mxn"] = df[usd_col] * df["mxn_usd"]
        df["oil_mxn"] = df["oil_wti"] * df["mxn_usd"]

    df = df.dropna(how="all")
    return df, True


def _synthetic_commodities() -> pd.DataFrame:
    """
    Generates plausible monthly commodity data 2016-present
    calibrated to approximate real levels and key macro events.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range(START_DATE, TODAY, freq="MS")
    n = len(idx)
    t = np.arange(n)

    def make_series(base, trend, vol, shocks: dict) -> np.ndarray:
        noise = rng.normal(0, vol, n)
        arr = base + trend * t + noise.cumsum() * 0.3
        for month_offset, magnitude in shocks.items():
            if month_offset < n:
                # decay the shock over 6 months
                for k in range(min(6, n - month_offset)):
                    arr[month_offset + k] += magnitude * np.exp(-k * 0.5)
        return np.maximum(arr, base * 0.4)

    # Month offsets (2016-01 = 0)
    covid_start   = 50   # 2020-03
    ukraine_start = 73   # 2022-02
    oil_spike_now = n - 3

    mxn = make_series(18.5, 0.025, 0.3,
                      {covid_start: 5, ukraine_start: 2, oil_spike_now: 2})

    corn_usd = make_series(3.8, 0.008, 0.15,
                           {ukraine_start: 2.5, covid_start: 0.5})

    soy_usd = make_series(9.5, 0.012, 0.2,
                          {ukraine_start: 4.0, covid_start: 1.0})

    wheat_usd = make_series(5.0, 0.010, 0.18,
                            {ukraine_start: 5.5, covid_start: 0.3})

    oil_wti = make_series(52, 0.05, 2.5,
                          {covid_start: -25, covid_start + 3: 15,
                           ukraine_start: 35, oil_spike_now: 12})
    oil_wti = np.maximum(oil_wti, 20)

    df = pd.DataFrame(index=idx)
    df["corn_usd"]   = corn_usd
    df["soy_usd"]    = soy_usd
    df["wheat_usd"]  = wheat_usd
    df["oil_wti"]    = oil_wti
    df["mxn_usd"]    = mxn
    df["corn_mxn"]   = corn_usd * mxn
    df["soy_mxn"]    = soy_usd  * mxn
    df["wheat_mxn"]  = wheat_usd * mxn
    df["oil_mxn"]    = oil_wti  * mxn
    return df


def generate_egg_prices(commodities: pd.DataFrame) -> pd.DataFrame:
    """
    Generates synthetic Querétaro egg prices (MXN/kg) correlated with
    the provided commodity data.

    Producer price ≈ 75-80% of retail price.

    Calibration targets:
    - 2016-2019 producer: 14-18 MXN/kg, retail: 18-22 MXN/kg
    - 2020 COVID dip then recovery
    - 2022 Ukraine feed spike → egg price rise with ~3-month lag
    - 2024 Q3-Q4 peak: producer ~38-42, retail ~44-50 MXN/kg
    - 2025-2026 correction (market saturation): producer ~18-22 MXN/kg
    """
    rng = np.random.default_rng(99)
    idx = commodities.index
    n = len(idx)

    # Base trend
    base_producer = 14.0
    trend = np.linspace(0, 12, n)   # long-run structural rise

    # Feed cost composite (corn + soy dominant in poultry feed)
    feed = (
        0.55 * _normalize(commodities.get("corn_mxn",
               commodities.get("corn_usd", pd.Series(1, index=idx)))) +
        0.30 * _normalize(commodities.get("soy_mxn",
               commodities.get("soy_usd",  pd.Series(1, index=idx)))) +
        0.15 * _normalize(commodities.get("oil_mxn",
               commodities.get("oil_wti",  pd.Series(1, index=idx))))
    )

    # Lag feed cost by 3 months (production cycle)
    feed_lagged = feed.shift(3).bfill()

    # COVID shock (demand drop + supply disruption)
    covid_idx = _month_offset(idx, 2020, 3)
    covid_shock = np.zeros(n)
    if covid_idx:
        for k in range(9):
            if covid_idx + k < n:
                covid_shock[covid_idx + k] = -1.5 * np.exp(-k * 0.4)

    # Ukraine war grain shock (lag 3-4 months → felt mid-2022)
    ukraine_idx = _month_offset(idx, 2022, 5)
    ukraine_shock = np.zeros(n)
    if ukraine_idx:
        for k in range(18):
            if ukraine_idx + k < n:
                ukraine_shock[ukraine_idx + k] = 2.0 * (1 - np.exp(-k * 0.3)) * np.exp(-k * 0.08)

    # Market saturation correction (many small farms flooded market ~2025)
    saturation_idx = _month_offset(idx, 2025, 1)
    saturation_shock = np.zeros(n)
    if saturation_idx:
        for k in range(18):
            if saturation_idx + k < n:
                saturation_shock[saturation_idx + k] = -9.0 * (1 - np.exp(-k * 0.35)) * np.exp(-k * 0.06)

    # Current oil/inflation pressure (building from early 2026)
    oil_now_idx = _month_offset(idx, 2026, 2)
    oil_now_shock = np.zeros(n)
    if oil_now_idx:
        for k in range(n - oil_now_idx):
            oil_now_shock[oil_now_idx + k] = 0.8 * k * np.exp(-k * 0.15)

    noise = rng.normal(0, 0.6, n)

    producer = (
        base_producer
        + trend
        + feed_lagged.values * 6.0
        + covid_shock
        + ukraine_shock
        + saturation_shock
        + oil_now_shock
        + noise
    )
    producer = np.maximum(producer, 10)

    retail = producer * rng.uniform(1.20, 1.28, n)
    retail += rng.normal(0, 0.4, n)
    retail = np.maximum(retail, producer * 1.10)

    df = pd.DataFrame({
        "egg_producer": producer,
        "egg_retail":   retail,
    }, index=idx)

    return df


# ── Real SENASICA anchor data — ALL STATES ───────────────────────────────────
# Source: datos.gob.mx / SENASICA "Unidades de producción avícola registradas"
# Two published snapshots; national totals used for calibration.

SENASICA_JUNE_2025 = {
    "Aguascalientes": 295, "Baja California": 439, "Baja California Sur": 494,
    "Campeche": 136, "Chiapas": 500, "Chihuahua": 523, "Coahuila": 946,
    "Colima": 179, "Ciudad de México": 140, "Durango": 447,
    "Estado de México": 2265, "Guanajuato": 2399, "Guerrero": 278,
    "Hidalgo": 512, "Jalisco": 1442, "Michoacán": 695, "Morelos": 415,
    "Nayarit": 157, "Nuevo León": 332, "Oaxaca": 394, "Puebla": 814,
    "Querétaro": 659, "Quintana Roo": 214, "San Luis Potosí": 362,
    "Sinaloa": 498, "Sonora": 1016, "Tabasco": 122, "Tamaulipas": 301,
    "Tlaxcala": 185, "Veracruz": 1879, "Yucatán": 330, "Zacatecas": 307,
}

SENASICA_DEC_2025 = {
    "Aguascalientes": 404, "Baja California": 622, "Baja California Sur": 570,
    "Campeche": 231, "Chiapas": 677, "Chihuahua": 599, "Coahuila": 1038,
    "Colima": 233, "Ciudad de México": 773, "Durango": 497,
    "Estado de México": 2832, "Guanajuato": 5795, "Guerrero": 325,
    "Hidalgo": 648, "Jalisco": 3409, "Michoacán": 1008, "Morelos": 469,
    "Nayarit": 189, "Nuevo León": 496, "Oaxaca": 451, "Puebla": 1252,
    "Querétaro": 958, "Quintana Roo": 302, "San Luis Potosí": 432,
    "Sinaloa": 673, "Sonora": 1810, "Tabasco": 247, "Tamaulipas": 409,
    "Tlaxcala": 254, "Veracruz": 2126, "Yucatán": 1032, "Zacatecas": 351,
}

# Derived national totals
_SENASICA_NATIONAL_JUNE = sum(SENASICA_JUNE_2025.values())   # 19,675
_SENASICA_NATIONAL_DEC  = sum(SENASICA_DEC_2025.values())    # 31,112
_SENASICA_NATIONAL_SURGE = (_SENASICA_NATIONAL_DEC - _SENASICA_NATIONAL_JUNE) / _SENASICA_NATIONAL_JUNE  # +58.1%

# Querétaro slice (for backward compatibility)
SENASICA_QRO_UNITS = {
    "2025-06": SENASICA_JUNE_2025["Querétaro"],   # 659
    "2025-12": SENASICA_DEC_2025["Querétaro"],    # 958
}

_SENASICA_BASELINE_UNITS = 420        # estimated Querétaro pre-2024 normal
_TONS_PER_UNIT = 14_000 / _SENASICA_BASELINE_UNITS   # ≈ 33 tons/unit/month


def _synthetic_production(idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Querétaro egg production proxy (tons/month), calibrated to real SENASICA national data:
      June 2025 national total : 19,675 units across 32 states
      Dec  2025 national total : 31,112 units → national surge +58.1%
      Querétaro specifically   : 659 → 958 (+45.4%) — below national average
    Uses national surge rate (stronger signal) for Dec 2025 anchor.
    Interpolates/extrapolates all other months from those anchors.
    Replace with real SIAP tonnage data via the sidebar uploader if available.
    """
    rng = np.random.default_rng(77)
    n   = len(idx)
    arr = np.empty(n)

    for i, ts in enumerate(idx):
        yr, mo = ts.year, ts.month
        key = f"{yr}-{mo:02d}"

        if key == "2025-06":
            units = 659.0                              # real SENASICA anchor

        elif key == "2025-12":
            units = 958.0                              # real SENASICA anchor

        elif yr < 2022:
            units = _SENASICA_BASELINE_UNITS           # stable pre-Ukraine baseline

        elif yr == 2022:
            units = _SENASICA_BASELINE_UNITS * 0.97   # Ukraine feed shock → marginal exits

        elif yr == 2023:
            units = _SENASICA_BASELINE_UNITS * 1.02

        elif yr == 2024 and mo <= 6:
            units = _SENASICA_BASELINE_UNITS * 1.08   # rising prices, chick purchases begin

        elif yr == 2024 and mo > 6:
            # Ramp from ~462 (Jul 2024) toward 659 (Jun 2025) over 11 months
            progress = (mo - 6) / 11.0
            units = 462 + progress * (659 - 462)

        elif yr == 2025 and mo < 6:
            # Ramp from Jan 2025 estimate (550) toward 659 (Jun 2025)
            progress = mo / 6.0
            units = 550 + progress * (659 - 550)

        elif yr == 2025 and 6 < mo < 12:
            # Interpolate Jun→Dec using NATIONAL surge (+58.1%), not just QRO (+45.4%)
            # National avg per state: 615 Jun → 972 Dec
            # Scale to QRO base: 659 × 1.581 = 1,042 implied Dec anchor
            dec_anchor = 659 * (1 + _SENASICA_NATIONAL_SURGE)
            progress = (mo - 6) / 6.0
            units = 659 + progress * (dec_anchor - 659)

        elif yr == 2025 and mo == 12:
            units = 659 * (1 + _SENASICA_NATIONAL_SURGE)   # ≈ 1,042 (national-weighted)

        else:
            # 2026+: marginal farm exits as prices stay low; gradual correction
            dec_anchor = 659 * (1 + _SENASICA_NATIONAL_SURGE)
            months_past_dec25 = (yr - 2025) * 12 + max(mo - 12, 0)
            units = max(dec_anchor * (0.88 ** (months_past_dec25 / 12)), 650)

        noise = rng.normal(0, 12)
        arr[i] = max((units + noise) * _TONS_PER_UNIT, 14_000 * 0.5)

    return arr


def build_dataset() -> tuple[pd.DataFrame, bool]:
    """
    Returns (full_monthly_df, is_live_data).
    Columns: egg_producer, egg_retail, egg_production_tons,
             corn_usd, soy_usd, wheat_usd,
             oil_wti, mxn_usd, corn_mxn, soy_mxn, wheat_mxn, oil_mxn
    """
    commodities, is_live = fetch_commodities()
    eggs = generate_egg_prices(commodities)
    prod = pd.Series(
        _synthetic_production(commodities.index),
        index=commodities.index,
        name="egg_production_tons",
    )
    df = pd.concat([eggs, prod, commodities], axis=1).dropna(how="all")
    df = df.ffill().bfill()
    return df, is_live


# ── Live price snapshot ──────────────────────────────────────────────────────

LIVE_TICKERS = {
    "ZC=F":  ("Corn",     "USD/bu",  1/100),   # cents/bushel → USD/bushel
    "ZS=F":  ("Soy",      "USD/bu",  1/100),
    "ZW=F":  ("Wheat",    "USD/bu",  1/100),
    "CL=F":  ("Oil WTI",  "USD/bbl", 1),
    "MXN=X": ("MXN/USD",  "MXN",     1),
}


def fetch_live_prices() -> dict:
    """
    Returns {ticker: {label, price, prev_close, change_pct, change_abs, unit}}
    for each ticker in LIVE_TICKERS. Falls back gracefully on failure.
    """
    from datetime import datetime
    results = {}
    for sym, (label, unit, factor) in LIVE_TICKERS.items():
        entry = {"label": label, "unit": unit, "price": None,
                 "prev_close": None, "change_pct": None, "change_abs": None,
                 "error": None}
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info
            price     = fi.last_price * factor
            prev      = fi.previous_close * factor
            chg_abs   = price - prev
            chg_pct   = chg_abs / prev * 100 if prev else 0.0
            entry.update({
                "price":      price,
                "prev_close": prev,
                "change_abs": chg_abs,
                "change_pct": chg_pct,
            })
        except Exception as exc:
            entry["error"] = str(exc)
        results[sym] = entry
    results["_fetched_at"] = datetime.now().strftime("%H:%M:%S")
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _month_offset(idx: pd.DatetimeIndex, year: int, month: int) -> int | None:
    target = pd.Timestamp(year=year, month=month, day=1)
    matches = np.where(idx >= target)[0]
    return int(matches[0]) if len(matches) else None
