"""
app.py – Querétaro Egg Price Forecast Dashboard
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from data_fetcher import build_dataset, fetch_live_prices, LIVE_TICKERS
from model import (
    adf_summary, run_granger, compute_irf,
    forecast_12m, run_all_scenarios, SCENARIOS
)
from hedge_signals import generate_signals

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Querétaro Egg Price Forecast",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND_YELLOW = "#F5C518"
BRAND_RED    = "#C0392B"
BRAND_GREEN  = "#27AE60"
BRAND_BLUE   = "#2980B9"
BRAND_DARK   = "#1A1A2E"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #F5C518; margin-bottom: 0;
    }
    .sub-header { font-size: 1rem; color: #AAAAAA; margin-top: 0; }
    .signal-card {
        border-radius: 8px; padding: 14px 16px; margin: 6px 0;
        border-left: 5px solid;
    }
    .signal-high   { background:#2D1212; border-color:#C0392B; }
    .signal-medium { background:#2D2412; border-color:#E67E22; }
    .signal-low    { background:#122D12; border-color:#27AE60; }
    .metric-box {
        background:#1E1E2E; border-radius:8px; padding:12px 16px;
        text-align:center; margin:4px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    st.markdown("### Data")
    uploaded = st.file_uploader(
        "Upload custom egg price CSV",
        type=["csv"],
        help="Columns: date (YYYY-MM-DD), egg_producer, egg_retail (MXN/kg)"
    )
    use_live = st.checkbox("Fetch live commodity data (requires internet)", value=True)

    st.markdown("---")
    st.markdown("### Scenario")
    scenario_name = st.selectbox("Stress scenario", list(SCENARIOS.keys()))

    st.markdown("### Production cycle lag")
    pullet_weeks = st.slider(
        "Weeks from chick purchase to first egg", 16, 22, 18,
        help="Affects flock expansion signal timing"
    )

    st.markdown("---")
    st.markdown("### Granger causality")
    granger_max_lag = st.slider("Max lag to test (months)", 2, 12, 6)

    st.markdown("---")
    st.markdown(
        "<small>Model: Vector Autoregression (VAR)<br>"
        "Variables: egg price, corn, soy, oil WTI, MXN/USD<br>"
        "Forecast horizon: 12 months<br>"
        "Data: SNIIM (egg) · CBOT / yfinance (commodities)</small>",
        unsafe_allow_html=True
    )


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_data(live: bool):
    return build_dataset()


df, is_live = load_data(use_live)

# Override egg prices with uploaded file if provided
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded, parse_dates=["date"], index_col="date")
        user_df.index = pd.to_datetime(user_df.index).to_period("M").to_timestamp()
        commodity_cols = [c for c in df.columns if c not in ["egg_producer", "egg_retail"]]
        df = user_df[["egg_producer", "egg_retail"]].join(
            df[commodity_cols], how="left"
        )
        st.sidebar.success(f"✅ Datos cargados: {len(df)} meses ({df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')})")
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el archivo: {e}")

st.session_state.df = df

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<p class="main-header">🥚 Querétaro Egg Price Forecast</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">VAR model · 10-year history · 12-month forecast · '
        'Hedge signals for producers & resellers</p>',
        unsafe_allow_html=True
    )
with col_badge:
    st.markdown(
        f"**Data:** {'🟢 Live' if is_live else '🟡 Synthetic'}",
        unsafe_allow_html=True
    )

st.markdown("---")

# ── Live market prices ────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_live_prices():
    return fetch_live_prices()

lp_header, lp_refresh = st.columns([8, 1])
with lp_header:
    st.markdown("### 📡 Live Market Prices")
with lp_refresh:
    if st.button("🔄 Refresh", help="Refresh live prices (auto-updates every 60 s)"):
        st.cache_data.clear()
        st.rerun()

live = load_live_prices()
fetched_at = live.get("_fetched_at", "—")

_TICKER_ORDER = ["ZC=F", "ZS=F", "ZW=F", "CL=F", "MXN=X"]
live_cols = st.columns(len(_TICKER_ORDER))

for col_widget, sym in zip(live_cols, _TICKER_ORDER):
    entry = live.get(sym, {})
    label = entry.get("label", sym)
    unit  = entry.get("unit", "")
    if entry.get("error") or entry.get("price") is None:
        col_widget.metric(f"{label} ({unit})", "N/A", help="Could not fetch live data")
        continue
    price     = entry["price"]
    chg_abs   = entry["change_abs"]
    chg_pct   = entry["change_pct"]
    # Format price based on unit
    if unit == "MXN":
        price_str = f"{price:.4f}"
    elif unit == "USD/bbl":
        price_str = f"${price:.2f}"
    else:
        price_str = f"${price:.2f}"
    delta_str = f"{chg_abs:+.2f} ({chg_pct:+.2f}%)"
    col_widget.metric(
        label=f"{label}  ·  {unit}",
        value=price_str,
        delta=delta_str,
    )

st.caption(f"Last fetched: {fetched_at} local time · Source: Yahoo Finance · Auto-refreshes every 60 s")
st.markdown("---")

# ── Key metrics bar ───────────────────────────────────────────────────────────
last_producer = df["egg_producer"].dropna().iloc[-1]
last_retail   = df["egg_retail"].dropna().iloc[-1]
last_corn     = df.get("corn_mxn", df.get("corn_usd", pd.Series())).dropna()
last_corn_val = last_corn.iloc[-1] if not last_corn.empty else None
last_oil      = df["oil_wti"].dropna().iloc[-1] if "oil_wti" in df.columns else None
last_mxn      = df["mxn_usd"].dropna().iloc[-1] if "mxn_usd" in df.columns else None

prev = df.iloc[-4:-1]   # 3 months ago for delta

def delta_pct(current, col):
    s = prev[col].dropna() if col in prev else pd.Series()
    if s.empty or s.iloc[0] == 0:
        return None
    return f"{((current - s.iloc[0]) / s.iloc[0]) * 100:+.1f}%"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Egg (producer)",  f"${last_producer:.2f} MXN/kg",
          delta_pct(last_producer, "egg_producer"))
m2.metric("Egg (retail)",    f"${last_retail:.2f} MXN/kg",
          delta_pct(last_retail, "egg_retail"))
if last_corn_val:
    col = "corn_mxn" if "corn_mxn" in df.columns else "corn_usd"
    m3.metric("Corn (MXN/bu)" if col == "corn_mxn" else "Corn (USD/bu)",
              f"${last_corn_val:.1f}", delta_pct(last_corn_val, col))
if last_oil:
    m4.metric("Oil WTI (USD/bbl)", f"${last_oil:.1f}",
              delta_pct(last_oil, "oil_wti"))
if last_mxn:
    m5.metric("MXN/USD", f"{last_mxn:.2f}",
              delta_pct(last_mxn, "mxn_usd"))

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview",
    "🔗 Lag & Causality",
    "📈 12-Month Forecast",
    "🛡️ Hedge Signals",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Market Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Historical prices — Querétaro & key commodities")

    # Egg price history
    fig_egg = go.Figure()
    fig_egg.add_trace(go.Scatter(
        x=df.index, y=df["egg_producer"],
        name="Producer price", line=dict(color=BRAND_YELLOW, width=2.5),
        hovertemplate="Producer: $%{y:.2f} MXN/kg<extra></extra>"
    ))
    fig_egg.add_trace(go.Scatter(
        x=df.index, y=df["egg_retail"],
        name="Retail price", line=dict(color=BRAND_BLUE, width=2, dash="dot"),
        hovertemplate="Retail: $%{y:.2f} MXN/kg<extra></extra>"
    ))
    # Annotate key events
    events = {
        "COVID-19\nshock": "2020-03-01",
        "Ukraine war\n(grain spike)": "2022-02-01",
        "Market\nsaturation": "2025-01-01",
        "Oil pressure\n(now)": "2026-02-01",
    }
    for label, date in events.items():
        ts = pd.Timestamp(date)
        if df.index.min() <= ts <= df.index.max():
            fig_egg.add_vline(x=ts.value, line=dict(color="gray", dash="dash", width=1),
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=10,
            )
    fig_egg.update_layout(
        title=f"Egg prices (MXN/kg) — Querétaro, {df.index.min().year}–present",
        yaxis_title="MXN / kg",
        legend=dict(orientation="h", y=-0.15),
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_egg, use_container_width=True)

    # Commodity prices
    st.markdown("#### Feed inputs & energy")
    avail_cols = {
        "corn_mxn":  ("Corn (MXN/bu)", BRAND_YELLOW),
        "soy_mxn":   ("Soy (MXN/bu)",  BRAND_GREEN),
        "wheat_mxn": ("Wheat (MXN/bu)", BRAND_BLUE),
        "oil_wti":   ("Oil WTI (USD/bbl)", BRAND_RED),
    }

    fig_comm = make_subplots(specs=[[{"secondary_y": True}]])
    for col, (label, color) in avail_cols.items():
        if col not in df.columns:
            continue
        secondary = col == "oil_wti"
        fig_comm.add_trace(
            go.Scatter(x=df.index, y=df[col], name=label,
                       line=dict(color=color, width=2),
                       hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>"),
            secondary_y=secondary
        )
    fig_comm.update_layout(
        title="Commodity prices over time",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    fig_comm.update_yaxes(title_text="MXN / bushel", secondary_y=False)
    fig_comm.update_yaxes(title_text="USD / barrel (Oil WTI)", secondary_y=True)
    st.plotly_chart(fig_comm, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Correlation matrix")
    corr_cols = [c for c in ["egg_producer", "egg_retail", "corn_mxn", "soy_mxn",
                              "wheat_mxn", "oil_wti", "mxn_usd"] if c in df.columns]
    corr = df[corr_cols].corr()

    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        title="Pearson correlation — egg prices vs. commodity drivers",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ADF stationarity
    with st.expander("Stationarity tests (ADF)"):
        adf_cols = [c for c in ["egg_producer", "corn_mxn", "soy_mxn",
                                 "oil_wti", "mxn_usd"] if c in df.columns]
        st.dataframe(adf_summary(df[adf_cols]), use_container_width=True, hide_index=True)
        st.caption("Non-stationary series are first-differenced before fitting the VAR model.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Lag & Causality
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Lag & Granger causality analysis")
    st.markdown(
        "Tests whether lagged values of each commodity **statistically cause** "
        "changes in egg producer prices (Granger causality, F-test p-values). "
        "**p < 0.05** = commodity leads egg price at that lag."
    )

    with st.spinner("Running Granger causality tests..."):
        granger_df = run_granger(df, max_lag=granger_max_lag)

    if not granger_df.empty:
        fig_gc = px.imshow(
            granger_df.astype(float),
            color_continuous_scale="RdYlGn_r",   # red = significant
            zmin=0, zmax=0.15,
            text_auto=".3f",
            title="Granger causality p-values: X → egg producer price",
            labels=dict(x="Lag (months)", y="Variable", color="p-value"),
            template="plotly_dark", height=380,
        )
        fig_gc.add_shape(
            type="line", x0=-0.5, x1=granger_max_lag - 0.5, y0=-0.5, y1=-0.5,
        )
        st.plotly_chart(fig_gc, use_container_width=True)
        st.caption(
            "Green = low p-value (significant causality). "
            "Red = high p-value (no significant causality at that lag). "
            "Read: 'corn at 3-month lag significantly predicts egg prices'."
        )
    else:
        st.warning("Granger causality test could not run — check data quality.")

    # Cross-correlation
    st.markdown("#### Cross-correlation: commodities vs. egg price (with lag)")
    lag_plot_cols = [c for c in ["corn_mxn", "soy_mxn", "oil_wti", "mxn_usd"]
                     if c in df.columns]
    egg_s = df["egg_producer"].dropna()

    fig_xcorr = go.Figure()
    colors = [BRAND_YELLOW, BRAND_GREEN, BRAND_RED, BRAND_BLUE]
    for col, color in zip(lag_plot_cols, colors):
        s = df[col].dropna()
        common = egg_s.index.intersection(s.index)
        e = egg_s[common].values
        c = s[common].values
        lags_range = range(0, granger_max_lag + 1)
        xcorr = [
            float(pd.Series(e).corr(pd.Series(np.roll(c, lag)))
                  ) for lag in lags_range
        ]
        fig_xcorr.add_trace(go.Scatter(
            x=list(lags_range), y=xcorr,
            name=col, line=dict(color=color, width=2),
            mode="lines+markers",
        ))
    fig_xcorr.update_layout(
        title="Correlation between commodity[t-lag] and egg_producer[t]",
        xaxis_title="Lag (months)",
        yaxis_title="Correlation coefficient",
        template="plotly_dark", height=380,
        legend=dict(orientation="h", y=-0.2),
    )
    fig_xcorr.add_hline(y=0.5, line_dash="dash", line_color="gray",
                        annotation_text="r=0.5 threshold")
    st.plotly_chart(fig_xcorr, use_container_width=True)

    # Impulse Response
    st.markdown("#### Impulse Response Functions (IRF)")
    st.markdown(
        "Shows how egg producer price responds over 24 months to a "
        "**1 standard-deviation shock** in each variable."
    )
    try:
        with st.spinner("Computing IRF..."):
            from model import fit_var
            var_result, _, _, _, _ = fit_var(df)
            irf_df = compute_irf(var_result, periods=24)

        fig_irf = go.Figure()
        irf_colors = [BRAND_YELLOW, BRAND_GREEN, BRAND_RED, BRAND_BLUE, "#9B59B6"]
        for col, color in zip(irf_df.columns, irf_colors):
            if col == "egg_producer":
                continue
            fig_irf.add_trace(go.Scatter(
                x=irf_df.index, y=irf_df[col],
                name=f"Shock: {col}", line=dict(color=color, width=2),
            ))
        fig_irf.add_hline(y=0, line_color="white", line_dash="dot")
        fig_irf.update_layout(
            title="IRF: egg producer price response to 1 SD commodity shock",
            xaxis_title="Months after shock",
            yaxis_title="Change in egg price (MXN/kg, differenced)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_irf, use_container_width=True)
        st.caption(
            "Positive = egg price rises after the shock. "
            "Peak response month shows the dominant transmission lag."
        )
    except Exception as e:
        st.warning(f"IRF computation failed: {e}")

    # Structural lag context
    st.markdown("#### Structural production lags")
    lag_data = {
        "Stage": [
            "Oil spike → transport/fertilizer cost rise",
            "Grain price response to oil/energy shock",
            "Feed cost reaches egg producer",
            "Pullet purchase → first egg (new flock)",
            "New flock supply hits market (price response)",
            "Total: oil shock → egg price move",
        ],
        "Estimated lag": [
            "0–4 weeks",
            "1–3 months",
            "1–2 months",
            f"{pullet_weeks} weeks (~{pullet_weeks//4} months)",
            "1–2 months after lay starts",
            "~3–5 months (cost side) / ~6–8 months (supply side)",
        ],
        "Driver": [
            "Futures markets, spot fuel prices",
            "Grain markets (CBOT), MXN/USD exchange",
            "Feed contracts, regional spot prices",
            "Biological constraint (cannot be shortened)",
            "Market saturation or tightening",
            "Combined transmission chain",
        ],
    }
    st.dataframe(pd.DataFrame(lag_data), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – 12-Month Forecast
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"12-Month forecast — scenario: {scenario_name}")

    shocks = SCENARIOS[scenario_name]
    if shocks:
        shock_text = ", ".join(
            [f"{k.replace('_', ' ')} {'+' if v > 0 else ''}{v*100:.0f}%"
             for k, v in shocks.items()]
        )
        st.info(f"Active shocks: **{shock_text}**")

    with st.spinner("Running VAR forecast..."):
        fc_result = forecast_12m(df, scenario_shocks=shocks or None)

    fc_df      = fc_result["forecast_df"]
    last_price = fc_result["last_producer"]
    last_date  = fc_result["last_date"]

    # Main forecast chart
    hist_tail = df["egg_producer"].dropna().iloc[-36:]

    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        name="Historical (producer)", line=dict(color=BRAND_YELLOW, width=2.5),
    ))

    # 95% CI band
    fig_fc.add_trace(go.Scatter(
        x=list(fc_df.index) + list(fc_df.index[::-1]),
        y=list(fc_df["upper_95"]) + list(fc_df["lower_95"][::-1]),
        fill="toself", fillcolor="rgba(41,128,185,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% confidence band", showlegend=True,
    ))

    # 80% CI band
    fig_fc.add_trace(go.Scatter(
        x=list(fc_df.index) + list(fc_df.index[::-1]),
        y=list(fc_df["upper_80"]) + list(fc_df["lower_80"][::-1]),
        fill="toself", fillcolor="rgba(41,128,185,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% confidence band", showlegend=True,
    ))

    # Forecast mean
    # Connect last historical point to forecast
    bridge_x = [hist_tail.index[-1], fc_df.index[0]]
    bridge_y = [hist_tail.iloc[-1], fc_df["mean"].iloc[0]]
    fig_fc.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        line=dict(color=BRAND_BLUE, width=2.5, dash="dot"),
        showlegend=False,
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["mean"],
        name=f"Forecast ({scenario_name})",
        line=dict(color=BRAND_BLUE, width=2.5),
        hovertemplate="Forecast: $%{y:.2f}<extra></extra>",
    ))
    if "retail_mean" in fc_df.columns:
        fig_fc.add_trace(go.Scatter(
            x=fc_df.index, y=fc_df["retail_mean"],
            name="Forecast (retail)",
            line=dict(color=BRAND_RED, width=2, dash="dot"),
            hovertemplate="Retail forecast: $%{y:.2f}<extra></extra>",
        ))

    # Vertical line at today
    fig_fc.add_vline(
        x=last_date.value, line=dict(color="white", dash="dash", width=1),
        annotation_text="Today", annotation_position="top right",
    )

    fig_fc.update_layout(
        title=f"Egg producer price forecast — {scenario_name}",
        yaxis_title="MXN / kg",
        template="plotly_dark", height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Forecast values")
        display = fc_df[["mean", "lower_80", "upper_80"]].copy()
        display.index = display.index.strftime("%b %Y")
        display.columns = ["Mean (MXN/kg)", "Lower 80%", "Upper 80%"]
        display = display.round(2)
        st.dataframe(display, use_container_width=True)

    with col_r:
        st.markdown("#### Scenario comparison")
        with st.spinner("Running all scenarios..."):
            all_scenarios = run_all_scenarios(df)

        fig_sc = go.Figure()
        sc_colors = px.colors.qualitative.Plotly
        for i, (sc_name, sc_df) in enumerate(all_scenarios.items()):
            width = 3 if sc_name == scenario_name else 1.5
            dash = "solid" if sc_name == scenario_name else "dot"
            fig_sc.add_trace(go.Scatter(
                x=sc_df.index, y=sc_df["mean"],
                name=sc_name,
                line=dict(color=sc_colors[i % len(sc_colors)],
                          width=width, dash=dash),
            ))
        fig_sc.update_layout(
            title="All scenarios — 12-month mean forecast",
            yaxis_title="MXN / kg",
            template="plotly_dark", height=380,
            legend=dict(orientation="v", x=1.02),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # VAR model diagnostics
    with st.expander("Model diagnostics"):
        st.markdown(f"**VAR lag order selected:** {fc_result['lag_order']} months (AIC)")
        st.markdown(f"**Variables in system:** {', '.join(fc_result['available_vars'])}")

        var_res = fc_result["var_result"]
        try:
            st.markdown("**Model summary**")
            st.text(str(var_res.summary())[:3000])
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – Hedge Signals
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Hedge signals & recommendations")
    st.markdown(
        f"Based on scenario: **{scenario_name}** · "
        f"Production cycle: **{pullet_weeks} weeks**"
    )

    with st.spinner("Generating hedge signals..."):
        signals_out = generate_signals(
            df=df,
            forecast_result=fc_result,
            scenario_shocks=shocks or None,
        )

    traj = signals_out["price_trajectory"]

    # Price trajectory summary
    st.markdown("#### Price trajectory summary")
    st.dataframe(
        signals_out["summary_table"],
        use_container_width=True, hide_index=True
    )

    st.markdown("---")

    # Signals side by side
    col_prod, col_res = st.columns(2)

    def render_signal(sig: dict):
        urgency = sig.get("urgency", "LOW")
        css_class = {
            "HIGH":   "signal-high",
            "MEDIUM": "signal-medium",
            "LOW":    "signal-low",
        }.get(urgency, "signal-low")
        st.markdown(f"""
        <div class="signal-card {css_class}">
            <strong>{sig['signal']}</strong><br>
            <small><b>Category:</b> {sig['category']} &nbsp;|&nbsp;
            <b>Horizon:</b> {sig['horizon']}</small><br><br>
            {sig['action']}
        </div>
        """, unsafe_allow_html=True)

    with col_prod:
        st.markdown("### For Producers / Farms")
        for sig in signals_out["producer_signals"]:
            render_signal(sig)

    with col_res:
        st.markdown("### For Resellers / Distribuidores")
        for sig in signals_out["reseller_signals"]:
            render_signal(sig)

    st.markdown("---")

    # Hedge timing chart
    st.markdown("#### Hedge timing — when to act")
    fc_mean = fc_result["forecast_df"]["mean"]
    last_p  = fc_result["last_producer"]

    chg_series = ((fc_mean - last_p) / last_p * 100)
    months = [f"M+{i+1}" for i in range(len(fc_mean))]

    colors_bar = [
        BRAND_RED if v > 15 else BRAND_YELLOW if v > 5
        else BRAND_GREEN if v < -10 else "#888888"
        for v in chg_series
    ]

    fig_timing = go.Figure(go.Bar(
        x=months, y=chg_series.values,
        marker_color=colors_bar,
        hovertemplate="Month %{x}: %{y:.1f}%<extra></extra>",
        name="% change from today",
    ))
    fig_timing.add_hline(y=0, line_color="white", line_dash="dot")
    fig_timing.add_hline(y=10, line_color=BRAND_RED, line_dash="dash",
                         annotation_text="Strong accumulate threshold (+10%)")
    fig_timing.add_hline(y=-10, line_color=BRAND_GREEN, line_dash="dash",
                         annotation_text="Run-down threshold (−10%)")
    fig_timing.update_layout(
        title="Forecast % change from current price — by month",
        yaxis_title="% change vs. today",
        template="plotly_dark", height=380,
    )
    st.plotly_chart(fig_timing, use_container_width=True)

    # Risk factors
    st.markdown("#### Key risk factors to monitor")
    risks = [
        ("Oil WTI price",      "Watch weekly. Cross above $100/bbl triggers feed & transport cost cascade.",  "🔴"),
        ("Corn futures (CBOT)","Monitor monthly. Corn = ~55% of poultry feed cost. >$7/bu signals alert.",   "🟡"),
        ("Soy futures (CBOT)", "Monitor monthly. Soy = ~30% of feed. Correlated with corn but independent.","🟡"),
        ("MXN/USD rate",       "USD-denominated feed imports become more expensive as MXN weakens.",         "🟡"),
        ("SNIIM egg price",    "Weekly price check. Divergence >10% from model = recalibrate.",              "🟢"),
        ("Flock census (SIAP)","Quarterly. Track active laying hens in Querétaro/Bajío region.",             "🟢"),
        ("New producer entry", "Market saturation = lag 6-8 months before supply impact.",                   "🟡"),
    ]
    risk_df = pd.DataFrame(risks, columns=["Factor", "Watch for", "Priority"])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # Export
    st.markdown("---")
    csv_export = fc_result["forecast_df"].copy()
    csv_export.index.name = "date"
    st.download_button(
        label="Download 12-month forecast as CSV",
        data=csv_export.to_csv().encode("utf-8"),
        file_name="egg_price_forecast_12m.csv",
        mime="text/csv",
    )
