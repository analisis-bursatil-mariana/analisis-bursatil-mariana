# claseapp.py — Versión ULTRA PRO para examen de Ingeniería Financiera
# Requisitos:
#   pip install streamlit yfinance pandas numpy plotly statsmodels scikit-learn

import math
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay

# Imports opcionales (ARIMA + KMeans)
try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

# ===========================
#      CONFIG & THEME
# ===========================
st.set_page_config(page_title="Análisis Bursátil — Mariana", page_icon="📈", layout="wide")

with st.sidebar:
    st.markdown("### 🎨 Tema visual")
    light_mode = st.toggle("Usar tema claro", value=False)

# Paleta mejorada (legible y moderna)
PRIMARY = "#0f172a"       # azul petróleo profundo
ACCENT = "#5d87f7"        # azul fintech moderno
BG_DARK = "#0b1221"       # azul noche
BG_LIGHT = "#f4f7fb"      # blanco suave
CARD_DARK = "#141b2d"     # gris-azulado elegante
CARD_LIGHT = "#fafafa"    # gris-claro menos brillante
TEXT_DARK = "#e8ecf8"     # blanco azulado legible
TEXT_LIGHT = "#0f172a"    # azul petróleo oscuro
TXT_MUTED = "#d8e1f0" if not light_mode else "#475569"
BORDER = "#2d3856" if not light_mode else "#e0e4ef"

PLOTLY_TEMPLATE = "plotly_white" if light_mode else "plotly_dark"
FONT_COLOR = TEXT_LIGHT if light_mode else TEXT_DARK
BG = BG_LIGHT if light_mode else BG_DARK
CARD = CARD_LIGHT if light_mode else CARD_DARK
TEXT = TEXT_LIGHT if light_mode else TEXT_DARK

bg_style = BG_LIGHT if light_mode else f"linear-gradient(135deg, {BG} 0%, #0b1628 100%)"
shadow_style = "0 6px 18px rgba(0,0,0,.05)" if light_mode else "0 6px 22px rgba(0,0,0,.35)"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg_style};
        color:{TEXT};
        font-family: 'Inter', ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}
    .metric-card {{
        background:{CARD};
        padding:1.0rem 1.25rem;
        border:1px solid {BORDER};
        border-radius:18px;
        box-shadow:{shadow_style};
    }}
    .title-accent {{
        font-size:1.9rem;
        font-weight:700;
        color:{TEXT};
        letter-spacing:-0.5px;
    }}
    .subtitle {{
        color:{TXT_MUTED};
        font-size:0.95rem;
        margin-top:.25rem;
    }}
    .footer {{
        color:{TXT_MUTED};
        font-size:0.85rem;
        text-align:center;
        margin-top:20px;
    }}
    .signal-pill {{
        display:inline-flex;
        align-items:center;
        gap:0.4rem;
        padding:0.35rem 0.9rem;
        border-radius:999px;
        font-size:0.85rem;
        font-weight:600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
#      FUNCIONES ÚTILES
# ===========================
@st.cache_data(show_spinner=False)
def load_history(ticker: str, start: str = None, end: str = None, period: str = "max") -> pd.DataFrame:
    """
    Descarga OHLCV del ticker, aplana MultiIndex y deja índice de fechas naive.
    Crea 'Adj Close' si no existe.
    """
    if start or end:
        df = yf.download(
            ticker, start=start, end=end, auto_adjust=False, progress=False, group_by="column"
        )
    else:
        df = yf.download(
            ticker, period=period, auto_adjust=False, progress=False, group_by="column"
        )

    if df is None or df.empty:
        return pd.DataFrame()

    # Aplanar columnas si vienen como MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in c]).strip() for c in df.columns]
        df.columns = [c.split("_")[0] for c in df.columns]

    # Índice de fechas sin zona horaria
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Normalizar nombres (Open, High, Low, Close, Adj Close, Volume)
    rename_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Garantizar 'Adj Close'
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    return df


def period_slice(df_or_series, days: int = None, start_date: pd.Timestamp = None):
    """Subconjunto por últimos 'days' hábiles o desde 'start_date'."""
    if df_or_series is None or (hasattr(df_or_series, "empty") and df_or_series.empty):
        return df_or_series
    obj = df_or_series.copy()
    if start_date is not None:
        return obj.loc[obj.index >= pd.to_datetime(start_date)]
    if days is None:
        return obj
    return obj.iloc[-days:].copy()


def pct_return(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def annualized_vol(daily_returns: pd.Series) -> float:
    dr = daily_returns.dropna()
    if dr.empty:
        return np.nan
    return float(dr.std() * math.sqrt(252))


def compute_metrics(
    price: pd.Series, ref_price: pd.Series = None, rf_daily: float = 0.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Rendimiento total, vol anualizada, correlación, beta,
    Sharpe (anual) y VaR(95%) histórico.
    """
    ret_total = pct_return(price)
    daily = price.dropna().pct_change().dropna()
    vol_ann = annualized_vol(daily)

    sharpe = np.nan
    if not daily.empty:
        excess = daily - rf_daily
        denom = daily.std() * math.sqrt(252)
        sharpe = float((excess.mean() * 252) / denom) if denom > 0 else np.nan

    var95 = float(np.percentile(daily, 5)) if not daily.empty else np.nan

    corr = np.nan
    beta = np.nan
    if ref_price is not None:
        ref_daily = ref_price.dropna().pct_change().dropna()
        joined = pd.concat([daily, ref_daily], axis=1).dropna()
        if joined.shape[0] > 3:
            joined.columns = ["asset", "ref"]
            corr = float(joined["asset"].corr(joined["ref"]))
            var_ref = float(joined["ref"].var())
            if var_ref > 0:
                beta = float(joined["asset"].cov(joined["ref"]) / var_ref)

    return ret_total, vol_ann, corr, beta, sharpe, var95


def ytd_start(ts_index: pd.DatetimeIndex) -> pd.Timestamp:
    if len(ts_index) == 0:
        return pd.Timestamp.today().normalize()
    yr = ts_index.max().year
    return pd.Timestamp(f"{yr}-01-01")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def get_summary_from_yf(ticker: str) -> Dict[str, str]:
    info = {}
    try:
        t = yf.Ticker(ticker)
        info = t.get_info() or {}
    except Exception:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
    long_name = info.get("longName") or info.get("shortName") or ticker.upper()
    sector = info.get("sector") or info.get("industry") or "—"
    industry = info.get("industry") or info.get("industryDisp") or "—"
    summary = info.get("longBusinessSummary") or "Descripción no disponible desde Yahoo Finance."
    return {
        "name": str(long_name),
        "sector": str(sector),
        "industry": str(industry),
        "summary": str(summary),
    }


def get_valuation_inputs(ticker: str) -> Dict[str, float]:
    """Extrae insumos básicos para un DCF muy simplificado."""
    fcf = None
    shares = None
    growth = None
    beta = None
    market_cap = None
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            info = getattr(t, "info", {}) or {}

        beta = info.get("beta")
        market_cap = info.get("marketCap")
        shares = info.get("sharesOutstanding")

        # Intentar tomar FCF de info
        fcf = info.get("freeCashflow")

        # Si no hay, intentar de cashflow anual
        if fcf is None:
            try:
                cf = t.get_cashflow(freq="yearly")
            except Exception:
                cf = getattr(t, "cashflow", None)

            if cf is not None and not cf.empty:
                for key in ["FreeCashFlow", "Free Cash Flow", "Free cash flow"]:
                    if key in cf.index:
                        # Tomamos el último año disponible
                        fcf = float(cf.loc[key].iloc[0])
                        break

        # Crecimiento
        growth = info.get("earningsGrowth") or info.get("revenueGrowth")
    except Exception:
        pass

    return {
        "fcf": fcf,
        "shares": shares,
        "growth": growth,
        "beta": beta,
        "market_cap": market_cap,
    }


def generate_technical_signal(
    price: pd.Series,
    rsi_series: pd.Series,
    macd_line: pd.Series,
    signal_line: pd.Series,
    ma_fast: pd.Series,
    ma_slow: pd.Series,
) -> Dict[str, str]:
    """Regla simple para una señal educativa de compra/venta."""
    latest = {}
    latest["price"] = float(price.dropna().iloc[-1]) if price is not None and not price.dropna().empty else np.nan
    latest["rsi"] = float(rsi_series.dropna().iloc[-1]) if rsi_series is not None and not rsi_series.dropna().empty else np.nan
    latest["macd"] = float(macd_line.dropna().iloc[-1]) if macd_line is not None and not macd_line.dropna().empty else np.nan
    latest["macd_signal"] = (
        float(signal_line.dropna().iloc[-1]) if signal_line is not None and not signal_line.dropna().empty else np.nan
    )
    latest["ma_fast"] = float(ma_fast.dropna().iloc[-1]) if ma_fast is not None and not ma_fast.dropna().empty else np.nan
    latest["ma_slow"] = float(ma_slow.dropna().iloc[-1]) if ma_slow is not None and not ma_slow.dropna().empty else np.nan

    score = 0
    bullets = []

    # RSI
    if not np.isnan(latest["rsi"]):
        if latest["rsi"] < 30:
            score += 1
            bullets.append(f"RSI en zona de sobreventa ({latest['rsi']:.1f}).")
        elif latest["rsi"] > 70:
            score -= 1
            bullets.append(f"RSI en zona de sobrecompra ({latest['rsi']:.1f}).")
        else:
            bullets.append(f"RSI neutro ({latest['rsi']:.1f}).")

    # Cruce de medias
    if not np.isnan(latest["ma_fast"]) and not np.isnan(latest["ma_slow"]):
        if latest["ma_fast"] > latest["ma_slow"]:
            score += 1
            bullets.append("La media rápida está por encima de la lenta (tendencia alcista de corto plazo).")
        elif latest["ma_fast"] < latest["ma_slow"]:
            score -= 1
            bullets.append("La media rápida está por debajo de la lenta (tendencia bajista de corto plazo).")
        else:
            bullets.append("Medias móviles muy cercanas (tendencia indefinida).")

    # MACD
    if not np.isnan(latest["macd"]) and not np.isnan(latest["macd_signal"]):
        if latest["macd"] > latest["macd_signal"]:
            score += 1
            bullets.append("MACD por arriba de la señal (momento positivo).")
        elif latest["macd"] < latest["macd_signal"]:
            score -= 1
            bullets.append("MACD por debajo de la señal (momento negativo).")

    # Traducción del score a señal
    if score >= 2:
        label = "Compra fuerte"
        color = "#16a34a"
    elif score == 1:
        label = "Compra moderada"
        color = "#22c55e"
    elif score <= -2:
        label = "Venta fuerte"
        color = "#dc2626"
    elif score == -1:
        label = "Venta moderada"
        color = "#f97316"
    else:
        label = "Neutral"
        color = "#64748b"

    return {
        "label": label,
        "color": color,
        "score": str(score),
        "bullets": bullets,
    }


def run_arima_forecast(price: pd.Series, steps: int = 60) -> pd.DataFrame:
    """Pronóstico simple ARIMA(1,1,1) sobre log-precios."""
    if ARIMA is None:
        raise ImportError("statsmodels no está instalado.")
    s = price.dropna()
    if len(s) < 80:
        raise ValueError("No hay suficientes datos históricos para ARIMA (mínimo ~80 puntos).")

    log_s = np.log(s)
    model = ARIMA(log_s, order=(1, 1, 1))
    res = model.fit()
    fc = res.get_forecast(steps=steps)

    fc_mean = np.exp(fc.predicted_mean)
    conf_int = fc.conf_int(alpha=0.05)
    conf_int = np.exp(conf_int)

    # Crear fechas futuras de días hábiles
    last_date = s.index[-1]
    future_idx = pd.date_range(last_date + BDay(1), periods=steps, freq="B")

    fc_mean.index = future_idx
    conf_int.index = future_idx

    df = pd.DataFrame(
        {
            "Histórico": s,
            "Pronóstico": fc_mean,
            "Lower": conf_int.iloc[:, 0],
            "Upper": conf_int.iloc[:, 1],
        }
    )
    return df


def compute_regimes(price: pd.Series) -> pd.DataFrame:
    """Clusteriza regímenes de mercado (baja/media/alta volatilidad) con KMeans."""
    if KMeans is None:
        raise ImportError("scikit-learn no está instalado.")

    ret = price.pct_change().dropna()
    feat = pd.DataFrame(
        {
            "ret20": ret.rolling(20).mean(),
            "vol20": ret.rolling(20).std(),
        }
    ).dropna()

    if feat.shape[0] < 80:
        raise ValueError("No hay suficientes datos para clustering (mínimo ~80 puntos).")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    feat["cluster"] = kmeans.fit_predict(feat[["ret20", "vol20"]])

    # Ordenar clusters por volatilidad
    vol_rank = feat.groupby("cluster")["vol20"].mean().sort_values()
    ordered = vol_rank.index.tolist()

    mapping = {
        ordered[0]: "Baja volatilidad",
        ordered[1]: "Media volatilidad",
        ordered[2]: "Alta volatilidad",
    }
    colors = {
        "Baja volatilidad": "#22c55e",
        "Media volatilidad": "#eab308",
        "Alta volatilidad": "#ef4444",
    }

    feat["regimen"] = feat["cluster"].map(mapping)

    # Unir con precio y retornos
    df = pd.DataFrame({"Price": price}).join(feat[["regimen", "ret20", "vol20"]], how="left")
    df["color"] = df["regimen"].map(colors)
    return df


# ===========================
#           SIDEBAR
# ===========================
with st.sidebar:
    st.markdown("### ⚙️ Parámetros")
    ticker = st.text_input("Ticker a analizar", value="AAPL").strip().upper()
    benchmark = st.text_input("Índice de referencia", value="SPY").strip().upper()
    rf_pct = st.number_input(
        "Tasa libre anual (%) para Sharpe",
        min_value=0.0,
        max_value=15.0,
        value=4.5,
        step=0.1,
    )

    st.markdown("---")
    st.markdown("### 🕒 Ventanas")
    # Base-cero
    base_zero_window = st.selectbox("Ventana comparativa base-cero", ["6M", "1Y", "3Y", "5Y"], index=1)
    window_days = {"6M": 126, "1Y": 252, "3Y": 252 * 3, "5Y": 252 * 5}[base_zero_window]

    # Velas: quick o manual
    st.markdown("### 🕯️ Velas")
    candles_mode = st.radio("Modo de periodo", ["Rápido", "Personalizado"], index=0, horizontal=True)
    if candles_mode == "Rápido":
        candles_window = st.selectbox(
            "Ventana del gráfico de velas",
            ["1M", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
            index=3,
        )
        start_cus = None
        end_cus = None
    else:
        # Rango manual
        default_end = date.today()
        default_start = date(default_end.year - 1, default_end.month, default_end.day)
        start_cus = st.date_input("Desde", value=default_start)
        end_cus = st.date_input("Hasta", value=default_end)
        candles_window = None

    st.markdown("---")
    st.markdown("### 🧰 Indicadores técnicos")
    show_ma = st.checkbox("Medias móviles en velas", value=True)
    ma_periods = st.multiselect(
        "Periodos MA",
        [10, 20, 50, 100, 200],
        default=[20, 50],
        disabled=not show_ma,
    )
    show_rsi = st.checkbox("Mostrar RSI (14)", value=True)
    show_macd = st.checkbox("Mostrar MACD (12/26/9)", value=False)

    st.markdown("---")
    st.markdown("### 📊 Comparables")
    extra_peers = st.text_input(
        "Tickers adicionales (separados por coma)",
        value="MSFT, GOOG, AMZN",
    )

    st.caption("Tip: emisoras MX usan sufijo .MX (AMXL.MX, BIMBOA.MX, etc.)")

# ===========================
#      CABECERA + RESUMEN
# ===========================
info = get_summary_from_yf(ticker)

# Panel estilo pro arriba
st.markdown(
    f"""
    <div class="metric-card" style="margin-bottom:0.75rem; display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;">
        <div>
            <div style="font-size:0.9rem; color:{TXT_MUTED}; text-transform:uppercase; letter-spacing:1px;">
                Panel de análisis bursátil profesional
            </div>
            <div class="title-accent">{info['name']} ({ticker})</div>
            <div class="subtitle">{info['sector']} · {info['industry']}</div>
        </div>
        <div style="text-align:right; font-size:0.85rem; color:{TXT_MUTED};">
            Desarrollado por <b>Mariana Hernández</b><br/>
            Proyecto de Ingeniería Financiera · {date.today().year}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-card" style="margin-bottom:1rem;">
        <div style="margin-top:.25rem; line-height:1.5; color:{TXT_MUTED}; font-size:0.92rem;">
            {info['summary']}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================
#        DATOS HISTÓRICOS
# ===========================
hist = load_history(ticker, period="max")
ref_hist = load_history(benchmark, period="max")

if hist.empty:
    st.error("No se pudieron descargar datos para ese ticker. Verifica el símbolo.")
    st.stop()

adj_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
ref_adj_col = "Adj Close" if "Adj Close" in ref_hist.columns else "Close"

# Serie base para cálculos
price = hist.get(adj_col, hist.get("Close")).copy()
if isinstance(price, pd.DataFrame):
    price = price.squeeze()

ref_price = None
if not ref_hist.empty:
    ref_price = ref_hist.get(ref_adj_col, ref_hist.get("Close"))
    if isinstance(ref_price, pd.DataFrame):
        ref_price = ref_price.squeeze()

# ===========================
#  RENDIMIENTOS Y RIESGOS
# ===========================
period_days = {
    "YTD": None,  # especial
    "3M": 63,
    "6M": 126,
    "9M": 189,
    "1Y": 252,
    "3Y": 252 * 3,
    "5Y": 252 * 5,
}
rf_daily = (rf_pct / 100.0) / 252.0

rows = []
for label, days in period_days.items():
    if label == "YTD":
        start_y = ytd_start(price.index)
        slice_p = period_slice(price, start_date=start_y)
        slice_ref = period_slice(ref_price, start_date=start_y) if ref_price is not None else None
    else:
        slice_p = period_slice(price, days=days)
        slice_ref = period_slice(ref_price, days=days) if ref_price is not None else None

    ret, vol, corr, beta, sharpe, var95 = compute_metrics(slice_p, slice_ref, rf_daily=rf_daily)
    rows.append([label, ret, vol, corr, beta, sharpe, var95])

metrics_df = pd.DataFrame(
    rows,
    columns=["Periodo", "Rendimiento", "Volatilidad (ann.)", "Correlación", "Beta", "Sharpe", "VaR 95%"],
).set_index("Periodo")

# KPIs rápidos
latest_price = price.dropna().iloc[-1] if not price.dropna().empty else np.nan
last_date = price.dropna().index[-1] if not price.dropna().empty else None
ytd_val = metrics_df.loc["YTD", "Rendimiento"]
oney_val = metrics_df.loc["1Y", "Rendimiento"]
vol_1y = metrics_df.loc["1Y", "Volatilidad (ann.)"]

# Ventana para análisis técnico (120 días)
ta_window_days = 120
ta_price = period_slice(price, days=ta_window_days)
ta_rsi = rsi(ta_price, 14) if ta_price is not None and not ta_price.empty else pd.Series(dtype=float)
ta_macd_line, ta_signal_line, _ = macd(ta_price) if ta_price is not None and not ta_price.empty else (
    pd.Series(dtype=float),
    pd.Series(dtype=float),
    pd.Series(dtype=float),
)
ta_ma_fast = ta_price.rolling(20).mean() if ta_price is not None and not ta_price.empty else pd.Series(dtype=float)
ta_ma_slow = ta_price.rolling(50).mean() if ta_price is not None and not ta_price.empty else pd.Series(dtype=float)
signal_info = generate_technical_signal(ta_price, ta_rsi, ta_macd_line, ta_signal_line, ta_ma_fast, ta_ma_slow)

# Inputs para valuación rápida
valuation_inputs = get_valuation_inputs(ticker)

# ===========================
#            TABS
# ===========================
tab_overview, tab_tech, tab_risk, tab_peers, tab_valuation, tab_forecast, tab_ml = st.tabs(
    [
        "📊 Visión general",
        "💹 Análisis técnico",
        "⚖️ Riesgo y rendimiento",
        "🤝 Comparables",
        "💰 Valuación rápida",
        "📈 Proyección ARIMA",
        "🧠 ML: regímenes de mercado",
    ]
)


def metric_card(container, title, value):
    with container:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(title, value=value if value is not None else "—")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#      TAB: VISIÓN GENERAL
# ---------------------------
with tab_overview:
    st.subheader("📌 Snapshot del activo")

    colA, colB, colC, colD = st.columns(4)
    metric_card(colA, f"Precio {ticker}", f"{latest_price:,.2f}" if not np.isnan(latest_price) else "—")
    metric_card(colB, "Rendimiento YTD", f"{ytd_val*100:,.2f}%" if pd.notna(ytd_val) else "—")
    metric_card(colC, "Rendimiento 1Y", f"{oney_val*100:,.2f}%" if pd.notna(oney_val) else "—")
    metric_card(colD, "Volatilidad 1Y (ann.)", f"{vol_1y*100:,.2f}%" if pd.notna(vol_1y) else "—")

    st.markdown("#### 🔁 Comparación base cero vs benchmark")

    # Construcción robusta del DataFrame combinado
    pieces = [hist[[adj_col]].rename(columns={adj_col: ticker})]
    if not ref_hist.empty:
        pieces.append(ref_hist[[ref_adj_col]].rename(columns={ref_adj_col: benchmark}))
    combined = pd.concat(pieces, axis=1).dropna()

    if not combined.empty:
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = ["_".join(col).strip() for col in combined.columns.values]
            combined.columns = [c.split("_")[-1] for c in combined.columns]

        window_df = combined.iloc[-window_days:].copy() if combined.shape[0] > window_days else combined.copy()
        norm = window_df / window_df.iloc[0] * 100.0

        fig_norm = px.line(norm, labels={"value": "Índice base 100", "index": "Fecha"})
        fig_norm.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=FONT_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(
            fig_norm,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
        )
    else:
        st.info("No se pudo construir la comparación base-cero con el benchmark seleccionado.")

    st.markdown("#### 📝 Resumen cuantitativo")

    trend_text = "tendencia indefinida"
    if not ta_ma_fast.dropna().empty and not ta_ma_slow.dropna().empty:
        if ta_ma_fast.iloc[-1] > ta_ma_slow.iloc[-1]:
            trend_text = "tendencia alcista de corto plazo"
        elif ta_ma_fast.iloc[-1] < ta_ma_slow.iloc[-1]:
            trend_text = "tendencia bajista de corto plazo"

    if pd.notna(oney_val) and pd.notna(vol_1y):
        st.write(
            f"En el último año, **{ticker}** acumula un rendimiento aproximado de "
            f"{(oney_val*100):.2f}% con una volatilidad anual cercana a {(vol_1y*100):.2f}% "
            f"y muestra una {trend_text}, de acuerdo con las medias móviles de 20 y 50 días."
        )
    else:
        st.write(
            f"No hay suficientes datos recientes para construir un resumen cuantitativo robusto para {ticker}."
        )

# ---------------------------
#      TAB: ANÁLISIS TÉCNICO
# ---------------------------
with tab_tech:
    st.subheader("💹 Velas e indicadores técnicos")

    # Preparar rango de velas
    if candles_mode == "Personalizado":
        if end_cus < start_cus:
            st.error("La fecha 'Hasta' debe ser mayor o igual a 'Desde'.")
            candles_df = pd.DataFrame()
            candles_title = "Rango inválido"
        else:
            candles_df = hist.loc[
                (hist.index >= pd.to_datetime(start_cus)) & (hist.index <= pd.to_datetime(end_cus))
            ].copy()
            candles_title = f"{start_cus} → {end_cus}"
    else:
        if candles_window == "Max":
            candles_df = hist.copy()
        else:
            candles_days_map = {
                "1M": 21,
                "3M": 63,
                "6M": 126,
                "1Y": 252,
                "3Y": 252 * 3,
                "5Y": 252 * 5,
            }
            candles_df = period_slice(hist, days=candles_days_map[candles_window])
        candles_title = candles_window

    st.caption(f"Periodo mostrado: {candles_title}")

    candles_df = candles_df.rename(columns={c: c.title() for c in candles_df.columns})

    if set(["Open", "High", "Low", "Close"]).issubset(candles_df.columns) and not candles_df.empty:
        # Layout de 2 o 3 filas según RSI/MACD
        extra_row = 1 if (show_rsi or show_macd) else 0
        rows = 2 + extra_row
        heights = [0.6, 0.25, 0.15] if extra_row else [0.7, 0.3]

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=heights,
        )

        # Velas
        fig.add_trace(
            go.Candlestick(
                x=candles_df.index,
                open=candles_df["Open"],
                high=candles_df["High"],
                low=candles_df["Low"],
                close=candles_df["Close"],
                name=ticker,
            ),
            row=1,
            col=1,
        )

        # MAs
        if show_ma and adj_col in candles_df.columns:
            for p in ma_periods:
                ma = candles_df[adj_col].rolling(p).mean()
                fig.add_trace(
                    go.Scatter(
                        x=candles_df.index,
                        y=ma,
                        mode="lines",
                        name=f"SMA {p}",
                        line=dict(width=1.4),
                    ),
                    row=1,
                    col=1,
                )

        # Volumen
        if "Volume" in candles_df.columns:
            fig.add_trace(
                go.Bar(x=candles_df.index, y=candles_df["Volume"], name="Volumen"),
                row=2,
                col=1,
            )

        # RSI o MACD
        target_row = 3 if extra_row else 2
        if show_rsi and adj_col in candles_df.columns:
            r = rsi(candles_df[adj_col], 14)
            fig.add_trace(
                go.Scatter(x=r.index, y=r, mode="lines", name="RSI (14)"),
                row=target_row,
                col=1,
            )
            fig.add_hrect(
                y0=30,
                y1=70,
                line_width=0,
                fillcolor="rgba(93,135,247,0.10)",
                row=target_row,
                col=1,
            )

        # MACD (solo si no mostramos RSI para no saturar)
        if show_macd and not show_rsi and adj_col in candles_df.columns:
            m_line, s_line, m_hist = macd(candles_df[adj_col])
            fig.add_trace(
                go.Scatter(x=m_line.index, y=m_line, name="MACD", mode="lines"),
                row=target_row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=s_line.index, y=s_line, name="Signal", mode="lines"),
                row=target_row,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=m_hist.index, y=m_hist, name="Hist"),
                row=target_row,
                col=1,
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=600 if extra_row else 520,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Fecha",
            yaxis_title="Precio",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=FONT_COLOR),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
        )
    else:
        st.warning("No hay columnas OHLC completas para dibujar velas en el periodo seleccionado.")

    # Señal técnica
    st.markdown("#### 📍 Señal técnica (educativa)")

    signal_html = f"""
        <div class="metric-card" style="margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.85rem;color:{TXT_MUTED};">Señal agregada</div>
                    <div style="font-size:1.1rem;font-weight:600;">{ticker} — {signal_info['label']}</div>
                </div>
                <div class="signal-pill" style="background:{signal_info['color']}20;border:1px solid {signal_info['color']}80;color:{signal_info['color']};">
                    <span>Score: {signal_info['score']}</span>
                </div>
            </div>
            <ul style="margin-top:0.75rem;padding-left:1.1rem;font-size:0.9rem;color:{TXT_MUTED};">
                {''.join(f'<li>{b}</li>' for b in signal_info['bullets'])}
            </ul>
            <div style="font-size:0.8rem;color:{TXT_MUTED};margin-top:0.25rem;">
                *Esta señal es solo con fines académicos y no constituye una recomendación de inversión.*
            </div>
        </div>
    """
    st.markdown(signal_html, unsafe_allow_html=True)

# ---------------------------
#   TAB: RIESGO Y RENDIMIENTO
# ---------------------------
with tab_risk:
    st.subheader("⚖️ Matriz de rendimiento y riesgo")

    display_df = metrics_df.copy()
    display_df["Rendimiento"] = (display_df["Rendimiento"] * 100)
    display_df["Volatilidad (ann.)"] = (display_df["Volatilidad (ann.)"] * 100)
    display_df["VaR 95%"] = (display_df["VaR 95%"] * 100)

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Rendimiento": st.column_config.NumberColumn(
                "Rendimiento",
                help="Rendimiento acumulado del periodo",
                format="%.2f%%",
            ),
            "Volatilidad (ann.)": st.column_config.NumberColumn(
                "Volatilidad (ann.)",
                help="Desv. estándar anualizada (252 días)",
                format="%.2f%%",
            ),
            "Correlación": st.column_config.NumberColumn(
                "Correlación",
                help=f"Correlación diaria vs {benchmark}",
                format="%.2f",
            ),
            "Beta": st.column_config.NumberColumn(
                "Beta",
                help=f"Sensibilidad vs {benchmark}",
                format="%.2f",
            ),
            "Sharpe": st.column_config.NumberColumn(
                "Sharpe",
                help=f"Sharpe anual (rf={rf_pct:.2f}% a/a)",
                format="%.2f",
            ),
            "VaR 95%": st.column_config.NumberColumn(
                "VaR 95%",
                help="Pérdida diaria esperada al 5% (histórico)",
                format="%.2f%%",
            ),
        },
    )

    # Botón de descarga (CSV)
    csv_bytes = metrics_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Descargar tabla (CSV)",
        data=csv_bytes,
        file_name=f"metrics_{ticker}.csv",
        mime="text/csv",
    )

    st.markdown("#### 📈 Distribución de rendimientos diarios y volatilidad")

    daily_ret = price.dropna().pct_change().dropna()
    col1, col2 = st.columns(2)

    if not daily_ret.empty:
        with col1:
            fig_hist = px.histogram(daily_ret, nbins=40, labels={"value": "Rendimiento diario"})
            fig_hist.update_layout(
                template=PLOTLY_TEMPLATE,
                height=360,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=FONT_COLOR),
            )
            st.plotly_chart(
                fig_hist,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
            )

        with col2:
            rolling_vol = daily_ret.rolling(30).std() * np.sqrt(252)
            fig_vol = px.line(
                rolling_vol,
                labels={"value": "Volatilidad anualizada 30d", "index": "Fecha"},
            )
            fig_vol.update_layout(
                template=PLOTLY_TEMPLATE,
                height=360,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=FONT_COLOR),
            )
            st.plotly_chart(
                fig_vol,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
            )

        st.markdown("#### 📉 Drawdown histórico")
        running_max = price.cummax()
        drawdown = price / running_max - 1.0
        fig_dd = px.area(drawdown, labels={"value": "Drawdown", "index": "Fecha"})
        fig_dd.update_layout(
            template=PLOTLY_TEMPLATE,
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=FONT_COLOR),
        )
        st.plotly_chart(
            fig_dd,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
        )
    else:
        st.info("No hay suficientes datos para calcular rendimientos diarios.")

# ---------------------------
#       TAB: COMPARABLES
# ---------------------------
with tab_peers:
    st.subheader("🤝 Comparables y compatibilidad")

    sector = info.get("sector", "—")
    st.markdown(f"**Sector reportado por Yahoo Finance:** {sector}")

    peer_map = {
        "Technology": ["MSFT", "GOOG", "META", "NVDA"],
        "Communication Services": ["GOOG", "META", "NFLX", "DIS"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD"],
        "Consumer Defensive": ["WMT", "COST", "PG", "KO"],
        "Financial Services": ["JPM", "BAC", "GS", "MS"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH"],
        "Industrials": ["GE", "CAT", "UNP", "BA"],
        "Energy": ["XOM", "CVX", "BP", "TTE"],
    }

    suggested = [p for p in peer_map.get(sector, []) if p != ticker]
    parsed_extra = [t.strip().upper() for t in extra_peers.split(",") if t.strip()]
    all_candidates = list(dict.fromkeys(suggested + parsed_extra))

    st.markdown(
        "**Candidatos de comparación:** "
        + (", ".join(all_candidates) if all_candidates else "Sin candidatos.")
    )

    if all_candidates:
        data = {}
        data[ticker] = price
        for t in all_candidates:
            if t == ticker:
                continue
            try:
                h = load_history(t, period="max")
                col = "Adj Close" if "Adj Close" in h.columns else "Close"
                s = h.get(col)
                if s is not None and not s.dropna().empty:
                    data[t] = s
            except Exception:
                continue

        prices_peers = pd.DataFrame(data).dropna(how="any")

        if prices_peers.shape[1] >= 2:
            ret_peers = prices_peers.pct_change().dropna()
            corr = ret_peers.corr()
            corr_main = corr[ticker].drop(index=ticker).sort_values(ascending=False)

            st.markdown("#### 🧷 Correlación de rendimientos diarios vs activo principal")
            corr_df_display = corr_main.to_frame(name="Correlación")
            st.dataframe(
                corr_df_display.style.format({"Correlación": "{:.2f}"}),
                use_container_width=True,
            )

            st.markdown("#### 🗺️ Mapa de calor de correlaciones")
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                origin="lower",
                labels=dict(color="Correlación"),
            )
            fig_corr.update_layout(
                template=PLOTLY_TEMPLATE,
                height=500,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=FONT_COLOR),
            )
            st.plotly_chart(
                fig_corr,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
            )

            st.markdown(
                "En términos prácticos, podrías considerar como **\"compatibles\"** a las acciones con "
                "correlación más alta (por encima de ~0.6) para estrategias de pares o comparación de desempeño."
            )
        else:
            st.info(
                "No se pudieron descargar suficientes series históricas para construir correlaciones con los comparables seleccionados."
            )

# ---------------------------
#       TAB: VALUACIÓN
# ---------------------------
with tab_valuation:
    st.subheader("💰 Valuación rápida tipo DCF")

    col_izq, col_der = st.columns([1.2, 1])

    with col_izq:
        st.markdown("##### Supuestos del modelo")

        if valuation_inputs["fcf"] is None or valuation_inputs["shares"] is None:
            st.warning(
                "No se encontraron datos suficientes de **flujo de efectivo libre** o **acciones en circulación** "
                "desde Yahoo Finance. Ajusta manualmente los supuestos si tu profesor lo pide."
            )

        fcf_default = valuation_inputs["fcf"] if valuation_inputs["fcf"] is not None else 1_000_000_000
        shares_default = (
            float(valuation_inputs["shares"]) if valuation_inputs["shares"] is not None else 1_000_000_000
        )
        growth_default = (
            float(valuation_inputs["growth"]) * 100
            if valuation_inputs["growth"] not in (None, np.nan)
            else 3.0
        )
        beta_default = (
            float(valuation_inputs["beta"]) if valuation_inputs["beta"] not in (None, np.nan) else 1.0
        )

        fcf_input = st.number_input(
            "FCF actual (últimos 12 meses, en moneda del ticker)",
            value=float(fcf_default),
            step=max(float(fcf_default) / 10, 1.0),
            format="%.2f",
        )
        shares_input = st.number_input(
            "Acciones en circulación",
            value=float(shares_default),
            step=max(float(shares_default) / 10, 1.0),
            format="%.0f",
        )
        discount_rate = st.number_input(
            "Tasa de descuento (WACC aprox.) %", min_value=3.0, max_value=20.0, value=10.0, step=0.5
        )
        g_long = st.number_input(
    "Crecimiento a largo plazo g %",
    min_value=0.0,
    max_value=100.0,
    value=float(growth_default if growth_default <= 100 else 5.0),
    step=0.25,
)


        st.caption(
            "Modelo tipo Gordon: Valor de la empresa = FCF₁ / (r − g), "
            "donde FCF₁ = FCF₀ × (1 + g)."
        )

    with col_der:
        st.markdown("##### Datos obtenidos de Yahoo (referencia)")
        tabla_sup = pd.DataFrame(
            {
                "Concepto": [
                    "FCF Yahoo",
                    "Acciones en circulación",
                    "Crecimiento reportado",
                    "Beta",
                    "Market cap",
                ],
                "Valor": [
                    valuation_inputs["fcf"],
                    valuation_inputs["shares"],
                    valuation_inputs["growth"],
                    valuation_inputs["beta"],
                    valuation_inputs["market_cap"],
                ],
            }
        )
        st.dataframe(tabla_sup, hide_index=True, use_container_width=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    intrinsic_price = np.nan
    upside_pct = np.nan

    if discount_rate / 100.0 > g_long / 100.0 and shares_input > 0:
        fcf1 = fcf_input * (1 + g_long / 100.0)
        equity_value = fcf1 / ((discount_rate / 100.0) - (g_long / 100.0))
        intrinsic_price = equity_value / shares_input
        if not np.isnan(latest_price):
            upside_pct = (intrinsic_price / latest_price - 1.0) * 100.0

    metric_card(
        col1,
        "Valor intrínseco estimado",
        f"{intrinsic_price:,.2f}" if not np.isnan(intrinsic_price) else "—",
    )
    metric_card(
        col2,
        "Precio de mercado",
        f"{latest_price:,.2f}" if not np.isnan(latest_price) else "—",
    )
    metric_card(
        col3,
        "Upside / Downside vs mercado",
        f"{upside_pct:,.2f}%" if not np.isnan(upside_pct) else "—",
    )

    st.markdown(
        "🔎 **Interpretación del modelo:** si el valor intrínseco es mayor al precio actual, el modelo sugiere que "
        "la acción está subvaluada (potencial de upside). Si es menor, estaría sobrevaluada según estos supuestos."
    )
    st.caption(
        "Importante: este DCF es educativo, no una recomendación. Falta considerar deuda, caja, cambios en estructura "
        "de capital y escenarios de crecimiento más realistas."
    )

# ---------------------------
#      TAB: PROYECCIÓN ARIMA
# ---------------------------
with tab_forecast:
    st.subheader("📈 Proyección de precios con ARIMA(1,1,1)")

    horizon = st.slider(
        "Horizonte de pronóstico (días hábiles)",
        min_value=20,
        max_value=120,
        value=60,
        step=5,
    )

    if ARIMA is None:
        st.error("Necesitas instalar `statsmodels` para usar este módulo: `pip install statsmodels`.")
    else:
        try:
            df_fc = run_arima_forecast(price, steps=horizon)

            fig_fc = go.Figure()
            # Histórico
            hist_part = df_fc["Histórico"].dropna()
            fc_part = df_fc["Pronóstico"].dropna()

            fig_fc.add_trace(
                go.Scatter(
                    x=hist_part.index,
                    y=hist_part.values,
                    mode="lines",
                    name="Histórico",
                )
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=fc_part.index,
                    y=fc_part.values,
                    mode="lines",
                    name="Pronóstico",
                )
            )
            # Banda de confianza
            band = df_fc.dropna(subset=["Lower", "Upper"]).loc[fc_part.index]
            fig_fc.add_trace(
                go.Scatter(
                    x=list(band.index) + list(band.index[::-1]),
                    y=list(band["Upper"]) + list(band["Lower"][::-1]),
                    fill="toself",
                    name="Intervalo 95%",
                    opacity=0.2,
                    line=dict(width=0),
                    showlegend=True,
                )
            )

            fig_fc.update_layout(
                template=PLOTLY_TEMPLATE,
                height=500,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=FONT_COLOR),
                xaxis_title="Fecha",
                yaxis_title="Precio",
            )

            st.plotly_chart(
                fig_fc,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
            )

            if not np.isnan(latest_price):
                last_fc = fc_part.iloc[-1]
                move_pct = (last_fc / latest_price - 1.0) * 100.0
                st.write(
                    f"El modelo ARIMA(1,1,1) proyecta que en ~{horizon} días hábiles el precio esperado de **{ticker}** "
                    f"podría ubicarse alrededor de **{last_fc:,.2f}**, lo que implica un cambio aproximado de "
                    f"{move_pct:,.2f}% respecto al precio actual. "
                    "Recuerda que es un modelo estadístico simple y no debe interpretarse como predicción exacta."
                )
        except Exception as e:
            st.error(f"No se pudo ajustar el modelo ARIMA: {e}")

# ---------------------------
#      TAB: ML REGÍMENES
# ---------------------------
with tab_ml:
    st.subheader("🧠 Regímenes de mercado (KMeans sobre retorno y volatilidad)")

    if KMeans is None:
        st.error("Necesitas instalar `scikit-learn` para usar este módulo: `pip install scikit-learn`.")
    else:
        try:
            regimes_df = compute_regimes(price)

            if regimes_df["regimen"].dropna().empty:
                st.info("No hay suficientes datos para estimar regímenes de mercado.")
            else:
                # Gráfico de precio coloreado por régimen
                fig_reg = go.Figure()
                for regime_name, group in regimes_df.dropna(subset=["regimen"]).groupby("regimen"):
                    fig_reg.add_trace(
                        go.Scatter(
                            x=group.index,
                            y=group["Price"],
                            mode="lines",
                            name=regime_name,
                        )
                    )

                fig_reg.update_layout(
                    template=PLOTLY_TEMPLATE,
                    height=500,
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=FONT_COLOR),
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                )
                st.plotly_chart(
                    fig_reg,
                    use_container_width=True,
                    config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["autoScale2d"]},
                )

                # Tabla resumen por régimen
                st.markdown("#### 📊 Estadísticas por régimen")

                ret = price.pct_change()
                tmp = pd.DataFrame({"ret": ret}).join(regimes_df["regimen"]).dropna()
                summary = (
                    tmp.groupby("regimen")
                    .agg(
                        Días=("ret", "count"),
                        Rendimiento_medio_diario=("ret", "mean"),
                        Volatilidad_diaria=("ret", "std"),
                    )
                    .sort_values("Volatilidad_diaria")
                )
                summary["Rendimiento_medio_diario"] *= 100
                summary["Volatilidad_diaria"] *= 100

                st.dataframe(
                    summary.style.format(
                        {
                            "Rendimiento_medio_diario": "%.3f%%",
                            "Volatilidad_diaria": "%.3f%%",
                        }
                    ),
                    use_container_width=True,
                )

                st.caption(
                    "Interpretación típica: los regímenes de **baja volatilidad** suelen asociarse con mercados estables, "
                    "mientras que los de **alta volatilidad** concentran episodios de estrés. Esto es machine learning "
                    "no supervisado aplicado a series financieras."
                )
        except Exception as e:
            st.error(f"No se pudieron estimar los regímenes de mercado: {e}")

# ===========================
#            FOOTER
# ===========================
st.markdown("---")
st.markdown(
    f"""
    <div class="footer">
    © {date.today().year} Mariana Hernández — Todos los derechos reservados. ·
    Proyecto académico (Streamlit + Yahoo Finance). Esta app es solo para fines educativos y no constituye asesoría financiera.
    </div>
    """,
    unsafe_allow_html=True,
)
