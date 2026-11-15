"""
US SWING TRADING SCANNER
Python 3.7 compatible

Purpose:
- Scans a large US large-cap universe (S&P 100 + Nasdaq-100 style list, deduplicated)
- Downloads daily OHLCV data directly from Yahoo Finance's public chart endpoint
- Scores stocks for a 1-2 week swing trade
- Prints TOP 10 candidates with the reasons for selection
- Saves results to CSV
- Creates a combined normalized performance chart:
      first day of selected lookback = 0%
      subsequent performance = percentage change from that starting point

Recommended holding horizon:
- 1 to 10 trading days (roughly 1-2 weeks)

Requirements:
    pip install pandas numpy matplotlib requests

Designed for Python 3.7.
"""

import time
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

LOOKBACK_DAYS = 16          # chart + short-term momentum window
DOWNLOAD_DAYS = 90           # enough history for indicators
TOP_N = 10

MIN_PRICE = 10.0             # avoid very low-priced stocks
MIN_AVG_DOLLAR_VOLUME = 50_000_000   # $50M average daily traded value

# Scoring weights. Total = 100.
WEIGHT_MOMENTUM = 30
WEIGHT_TREND = 25
WEIGHT_VOLUME = 15
WEIGHT_RSI = 10
WEIGHT_BREAKOUT = 10
WEIGHT_STABILITY = 10

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.05

# Yahoo Finance sometimes uses '-' instead of '.' in tickers.
# The universe below is deliberately broad. Failed/obsolete symbols
# are skipped automatically, so the scanner keeps running.


# ============================================================
# LARGE US STOCK UNIVERSE
# S&P 100 + Nasdaq-100 style universe
# (deduplicated automatically)
# ============================================================

TICKERS = """
AAPL
ABBV
ABT
ACN
ADBE
ADI
ADP
AMGN
AMT
AMZN
AVGO
AXP
BA
BAC
BK
BKNG
BLK
BMY
BRK-B
C
CAT
CHTR
CL
CMCSA
COF
COP
COST
CRM
CSCO
CVS
CVX
DHR
DIS
DOW
DUK
EMR
EXC
F
FDX
GD
GE
GEHC
GEV
GILD
GM
GOOG
GOOGL
GS
HD
HON
IBM
INTC
INTU
ISRG
JNJ
JPM
KHC
KO
LIN
LLY
LMT
LOW
LRCX
MA
MCD
MDLZ
MDT
MET
META
MMM
MO
MOH
MRK
MRVL
MS
MSFT
MSI
MSTR
MU
NEE
NFLX
NKE
NOW
NVDA
ORCL
OXY
PANW
PEP
PFE
PG
PGR
PLTR
PM
PYPL
QCOM
RTX
SBUX
SCHW
SLB
SNOW
SO
SPG
T
TGT
TMO
TMUS
TSLA
TXN
UNH
UNP
UPS
USB
V
VLO
VRSK
VRTX
VST
VZ
WFC
WMT
XEL
XOM
ABNB
ADSK
AEP
ALGN
AMAT
ANSS
APP
ARM
ASML
AZN
CDNS
CEG
CPRT
CRWD
CSGP
CTAS
CTSH
DASH
DDOG
DLTR
EA
EBAY
ENPH
FAST
FI
FIS
FTNT
GFS
IDXX
ILMN
JD
KDP
KDP
KLAC
LULU
MAR
MCHP
MELI
MNST
MRNA
MRVL
NXPI
ODFL
ON
ORLY
PCAR
PDD
PENN
PYPL
REGN
ROST
SCCO
SIRI
SMCI
SNPS
SPLK
STX
TEAM
TER
TTD
TTWO
ULTA
VRSN
WDAY
WDC
WBD
WBA
WBD
WDAY
ZS
SNDK
COHR
VRT
ETN
PWR
HUBB
EME
URI
DE
PH
NUE
FCX
NEM
AEM
HAL
EOG
DVN
FANG
KMI
OKE
WMB
LNG
EQT
PSX
MPC
HES
SLB
DVN
COP
CARR
TT
ROK
CMI
PCAR
ITW
MMM
FAST
JCI
AME
PAYX
ROP
FTV
IR
GD
NOC
LHX
KTOS
AVAV
AXON
LDOS
SAIC
HII
TXT
TDG
HEI
RIVN
LCID
NIO
LI
XPEV
SOFI
HOOD
COIN
PLTR
APP
RBLX
U
NET
OKTA
TWLO
SHOP
SQ
DKNG
RCL
CCL
DAL
UAL
LUV
MAR
HLT
UBER
LYFT
ABNB
CVNA
CAR
"""

UNIVERSE = sorted(set(x.strip().upper() for x in TICKERS.splitlines() if x.strip()))


# ============================================================
# DATA DOWNLOAD
# ============================================================

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 Chrome/120 Safari/537.36"
})


def yahoo_symbol(symbol):
    return symbol.replace(".", "-")


def download_history(symbol, days=90):
    """
    Download daily data from Yahoo Finance chart API.
    Does not require yfinance.
    """
    end_ts = int(time.time())
    start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + yahoo_symbol(symbol)
    )

    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true"
    }

    try:
        r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        js = r.json()

        result = js.get("chart", {}).get("result")
        if not result:
            return None

        result = result[0]
        timestamps = result.get("timestamp")
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        if not timestamps:
            return None

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume")
        })

        # Use adjusted close where available.
        adj = result.get("indicators", {}).get("adjclose", [])
        if adj and "adjclose" in adj[0]:
            df["AdjClose"] = adj[0]["adjclose"]
        else:
            df["AdjClose"] = df["Close"]

        df = df.dropna(subset=["Close", "AdjClose", "Volume"])
        df = df.drop_duplicates("Date").sort_values("Date")

        if len(df) < 40:
            return None

        return df.reset_index(drop=True)

    except Exception:
        return None


# ============================================================
# INDICATORS
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def add_indicators(df):
    df = df.copy()

    c = df["AdjClose"]
    v = df["Volume"]

    df["EMA10"] = c.ewm(span=10, adjust=False).mean()
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()

    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()

    df["RSI14"] = calc_rsi(c, 14)

    df["AvgVol20"] = v.rolling(20).mean()
    df["AvgDollarVol20"] = (c * v).rolling(20).mean()

    # 20-day high/low, excluding today's candle.
    df["High20Prev"] = c.rolling(20).max().shift(1)
    df["Low20Prev"] = c.rolling(20).min().shift(1)

    # ATR percentage
    prev_close = c.shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR14"] = tr.rolling(14).mean()
    df["ATRpct"] = (df["ATR14"] / c) * 100

    # Returns
    df["Ret5"] = c.pct_change(5) * 100
    df["Ret10"] = c.pct_change(10) * 100
    df["Ret20"] = c.pct_change(20) * 100

    return df


# ============================================================
# SCORING
# ============================================================

def clamp(x, low=0.0, high=100.0):
    return max(low, min(high, x))


def score_stock(symbol, raw_df):
    df = add_indicators(raw_df)

    if len(df) < 60:
        return None

    last = df.iloc[-1]

    price = float(last["AdjClose"])
    avg_dollar_volume = float(last["AvgDollarVol20"])

    if price < MIN_PRICE:
        return None

    if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
        return None

    # --------------------------------------------------------
    # 1. MOMENTUM
    # Strong 5-10 day performance gets rewarded.
    # --------------------------------------------------------
    ret5 = float(last["Ret5"])
    ret10 = float(last["Ret10"])

    momentum_raw = (
        0.55 * (50 + ret5 * 5) +
        0.45 * (50 + ret10 * 2.5)
    )
    momentum_score = clamp(momentum_raw)

    # --------------------------------------------------------
    # 2. TREND
    # Price > EMA10 > EMA20 > EMA50 is ideal.
    # --------------------------------------------------------
    trend_score = 0

    if price > last["EMA10"]:
        trend_score += 25
    if last["EMA10"] > last["EMA20"]:
        trend_score += 25
    if last["EMA20"] > last["EMA50"]:
        trend_score += 25

    # Distance from 50 EMA, capped.
    dist50 = ((price / last["EMA50"]) - 1) * 100
    trend_score += clamp(25 + dist50 * 5, 0, 25)

    # --------------------------------------------------------
    # 3. VOLUME
    # Recent volume vs 20-day average.
    # --------------------------------------------------------
    vol_ratio = float(last["Volume"] / last["AvgVol20"]) if last["AvgVol20"] else 1

    if vol_ratio >= 2.0:
        volume_score = 100
    elif vol_ratio >= 1.5:
        volume_score = 85
    elif vol_ratio >= 1.2:
        volume_score = 70
    elif vol_ratio >= 1.0:
        volume_score = 55
    elif vol_ratio >= 0.8:
        volume_score = 40
    else:
        volume_score = 25

    # --------------------------------------------------------
    # 4. RSI
    # Prefer 55-72 for a swing.
    # Avoid extremely overbought stocks.
    # --------------------------------------------------------
    rsi = float(last["RSI14"])

    if 58 <= rsi <= 68:
        rsi_score = 100
    elif 55 <= rsi < 58 or 68 < rsi <= 72:
        rsi_score = 85
    elif 50 <= rsi < 55 or 72 < rsi <= 76:
        rsi_score = 65
    elif 45 <= rsi < 50 or 76 < rsi <= 80:
        rsi_score = 45
    else:
        rsi_score = 20

    # --------------------------------------------------------
    # 5. BREAKOUT / PROXIMITY TO 20-DAY HIGH
    # --------------------------------------------------------
    high20 = float(last["High20Prev"]) if pd.notna(last["High20Prev"]) else price
    distance_high = ((price / high20) - 1) * 100

    if distance_high >= 0:
        breakout_score = 100
    elif distance_high >= -1:
        breakout_score = 95
    elif distance_high >= -2:
        breakout_score = 85
    elif distance_high >= -4:
        breakout_score = 70
    elif distance_high >= -7:
        breakout_score = 50
    else:
        breakout_score = 25

    # --------------------------------------------------------
    # 6. STABILITY / SUITABLE VOLATILITY
    # We want movement, but not completely chaotic movement.
    # ATR 1.5%-5% is particularly useful for swing trading.
    # --------------------------------------------------------
    atrpct = float(last["ATRpct"])

    if 1.5 <= atrpct <= 5.0:
        stability_score = 100
    elif 1.0 <= atrpct < 1.5 or 5.0 < atrpct <= 7.0:
        stability_score = 80
    elif 0.7 <= atrpct < 1.0 or 7.0 < atrpct <= 9.0:
        stability_score = 60
    else:
        stability_score = 35

    total = (
        momentum_score * WEIGHT_MOMENTUM / 100 +
        trend_score * WEIGHT_TREND / 100 +
        volume_score * WEIGHT_VOLUME / 100 +
        rsi_score * WEIGHT_RSI / 100 +
        breakout_score * WEIGHT_BREAKOUT / 100 +
        stability_score * WEIGHT_STABILITY / 100
    )

    # --------------------------------------------------------
    # Human-readable reason
    # --------------------------------------------------------
    reasons = []

    if price > last["EMA20"]:
        reasons.append("above EMA20")

    if last["EMA20"] > last["EMA50"]:
        reasons.append("bullish EMA trend")

    if ret5 > 3:
        reasons.append("strong 5D momentum")

    if ret10 > 5:
        reasons.append("strong 10D momentum")

    if vol_ratio >= 1.2:
        reasons.append("volume expansion")

    if 55 <= rsi <= 72:
        reasons.append("healthy RSI")

    if distance_high >= -3:
        reasons.append("near 20D high")

    reason = ", ".join(reasons[:5])

    return {
        "Ticker": symbol,
        "Price": round(price, 2),
        "Score": round(total, 2),
        "5D %": round(ret5, 2),
        "10D %": round(ret10, 2),
        "RSI": round(rsi, 1),
        "ATR %": round(atrpct, 2),
        "Vol Ratio": round(vol_ratio, 2),
        "20D High Dist %": round(distance_high, 2),
        "Avg $ Volume M": round(avg_dollar_volume / 1_000_000, 1),
        "Reason": reason,
        "_data": df
    }



# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="US Swing Trading Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("🇺🇸 US Swing Trading Scanner")
st.markdown(
    """
    [📊 Open NSE Stock Chart Application](https://rpandaw-stock-chart.streamlit.app/)
    """,
    unsafe_allow_html=False
)
st.caption("Ranks liquid US stocks for a potential 1–10 trading-day swing. "
           "This is a screening tool, not an automatic buy signal.")

with st.sidebar:
    st.header("Scanner settings")
    top_n = st.slider("Number of candidates", 5, 25, TOP_N)
    lookback_days = st.slider("Chart lookback (trading days)", 5, 30, LOOKBACK_DAYS)
    min_price = st.number_input("Minimum stock price ($)", min_value=1.0,
                                max_value=1000.0, value=float(MIN_PRICE), step=1.0)
    min_avg_dollar_volume = st.number_input(
        "Minimum average daily dollar volume ($)",
        min_value=1000000.0,
        max_value=1000000000.0,
        value=float(MIN_AVG_DOLLAR_VOLUME),
        step=5000000.0,
        format="%.0f"
    )
    run_button = st.button("▶ Run US stock scanner", type="primary", use_container_width=True)

st.info(
    "The scanner downloads daily prices from Yahoo Finance. "
    "A full scan can take a few minutes because it checks a large ticker universe."
)

if run_button:
    # Apply the user's selected filters to the existing scoring function.
    MIN_PRICE = float(min_price)
    MIN_AVG_DOLLAR_VOLUME = float(min_avg_dollar_volume)

    progress = st.progress(0)
    status = st.empty()
    results = []
    failed = []
    filtered = []
    total = len(UNIVERSE)

    for i, symbol in enumerate(UNIVERSE, 1):
        status.write("Checking {}/{}: **{}**".format(i, total, symbol))
        df = download_history(symbol, DOWNLOAD_DAYS)

        if df is None:
            failed.append(symbol)
        else:
            result = score_stock(symbol, df)
            if result is None:
                filtered.append(symbol)
            else:
                results.append(result)

        progress.progress(float(i) / float(total))
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    status.empty()
    progress.empty()

    if not results:
        st.error("No stocks passed the filters. Try lowering the minimum price or dollar-volume filter.")
    else:
        result_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "_data"}
            for r in results
        ])
        result_df = result_df.sort_values("Score", ascending=False).reset_index(drop=True)
        top = result_df.head(int(top_n)).copy()

        st.success("Scan complete: {} stocks passed the filters; {} failed to download; {} were filtered out.".format(
            len(results), len(failed), len(filtered)
        ))

        st.subheader("Top swing candidates")
        display_cols = [
            "Ticker", "Price", "Score", "5D %", "10D %", "RSI",
            "ATR %", "Vol Ratio", "20D High Dist %", "Reason"
        ]
        st.dataframe(top[display_cols], use_container_width=True, hide_index=True)

        # Build the top-ranked stock objects for the normalized chart.
        top_symbols = top["Ticker"].tolist()
        result_lookup = {r["Ticker"]: r for r in results}
        top_objects = [result_lookup[s] for s in top_symbols if s in result_lookup]

        st.subheader("Top candidates — normalized performance")
        fig, ax = plt.subplots(figsize=(14, 7))
        for result in top_objects:
            d = result["_data"].tail(int(lookback_days)).copy()
            if len(d) < 2:
                continue
            start_price = float(d["AdjClose"].iloc[0])
            pct = (d["AdjClose"] / start_price - 1) * 100
            line, = ax.plot(d["Date"], pct, linewidth=2, label=result["Ticker"])
            ax.annotate(
                result["Ticker"],
                xy=(d["Date"].iloc[-1], pct.iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=line.get_color(),
                va="center"
            )

        ax.axhline(0, linewidth=1)
        ax.set_title("Top Swing Candidates — Normalized Performance (Start = 0%)")
        ax.set_ylabel("Performance (%)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Download results")
        all_csv = result_df.to_csv(index=False).encode("utf-8-sig")
        top_csv = top.to_csv(index=False).encode("utf-8-sig")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇ Download all ranked stocks (CSV)",
                data=all_csv,
                file_name="US_Swing_Scanner_All_Ranked.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "⬇ Download top candidates (CSV)",
                data=top_csv,
                file_name="US_Swing_Scanner_Top10.csv",
                mime="text/csv",
                use_container_width=True
            )

        with st.expander("Scan details"):
            st.write("Universe size: {}".format(total))
            st.write("Stocks passing filters: {}".format(len(results)))
            st.write("Download failures: {}".format(len(failed)))
            if failed:
                st.write("Failed symbols:", ", ".join(failed))
            st.write("Filtered symbols: {}".format(len(filtered)))

        st.warning(
            "A high score is not a buy recommendation. Check the live price, market trend, "
            "earnings/news, liquidity, and entry risk before trading."
        )
else:
    st.markdown("### How to use")
    st.markdown(
        "1. Adjust the filters in the left sidebar if required.  \n"
        "2. Click **Run US stock scanner**.  \n"
        "3. Review the ranked candidates and normalized performance chart.  \n"
        "4. Download the CSV files if you want to analyse the results further."
    )
