"""
Qullamaggie Inspired Screener — Streamlit Web App
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Qullamaggie Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d0f14; color: #e2e6f0; }
  section[data-testid="stSidebar"] { background-color: #161920; }
  .metric-card {
    background: #161920; border: 1px solid #252a3a;
    border-radius: 10px; padding: 14px 18px; text-align: center;
  }
  .metric-val { font-size: 28px; font-weight: 700; }
  .metric-lbl { font-size: 11px; color: #7a8299; margin-top: 2px; }
  .tag-green  { background:#1a3a27; color:#26ff7f; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .tag-red    { background:#3a1a1a; color:#ff6b6b; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .tag-yellow { background:#3a3010; color:#f0c040; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .tag-grey   { background:#252535; color:#888;    padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  div[data-testid="stDataFrame"] { border-radius: 8px; }
  .stButton>button {
    background: #6c8eff; color: white; border: none;
    border-radius: 8px; font-weight: 600; padding: 10px 24px;
    width: 100%;
  }
  .stButton>button:hover { background: #5a7aee; }
  .upload-info { background:#1e2230; border:1px solid #252a3a; border-radius:8px; padding:12px; font-size:12px; color:#7a8299; }
</style>
""", unsafe_allow_html=True)

# ── DEFAULT TICKERS ──────────────────────────────────────────
DEFAULT_TICKERS = [
    "NVDA","AMD","META","AAPL","MSFT","AMZN","GOOGL","TSLA","AVGO","ARM",
    "ORCL","CRM","NOW","PLTR","SNOW","MSTR","COIN","HOOD","RBLX","U",
    "SMCI","AMAT","LRCX","KLAC","MRVL","QCOM","MU","INTC","TSM","ASML",
    "ON","WOLF","CRUS","MTSI","AMBA","ALGM","AEHR","ACLS","MPWR","SLAB",
    "DXCM","ISRG","PODD","INSP","TMDX","RXRX","NVCR","ACAD","ALNY","IONS",
    "RGEN","KRYS","BEAM","EDIT","CRSP","NTLA","FOLD","BLUE","FATE","RCUS",
    "CCJ","UEC","NXE","DNN","UUUU","SMR","OKLO","BWXT","NNE","LEU",
    "MARA","RIOT","HUT","CLSK","IREN","BTBT","WULF","CIFR","CORZ","BTDR",
    "KTOS","AXON","LHX","NOC","GD","RTX","HEI","TDG","LDOS","SAIC",
    "CELH","HIMS","DUOL","SOUN","PRCT","IONQ","RGTI","QBTS","ACHR","JOBY",
    "GS","MS","JPM","BAC","V","MA","AXP","AFRM","UPST","SOFI","NU","MELI",
    "CAVA","BROS","WING","TXRH","CMG","SBUX",
    "DDOG","NET","ZS","CRWD","S","PANW","FTNT","CYBR","GTLB","BILL","HUBS","MNDY",
    "BW","PLUG","FCEL","BLDP","RDW","RKLB","ASTS","LUNR","SATL","NRGV",
    "AXSM","ITCI","ROIV","KYMR","XPEV","LI","NIO","SE","GRAB","STNE",
]

# ── TECH FUNCTIONS ───────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(window=n).mean()

def atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=n).mean()

def rs_rating(ticker_ret, bench_ret):
    def wr(ret):
        q = max(1, len(ret) // 4)
        r = [(1 + ret.iloc[-i*q:-(i-1)*q if i > 1 else None]).prod() - 1 for i in range(1, 5)]
        return 2*r[0] + r[1] + r[2] + r[3]
    diff = wr(ticker_ret) - wr(bench_ret)
    return round(min(99, max(1, 50 + 50 * np.tanh(diff * 3))), 1)

def classify(row):
    if not row["ma_aligned"]:
        return "Weak", "⚫"
    if row["price"] >= row["high_52w"] * 0.98:
        return "Breakout 🚀", "🟢"
    if row["pct_from_ema10"] > 20:
        return "Extended ⚠️", "🔴"
    if row["rs"] >= 90 and row["pct_from_ema10"] <= 10:
        return "Bull Trend ✅", "🟢"
    if row["rs"] >= 75:
        return "Hold 🟡", "🟡"
    return "Weak", "⚫"

def analyze(ticker, spy_ret, cfg):
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if hist is None or len(hist) < 60:
            return None
        c, h, l, v = hist["Close"].dropna(), hist["High"].dropna(), hist["Low"].dropna(), hist["Volume"].dropna()
        price = float(c.iloc[-1])
        if price < cfg["min_price"]: return None

        e10  = float(ema(c,10).iloc[-1])
        s20  = float(sma(c,20).iloc[-1])
        s50  = float(sma(c,50).iloc[-1])
        s100 = float(sma(c,100).iloc[-1])
        s200 = float(sma(c,200).iloc[-1])

        ma_ok = (price>=e10*0.99 and e10>=s20*0.99 and s20>=s50*0.99
                 and s50>=s100*0.98 and s100>=s200*0.97)

        atr_v   = float(atr(h,l,c).iloc[-1])
        atr_pct = round(atr_v / price * 100, 2)
        if atr_pct < cfg["min_atr_pct"]: return None

        h52 = float(h.tail(252).max())
        l52 = float(l.tail(252).min())
        h20, l20 = float(h.tail(20).max()), float(l.tail(20).min())
        rng20 = h20 - l20
        p20 = round((price - l20) / rng20 * 100, 1) if rng20 > 0 else 0
        if p20 < cfg["min_range_pct"]: return None

        avg_vol_m = round(float(v.tail(20).mean()) * price / 1e6, 1)
        if avg_vol_m < cfg["min_vol_m"]: return None

        ret    = c.pct_change().dropna()
        common = ret.index.intersection(spy_ret.index)
        if len(common) < 30: return None
        rs = rs_rating(ret.loc[common], spy_ret.loc[common])
        if rs < cfg["min_rs"]: return None

        def r(n): return round((price / float(c.iloc[-n]) - 1) * 100, 1) if len(c) > n else None

        row = dict(
            Ticker=ticker,
            Price=round(price, 2),
            RS=rs,
            MA_Aligned=ma_ok,
            ATR_pct=atr_pct,
            Range_20D=p20,
            From_52W_High=round((price/h52-1)*100,1),
            EMA10=round(e10,2), SMA20=round(s20,2), SMA50=round(s50,2),
            SMA200=round(s200,2),
            Pct_EMA10=round((price/e10-1)*100,1),
            Pct_SMA50=round((price/s50-1)*100,1),
            Pct_SMA200=round((price/s200-1)*100,1),
            Ret_1M=r(21), Ret_3M=r(63), Ret_6M=r(126),
            Vol_M=avg_vol_m,
            SL_1ATR=round(price-atr_v,2),
            SL_2ATR=round(price-2*atr_v,2),
            T1=round(price+2*atr_v,2),
            T2=round(price+3*atr_v,2),
            high_52w=h52,
            pct_from_ema10=round((price/e10-1)*100,1),
        )
        row["Status"], row["Dot"] = classify(row)
        return row
    except:
        return None

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Qullamaggie Screener")
    st.markdown("---")

    st.markdown("### ⚙️ Filtry skanera")
    min_rs    = st.slider("Min RS Rating",    50, 99, 90, 1)
    min_atr   = st.slider("Min ATR%",         1.0, 15.0, 3.0, 0.5)
    min_price = st.number_input("Min Cena ($)", 1.0, 500.0, 5.0, 1.0)
    min_vol   = st.number_input("Min Vol ($M/dzień)", 1.0, 100.0, 5.0, 1.0)
    min_range = st.slider("Min 20D Range%",   0, 100, 40, 5)
    ma_required = st.checkbox("Tylko MA Aligned", value=False)

    cfg = dict(
        min_rs=min_rs, min_atr_pct=min_atr, min_price=min_price,
        min_vol_m=min_vol, min_range_pct=min_range, ma_req=ma_required,
    )

    st.markdown("---")
    st.markdown("### 📂 Lista spółek")

    ticker_mode = st.radio("Źródło tickerów", ["Domyślna lista", "Wgraj plik", "Wpisz ręcznie"])

    custom_tickers = None

    if ticker_mode == "Wgraj plik":
        uploaded = st.file_uploader("TXT lub CSV (jeden ticker per linia)", type=["txt","csv"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            custom_tickers = [
                t.strip().upper()
                for line in content.splitlines()
                for t in line.replace(","," ").replace(";"," ").split()
                if t.strip() and 1 <= len(t.strip()) <= 6
            ]
            st.success(f"✅ Wczytano {len(custom_tickers)} tickerów")
            with st.expander("Podgląd listy"):
                st.write("  ".join(custom_tickers))

    elif ticker_mode == "Wpisz ręcznie":
        manual = st.text_area("Tickery (oddziel przecinkiem lub enterem)", height=120,
                              placeholder="NVDA, AMD, TSLA\nAAPL\nMETA")
        if manual.strip():
            custom_tickers = [
                t.strip().upper()
                for t in manual.replace("\n",",").split(",")
                if t.strip() and 1 <= len(t.strip()) <= 6
            ]
            st.info(f"📋 {len(custom_tickers)} tickerów")

    tickers = custom_tickers if custom_tickers else DEFAULT_TICKERS
    st.caption(f"Do skanowania: **{len(tickers)}** spółek")

    st.markdown("---")
    run_btn = st.button("🚀 Uruchom skanowanie", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#7a8299;line-height:1.6'>
    Inspirowane strategią <b>Kristjana Kullamägi</b><br>
    Dane: Yahoo Finance<br>
    ⚠️ Tylko do celów edukacyjnych
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────
st.markdown("# 📈 Qullamaggie Inspired Screener")
st.markdown(f"*{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · Yahoo Finance · NYSE/NASDAQ*")

# Criteria info
with st.expander("ℹ️ Kryteria skanera"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Filtry wejściowe:**
        - RS Rating ≥ próg (siła vs SPY)
        - Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200
        - ATR% ≥ próg (zmienność)
        - Pozycja w 20D zakresie ≥ próg
        """)
    with col2:
        st.markdown("""
        **Klasyfikacja setupów:**
        - 🚀 **Breakout** — blisko 52W High
        - ✅ **Bull Trend** — silny trend, RS≥90, blisko EMA10
        - 🟡 **Hold** — w trendzie, nie idealny punkt wejścia
        - ⚠️ **Extended** — za daleko od EMA10 (>20%)
        - ⚫ **Weak** — brak byczego układu MA
        """)

# ── SCAN ─────────────────────────────────────────────────────
if run_btn:
    st.markdown("---")

    # Fetch SPY
    spy_placeholder = st.empty()
    spy_placeholder.info("📡 Pobieram dane benchmarku SPY...")
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        spy_ret  = spy_hist["Close"].pct_change().dropna()
        spy_placeholder.success("✅ SPY załadowany")
    except Exception as e:
        spy_placeholder.error(f"❌ Błąd pobierania SPY: {e}")
        st.stop()

    # Progress bar
    st.markdown(f"**Skanowanie {len(tickers)} spółek...**")
    progress_bar  = st.progress(0)
    status_text   = st.empty()
    results       = []

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers))
        status_text.markdown(f"`{ticker}` — {i+1}/{len(tickers)}")

        row = analyze(ticker, spy_ret, cfg)
        if row:
            results.append(row)
        time.sleep(0.25)

    progress_bar.empty()
    status_text.empty()
    spy_placeholder.empty()

    if not results:
        st.warning("⚠️ Brak spółek spełniających kryteria. Spróbuj zmniejszyć progi w panelu bocznym.")
        st.stop()

    df = pd.DataFrame(results).sort_values("RS", ascending=False)

    # Opcjonalny filtr MA
    if cfg["ma_req"]:
        df = df[df["MA_Aligned"] == True]

    # Zapisz do session state
    st.session_state["results"] = df
    st.session_state["scan_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# ── WYNIKI ────────────────────────────────────────────────────
if "results" in st.session_state:
    df = st.session_state["results"]
    scan_time = st.session_state.get("scan_time","")

    st.markdown(f"### Wyniki skanowania `{scan_time}` — **{len(df)} spółek**")

    # ── STAT CARDS ──
    col1, col2, col3, col4, col5 = st.columns(5)
    counts = {
        "Breakout":  len(df[df["Status"].str.contains("Breakout")]),
        "Bull":      len(df[df["Status"].str.contains("Bull")]),
        "Hold":      len(df[df["Status"].str.contains("Hold")]),
        "Extended":  len(df[df["Status"].str.contains("Extended")]),
        "Weak":      len(df[df["Status"].str.contains("Weak")]),
    }
    for col, (label, val, color) in zip(
        [col1,col2,col3,col4,col5],
        [("🚀 Breakout",  counts["Breakout"], "#26ff7f"),
         ("✅ Bull Trend", counts["Bull"],     "#26a65b"),
         ("🟡 Hold",       counts["Hold"],     "#f0c040"),
         ("⚠️ Extended",   counts["Extended"], "#e84545"),
         ("⚫ Weak",        counts["Weak"],     "#555")],
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val" style="color:{color}">{val}</div>
          <div class="metric-lbl">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── FILTERS ──
    fc1, fc2, fc3, fc4 = st.columns([2,2,2,2])
    with fc1:
        f_status = st.selectbox("Status", ["Wszystkie","Breakout","Bull Trend","Hold","Extended","Weak"])
    with fc2:
        f_ma = st.selectbox("MA Aligned", ["Wszystkie","Tak","Nie"])
    with fc3:
        f_rs_min = st.number_input("Min RS (tabela)", 0, 99, 0)
    with fc4:
        f_search = st.text_input("Szukaj tickera", "")

    dff = df.copy()
    if f_status != "Wszystkie":
        dff = dff[dff["Status"].str.contains(f_status)]
    if f_ma == "Tak":
        dff = dff[dff["MA_Aligned"] == True]
    elif f_ma == "Nie":
        dff = dff[dff["MA_Aligned"] == False]
    if f_rs_min > 0:
        dff = dff[dff["RS"] >= f_rs_min]
    if f_search.strip():
        dff = dff[dff["Ticker"].str.upper().str.contains(f_search.strip().upper())]

    st.caption(f"Wyświetlane: **{len(dff)}** spółek")

    # ── TABLE ──
    display_cols = ["Ticker","Status","RS","Price","Pct_EMA10","Pct_SMA50","Pct_SMA200",
                    "MA_Aligned","ATR_pct","Range_20D","From_52W_High",
                    "Ret_1M","Ret_3M","Ret_6M","Vol_M","SL_1ATR","SL_2ATR","T1","T2"]

    dff_display = dff[display_cols].copy()
    dff_display.columns = ["Ticker","Status","RS","Cena $","% EMA10","% SMA50","% SMA200",
                           "MA OK","ATR%","20D Rng%","52W High%",
                           "1M%","3M%","6M%","Vol $M","SL 1ATR","SL 2ATR","T1 (2R)","T2 (3R)"]

    def style_table(df_s):
        def color_pct(val):
            if pd.isna(val): return ""
            try:
                v = float(val)
                if v > 0:  return "color: #26ff7f"
                if v < 0:  return "color: #e84545"
            except: pass
            return ""
        def color_status(val):
            s = str(val)
            if "Breakout" in s: return "color: #26ff7f; font-weight: bold"
            if "Bull"     in s: return "color: #26a65b; font-weight: bold"
            if "Extended" in s: return "color: #e84545"
            if "Hold"     in s: return "color: #f0c040"
            return "color: #555"
        def color_rs(val):
            try:
                v = float(val)
                if v >= 95: return "color: #26ff7f; font-weight:bold"
                if v >= 90: return "color: #6c8eff"
                if v >= 80: return "color: #f0c040"
            except: pass
            return ""

        pct_cols = ["% EMA10","% SMA50","% SMA200","52W High%","1M%","3M%","6M%"]
        styled = df_s.style
        for col in pct_cols:
            if col in df_s.columns:
                styled = styled.applymap(color_pct, subset=[col])
        if "Status" in df_s.columns:
            styled = styled.applymap(color_status, subset=["Status"])
        if "RS" in df_s.columns:
            styled = styled.applymap(color_rs, subset=["RS"])
        return styled

    st.dataframe(
        style_table(dff_display),
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    # ── LINKS ──
    st.markdown("**🔗 Linki do wykresów**")
    link_cols = st.columns(min(10, len(dff)))
    for i, (_, row) in enumerate(dff.head(10).iterrows()):
        t = row["Ticker"]
        link_cols[i % len(link_cols)].markdown(
            f"[{t}](https://www.tradingview.com/chart/?symbol={t})", unsafe_allow_html=True
        )

    # ── EXPORT ──
    st.markdown("---")
    csv = dff[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Pobierz wyniki CSV",
        data=csv,
        file_name=f"qullamaggie_{datetime.date.today()}.csv",
        mime="text/csv",
    )

else:
    # Pusty stan startowy
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#7a8299'>
      <div style='font-size:48px;margin-bottom:16px'>📈</div>
      <div style='font-size:18px;font-weight:600;color:#e2e6f0;margin-bottom:8px'>
        Gotowy do skanowania
      </div>
      <div style='font-size:14px'>
        Ustaw filtry w panelu bocznym i kliknij <b>🚀 Uruchom skanowanie</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Pokaż kryteria jako reminder
    st.markdown("#### Aktywne kryteria:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Min RS Rating", min_rs)
    c2.metric("Min ATR%", f"{min_atr}%")
    c3.metric("Min Vol", f"${min_vol}M")
    c1.metric("Min Cena", f"${min_price}")
    c2.metric("Min 20D Range", f"{min_range}%")
    c3.metric("MA Aligned wymagany", "Tak" if ma_required else "Nie")
