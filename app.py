"""
Qullamaggie Inspired Screener — Streamlit Web App v2
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time, datetime, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Qullamaggie Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0d0f14; color: #e2e6f0; }
section[data-testid="stSidebar"] { background-color: #161920 !important; }
.metric-card {
  background:#161920; border:1px solid #252a3a;
  border-radius:10px; padding:14px 18px; text-align:center; margin-bottom:4px;
}
.metric-val { font-size:26px; font-weight:700; }
.metric-lbl { font-size:11px; color:#7a8299; margin-top:2px; }
div[data-testid="stDataFrame"] { border-radius:8px; }
.stButton>button {
  background:#6c8eff !important; color:white !important; border:none !important;
  border-radius:8px !important; font-weight:600 !important; padding:10px 24px !important;
  width:100% !important;
}
.debug-box {
  background:#1e2230; border:1px solid #252a3a; border-radius:8px;
  padding:12px; font-size:11px; color:#7a8299; font-family:monospace;
  max-height:200px; overflow-y:auto;
}
</style>
""", unsafe_allow_html=True)

# ── TICKERS ──────────────────────────────────────────────────
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
    "DDOG","NET","ZS","CRWD","PANW","FTNT","CYBR","GTLB","BILL","HUBS","MNDY",
    "BW","PLUG","FCEL","BLDP","RDW","RKLB","ASTS","LUNR","SATL","NRGV",
    "AXSM","ITCI","ROIV","KYMR","XPEV","LI","NIO","SE","GRAB","STNE",
]

# ── TECH HELPERS ─────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(window=n).mean()

def calc_atr(high, low, close, n=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=n).mean()

def calc_rs(ticker_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    RS Rating 1–99 oparty na 12M stopie zwrotu z wagą na ostatni kwartał.
    Prosta i odporna metoda — bez tanh.
    """
    # Wyrównaj indeksy
    idx = ticker_ret.index.intersection(bench_ret.index)
    if len(idx) < 20:
        return 50.0
    tr = ticker_ret.loc[idx]
    br = bench_ret.loc[idx]

    # Całkowita stopa 12M
    ret_t = float((1 + tr).prod() - 1)
    ret_b = float((1 + br).prod() - 1)

    # Stopa ostatnich 3M (waga x2)
    q = max(1, len(idx) // 4)
    ret_t3 = float((1 + tr.iloc[-q:]).prod() - 1)
    ret_b3 = float((1 + br.iloc[-q:]).prod() - 1)

    score = (2 * (ret_t3 - ret_b3) + (ret_t - ret_b)) * 100
    # Mapuj na 1-99 (score ~-50..+50 → 1..99)
    rs = 50 + score
    return round(float(np.clip(rs, 1, 99)), 1)

def classify(price, ema10, high_52w, pct_ema10, rs, ma_ok):
    if not ma_ok:
        return "Weak ⚫"
    if price >= high_52w * 0.98:
        return "Breakout 🚀"
    if pct_ema10 > 20:
        return "Extended ⚠️"
    if rs >= 90 and pct_ema10 <= 10:
        return "Bull Trend ✅"
    if rs >= 75:
        return "Hold 🟡"
    return "Weak ⚫"

def analyze(ticker: str, spy_ret: pd.Series, cfg: dict, log: list) -> dict | None:
    try:
        t_obj = yf.Ticker(ticker)
        hist  = t_obj.history(period="1y", interval="1d", auto_adjust=True)

        if hist is None or len(hist) < 60:
            log.append(f"  SKIP {ticker}: za mało danych ({len(hist) if hist is not None else 0} dni)")
            return None

        c = hist["Close"].dropna()
        h = hist["High"].dropna()
        l = hist["Low"].dropna()
        v = hist["Volume"].dropna()

        price = float(c.iloc[-1])
        if price < cfg["min_price"]:
            log.append(f"  SKIP {ticker}: cena {price:.2f} < {cfg['min_price']}")
            return None

        e10  = float(ema(c, 10).iloc[-1])
        s20  = float(sma(c, 20).iloc[-1])
        s50  = float(sma(c, 50).iloc[-1])
        s100 = float(sma(c, 100).iloc[-1])
        s200 = float(sma(c, 200).iloc[-1])

        ma_ok = (price  >= e10  * 0.99 and
                 e10    >= s20  * 0.99 and
                 s20    >= s50  * 0.99 and
                 s50    >= s100 * 0.98 and
                 s100   >= s200 * 0.97)

        atr_v   = float(calc_atr(h, l, c).iloc[-1])
        atr_pct = round(atr_v / price * 100, 2)
        if atr_pct < cfg["min_atr_pct"]:
            log.append(f"  SKIP {ticker}: ATR% {atr_pct} < {cfg['min_atr_pct']}")
            return None

        h52  = float(h.tail(252).max())
        l52  = float(l.tail(252).min())
        h20  = float(h.tail(20).max())
        l20  = float(l.tail(20).min())
        rng20 = h20 - l20
        p20   = round((price - l20) / rng20 * 100, 1) if rng20 > 0 else 0
        if p20 < cfg["min_range_pct"]:
            log.append(f"  SKIP {ticker}: 20D Range {p20}% < {cfg['min_range_pct']}%")
            return None

        avg_vol_m = round(float(v.tail(20).mean()) * price / 1e6, 1)
        if avg_vol_m < cfg["min_vol_m"]:
            log.append(f"  SKIP {ticker}: Vol {avg_vol_m}M < {cfg['min_vol_m']}M")
            return None

        # RS
        ret    = c.pct_change().dropna()
        common = ret.index.normalize().intersection(spy_ret.index.normalize())
        if len(common) < 30:
            log.append(f"  SKIP {ticker}: za mało wspólnych dni z SPY ({len(common)})")
            return None
        # Przekształć do zwykłego indeksu dat
        ret_aligned   = ret.copy();  ret_aligned.index   = ret.index.normalize()
        spy_aligned   = spy_ret.copy(); spy_aligned.index = spy_ret.index.normalize()
        rs = calc_rs(ret_aligned.loc[common], spy_aligned.loc[common])

        if rs < cfg["min_rs"]:
            log.append(f"  SKIP {ticker}: RS {rs:.0f} < {cfg['min_rs']}")
            return None

        def ret_n(n):
            return round((price / float(c.iloc[-n]) - 1) * 100, 1) if len(c) > n else None

        pct_ema10  = round((price / e10  - 1) * 100, 1)
        pct_sma50  = round((price / s50  - 1) * 100, 1)
        pct_sma200 = round((price / s200 - 1) * 100, 1)

        status = classify(price, e10, h52, pct_ema10, rs, ma_ok)
        log.append(f"  ✅ {ticker}: RS={rs:.0f}  {status}  ${price:.2f}")

        return dict(
            Ticker=ticker, Status=status,
            RS=rs, Price=round(price, 2),
            Pct_EMA10=pct_ema10, Pct_SMA50=pct_sma50, Pct_SMA200=pct_sma200,
            MA_Aligned=ma_ok,
            ATR_pct=atr_pct, Range_20D=p20,
            From_52W_High=round((price / h52 - 1) * 100, 1),
            Ret_1M=ret_n(21), Ret_3M=ret_n(63), Ret_6M=ret_n(126),
            Vol_M=avg_vol_m,
            SL_1ATR=round(price - atr_v, 2),
            SL_2ATR=round(price - 2 * atr_v, 2),
            T1=round(price + 2 * atr_v, 2),
            T2=round(price + 3 * atr_v, 2),
            EMA10=round(e10, 2), SMA50=round(s50, 2), SMA200=round(s200, 2),
        )
    except Exception as ex:
        log.append(f"  ERR {ticker}: {ex}")
        return None

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Qullamaggie")
    st.markdown("---")
    st.markdown("### ⚙️ Filtry skanera")

    min_rs    = st.slider("Min RS Rating",   50, 99, 85, 1,
                          help="Siła relatywna vs SPY (50=neutralna, 99=top 1%)")
    min_atr   = st.slider("Min ATR%",        0.5, 15.0, 2.0, 0.5,
                          help="Minimalna zmienność jako % ceny")
    min_price = st.number_input("Min Cena ($)", 1.0, 500.0, 3.0, 1.0)
    min_vol   = st.number_input("Min Vol ($M/dzień)", 0.5, 100.0, 2.0, 0.5)
    min_range = st.slider("Min 20D Range%",  0, 100, 30, 5,
                          help="Pozycja ceny w 20-dniowym zakresie")
    ma_required = st.checkbox("Tylko MA Aligned", value=False)

    cfg = dict(
        min_rs=min_rs, min_atr_pct=min_atr, min_price=min_price,
        min_vol_m=min_vol, min_range_pct=min_range, ma_req=ma_required,
    )

    st.markdown("---")
    st.markdown("### 📂 Lista spółek")
    ticker_mode = st.radio("Źródło", ["Domyślna lista (~130)", "Wgraj plik TXT/CSV", "Wpisz ręcznie"])

    custom_tickers = None

    if ticker_mode == "Wgraj plik TXT/CSV":
        uploaded = st.file_uploader("Plik z tickerami", type=["txt","csv"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            custom_tickers = list(dict.fromkeys([
                t.strip().upper()
                for line in content.splitlines()
                for t in line.replace(","," ").replace(";"," ").split()
                if t.strip() and 1 <= len(t.strip()) <= 6
            ]))
            st.success(f"✅ {len(custom_tickers)} tickerów")

    elif ticker_mode == "Wpisz ręcznie":
        manual = st.text_area("Tickery (przecinek lub enter)", height=100,
                              placeholder="NVDA\nAMD\nTSLA, AAPL")
        if manual.strip():
            custom_tickers = list(dict.fromkeys([
                t.strip().upper()
                for t in manual.replace("\n",",").split(",")
                if t.strip() and 1 <= len(t.strip()) <= 6
            ]))
            st.info(f"📋 {len(custom_tickers)} tickerów gotowych")
        else:
            st.caption("Wpisz tickery powyżej, żeby ich użyć")

    tickers = custom_tickers if custom_tickers else DEFAULT_TICKERS
    st.caption(f"Do skanowania: **{len(tickers)}** spółek")

    st.markdown("---")
    show_debug = st.checkbox("🔍 Pokaż log skanowania", value=False)
    run_btn = st.button("🚀 Uruchom skanowanie", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#7a8299;line-height:1.7'>
    Inspirowane strategią <b>Kristjana Kullamägi</b><br>
    Dane: Yahoo Finance · Tylko edukacyjnie<br>
    ⚠️ Nie jest poradą inwestycyjną
    </div>""", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────
st.markdown("# 📈 Qullamaggie Inspired Screener")
st.markdown(f"*{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · Yahoo Finance · NYSE/NASDAQ*")

with st.expander("ℹ️ Kryteria skanera"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""**Filtry wejściowe:**
- RS Rating ≥ próg (siła vs SPY)
- Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200
- ATR% ≥ próg · Pozycja 20D range ≥ próg""")
    with c2:
        st.markdown("""**Klasyfikacja:**
- 🚀 **Breakout** — blisko 52W High
- ✅ **Bull Trend** — RS≥90, ≤10% od EMA10
- 🟡 **Hold** — w trendzie
- ⚠️ **Extended** — >20% od EMA10
- ⚫ **Weak** — brak byczego MA""")

# ── SCAN ─────────────────────────────────────────────────────
if run_btn:
    scan_log = []
    st.markdown("---")

    spy_ph = st.empty()
    spy_ph.info("📡 Pobieram SPY benchmark...")
    try:
        spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
        spy_ret  = spy_hist["Close"].pct_change().dropna()
        spy_ret.index = spy_ret.index.normalize()
        spy_ph.success(f"✅ SPY: {len(spy_ret)} dni danych")
    except Exception as e:
        spy_ph.error(f"❌ Błąd SPY: {e}")
        st.stop()

    st.markdown(f"**Skanowanie {len(tickers)} spółek...**  *(może zająć kilka minut)*")
    prog  = st.progress(0)
    stxt  = st.empty()
    found = st.empty()
    results = []

    for i, ticker in enumerate(tickers):
        prog.progress((i + 1) / len(tickers))
        stxt.markdown(f"Analizuję: `{ticker}` — {i+1}/{len(tickers)} &nbsp; ✅ znaleziono: **{len(results)}**")

        row = analyze(ticker, spy_ret, cfg, scan_log)
        if row:
            results.append(row)
        time.sleep(0.2)

    prog.empty()
    stxt.empty()
    spy_ph.empty()

    if show_debug:
        with st.expander(f"📋 Log skanowania ({len(scan_log)} wpisów)"):
            st.markdown('<div class="debug-box">' +
                        "<br>".join(scan_log[-200:]) + "</div>",
                        unsafe_allow_html=True)

    if not results:
        st.error("⚠️ Brak spółek spełniających kryteria.")
        st.info("💡 Wskazówki: zmniejsz Min RS do 70–80, Min ATR do 1%, Min Vol do 1M")
        st.stop()

    df = pd.DataFrame(results).sort_values("RS", ascending=False)
    if cfg["ma_req"]:
        df = df[df["MA_Aligned"]]

    st.session_state["results"]   = df
    st.session_state["scan_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["scan_log"]  = scan_log

# ── WYNIKI ────────────────────────────────────────────────────
if "results" in st.session_state:
    df        = st.session_state["results"]
    scan_time = st.session_state.get("scan_time","")

    st.markdown(f"### Wyniki — `{scan_time}` — **{len(df)} spółek**")

    # Stat cards
    cols = st.columns(5)
    for col, (label, key, color) in zip(cols, [
        ("🚀 Breakout",  "Breakout",  "#26ff7f"),
        ("✅ Bull Trend","Bull",       "#26a65b"),
        ("🟡 Hold",      "Hold",       "#f0c040"),
        ("⚠️ Extended",  "Extended",  "#e84545"),
        ("⚫ Weak",       "Weak",       "#666"),
    ]):
        n = len(df[df["Status"].str.contains(key)])
        col.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color}">{n}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Table filters
    tf1, tf2, tf3, tf4 = st.columns([2,2,2,2])
    with tf1: f_status = st.selectbox("Status", ["Wszystkie","Breakout","Bull","Hold","Extended","Weak"])
    with tf2: f_ma     = st.selectbox("MA", ["Wszystkie","Aligned ✅","Misaligned ❌"])
    with tf3: f_rs     = st.number_input("Min RS (tabela)", 0, 99, 0, key="tbl_rs")
    with tf4: f_q      = st.text_input("Szukaj tickera", key="tbl_q")

    dff = df.copy()
    if f_status != "Wszystkie": dff = dff[dff["Status"].str.contains(f_status)]
    if f_ma == "Aligned ✅":    dff = dff[dff["MA_Aligned"]==True]
    if f_ma == "Misaligned ❌": dff = dff[dff["MA_Aligned"]==False]
    if f_rs > 0:                dff = dff[dff["RS"] >= f_rs]
    if f_q.strip():             dff = dff[dff["Ticker"].str.upper().str.contains(f_q.strip().upper())]

    st.caption(f"Wyświetlane: **{len(dff)}** spółek")

    disp = dff[["Ticker","Status","RS","Price","Pct_EMA10","Pct_SMA50","Pct_SMA200",
                "MA_Aligned","ATR_pct","Range_20D","From_52W_High",
                "Ret_1M","Ret_3M","Ret_6M","Vol_M","SL_1ATR","SL_2ATR","T1","T2"]].copy()
    disp.columns = ["Ticker","Status","RS","Cena $","% EMA10","% SMA50","% SMA200",
                    "MA OK","ATR%","20D%","52W%","1M%","3M%","6M%","Vol $M",
                    "SL 1ATR","SL 2ATR","T1 2R","T2 3R"]

    def style_df(s):
        def cp(v):
            try:
                f = float(v)
                return "color:#26ff7f" if f>0 else ("color:#e84545" if f<0 else "")
            except: return ""
        def cs(v):
            sv = str(v)
            if "Breakout" in sv: return "color:#26ff7f;font-weight:bold"
            if "Bull"     in sv: return "color:#26a65b;font-weight:bold"
            if "Extended" in sv: return "color:#e84545"
            if "Hold"     in sv: return "color:#f0c040"
            return "color:#555"
        def cr(v):
            try:
                f = float(v)
                if f>=95: return "color:#26ff7f;font-weight:bold"
                if f>=90: return "color:#6c8eff;font-weight:bold"
                if f>=80: return "color:#f0c040"
            except: return ""
            return ""
        st2 = s.style
        for col in ["% EMA10","% SMA50","% SMA200","52W%","1M%","3M%","6M%"]:
            if col in s.columns: st2 = st2.applymap(cp, subset=[col])
        if "Status" in s.columns: st2 = st2.applymap(cs, subset=["Status"])
        if "RS"     in s.columns: st2 = st2.applymap(cr, subset=["RS"])
        return st2

    st.dataframe(style_df(disp), use_container_width=True, height=500, hide_index=True)

    # TradingView links
    if len(dff) > 0:
        st.markdown("**📊 TradingView**")
        links = "  |  ".join(
            f"[{r['Ticker']}](https://www.tradingview.com/chart/?symbol={r['Ticker']})"
            for _, r in dff.head(20).iterrows()
        )
        st.markdown(links)

    # Export
    st.markdown("---")
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Pobierz CSV", csv,
                       f"qullamaggie_{datetime.date.today()}.csv", "text/csv")

else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#7a8299'>
      <div style='font-size:52px'>📈</div>
      <div style='font-size:18px;font-weight:600;color:#e2e6f0;margin:12px 0 6px'>Gotowy do skanowania</div>
      <div style='font-size:14px'>Ustaw filtry w panelu bocznym i kliknij <b>🚀 Uruchom skanowanie</b></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Aktywne kryteria:")
    r1, r2, r3 = st.columns(3)
    r1.metric("Min RS Rating", min_rs)
    r2.metric("Min ATR%",      f"{min_atr}%")
    r3.metric("Min Vol",       f"${min_vol}M")
    r1.metric("Min Cena",      f"${min_price}")
    r2.metric("Min 20D Range", f"{min_range}%")
    r3.metric("MA wymagane",   "Tak" if ma_required else "Nie")
