# 📈 Qullamaggie Inspired Screener

Webowy skaner akcji inspirowany strategią **Kristjana Kullamägi (Qullamaggie)**.
Zbudowany na Streamlit + Yahoo Finance.

## 🚀 Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twoj-link.streamlit.app)

## 📊 Funkcje

- Skanowanie akcji NYSE/NASDAQ w czasie rzeczywistym
- Konfigurowalny panel filtrów (RS, ATR, cena, wolumen)
- Upload własnej listy tickerów (TXT/CSV) lub wpisanie ręcznie
- Klasyfikacja setupów: Breakout / Bull Trend / Hold / Extended / Weak
- Kolorowa tabela z sortowaniem i filtrami
- Eksport wyników do CSV
- Linki do TradingView dla każdej spółki

## 🎯 Kryteria skanera

| Kryterium | Domyślnie |
|---|---|
| RS Rating (vs SPY) | ≥ 90 |
| Układ MA | Price ≥ EMA10 ≥ SMA20 ≥ SMA50 ≥ SMA100 ≥ SMA200 |
| ATR% | ≥ 3% |
| Pozycja w 20D zakresie | ≥ 40% |
| Wolumen | ≥ $5M/dzień |

## 🏷️ Klasyfikacja setupów

| Status | Opis |
|---|---|
| 🚀 Breakout | Cena blisko 52W High, układ MA byczych |
| ✅ Bull Trend | RS≥90, cena blisko EMA10 (≤10%) |
| 🟡 Hold | W trendzie, nie idealny punkt wejścia |
| ⚠️ Extended | Za daleko od EMA10 (>20%) |
| ⚫ Weak | Brak byczego układu MA |

## 🛠️ Lokalne uruchomienie

```bash
git clone https://github.com/TWOJ-LOGIN/qullamaggie-screener
cd qullamaggie-screener
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy na Streamlit Cloud

1. Wgraj repo na GitHub
2. Wejdź na [share.streamlit.io](https://share.streamlit.io)
3. Wybierz repo → branch `main` → plik `app.py`
4. Kliknij **Deploy**

## ⚠️ Disclaimer

Aplikacja służy wyłącznie celom edukacyjnym i informacyjnym.
Nie stanowi porady inwestycyjnej. Inwestowanie wiąże się z ryzykiem utraty kapitału.
