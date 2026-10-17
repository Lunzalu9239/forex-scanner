"""
Forex Strategy Scanner dashboard.
Keep this file in the same folder as scanner.py, headlines.csv and signals_log.csv.
Start it by double-clicking open_dashboard.bat
"""
import os
import pandas as pd
import streamlit as st
import scanner as sc

st.set_page_config(page_title="Forex Strategy Scanner", layout="wide")
st.title("Forex Strategy Scanner")
st.caption("Research tool. Not financial advice. Paper trade on a demo account first.")


@st.cache_data(ttl=3600, show_spinner="Scanning the market...")
def run_scan(pair):
    df = sc.add_indicators(sc.get_data(pair))
    price, atr = float(df["close"].iloc[-1]), float(df["atr"].iloc[-1])
    rows, votes = [], []
    for label, fn in sc.STRATEGIES.items():
        sig = fn(df).fillna(0).astype(int)
        today = int(sig.iloc[-1])
        if today == 0:
            continue
        stats = sc.evaluate(df, sig)
        if stats["n"] >= sc.MIN_TRADES:
            evidence = f"{stats['win']:.0%} win over {stats['n']} past signals, avg {stats['avg']*100:+.2f}%/trade"
        else:
            evidence = f"only {stats['n']} past signals (too few)"
        rows.append({
            "Strategy": label,
            "Signal": "BUY" if today > 0 else "SELL",
            "Verdict": "SUPPORTED" if stats["ok"] else "unproven",
            "History": evidence,
        })
        if stats["ok"]:
            votes.append(today)

    if not votes:
        move = "NO TRADE (nothing proven is active)"
    elif sum(votes) == 0:
        move = "NO TRADE (proven strategies disagree)"
    else:
        d = 1 if sum(votes) > 0 else -1
        stop = price - d * 1.5 * atr
        move = f"{'BUY' if d > 0 else 'SELL'}, stop near {stop:.4f}, exit after ~{sc.HORIZON} days"
    return df, price, rows, move


top = st.columns([1, 5])
if top[0].button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
top[1].caption("Data is cached for 1 hour. Best read after the daily candle closes (morning in Nairobi).")

tab1, tab2, tab3, tab4 = st.tabs(["Today's scan", "Price chart", "Signal log", "News sentiment"])

# ---------- Tab 1: scan ----------
with tab1:
    results = {}
    for p in sc.PAIRS:
        try:
            results[p] = run_scan(p)
        except Exception as e:
            st.warning(f"{p}: could not load data ({e})")

    summary = pd.DataFrame([{
        "Pair": p.replace("=X", ""),
        "Last close": round(r[1], 4),
        "Active signals": len(r[2]),
        "Suggested move": r[3],
    } for p, r in results.items()])
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Details")
    for p, r in results.items():
        with st.expander(f"{p.replace('=X', '')}  -  {r[3]}"):
            if r[2]:
                st.dataframe(pd.DataFrame(r[2]), use_container_width=True, hide_index=True)
            else:
                st.write("No strategy is active on this pair today.")
    st.info("SUPPORTED = held a 55%+ win rate in both halves of 8 years of data. "
            "That is a historical result, not a promise.")

# ---------- Tab 2: chart ----------
with tab2:
    if results:
        choice = st.selectbox("Pair", list(results.keys()), format_func=lambda x: x.replace("=X", ""))
        d = results[choice][0][["close", "ema20", "ema50"]].tail(250)
        st.line_chart(d)
        st.caption("Last ~250 trading days: price with 20 and 50 day exponential averages.")

# ---------- Tab 3: signal log ----------
with tab3:
    if os.path.exists(sc.LOG_FILE):
        log = pd.read_csv(sc.LOG_FILE).sort_values("date", ascending=False)
        st.write(f"{len(log)} signals logged. In ~2 months, compare these with what the market really did.")
        st.dataframe(log, use_container_width=True, hide_index=True)
    else:
        st.write("No log yet. Run scanner.py (or the daily task) to start it.")

# ---------- Tab 4: sentiment ----------
with tab4:
    if os.path.exists("headlines.csv"):
        news = pd.read_csv("headlines.csv", parse_dates=["date"])
        st.subheader("Average daily headline sentiment")
        st.line_chart(news.groupby("date")["score"].mean())
        st.subheader("Latest headlines")
        st.dataframe(news.sort_values("date", ascending=False).head(25)[["date", "source", "title", "score"]],
                     use_container_width=True, hide_index=True)
        st.caption("Earlier tests found no reliable link between this sentiment and next-day moves. Shown for research only.")
    else:
        st.write("No headlines collected yet.")
