"""
Forex strategy scanner (daily timeframe).

For each pair it checks which classic strategies are active TODAY, shows how
each strategy has performed historically on that pair, and gives a suggested
direction ONLY from strategies whose history held up in both halves of the data.

Setup:  py -3.12 -m pip install yfinance pandas numpy
Run:    py -3.12 scanner.py
Best run once a day, after the daily candle has closed.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCHF=X", "USDCAD=X"]
HORIZON = 5          # days each trade is held (stats are measured this way)
COST = 0.0001        # ~1 pip round-trip cost assumed
MIN_TRADES = 30      # a strategy needs this many past signals to be judged
MIN_WIN = 0.55       # win rate needed in BOTH halves of history
LOG_FILE = "signals_log.csv"


# ---------- data and indicators ----------
def get_data(pair):
    raw = yf.download(pair, period="8y", interval="1d", progress=False)
    df = pd.DataFrame({
        "close": raw["Close"].squeeze(),
        "high": raw["High"].squeeze(),
        "low": raw["Low"].squeeze(),
    }).dropna()
    return df


def add_indicators(df):
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["sma100"] = c.rolling(100).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift()).abs(),
        (df["low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd"] = macd
    df["macd_sig"] = macd.ewm(span=9, adjust=False).mean()

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_up"] = mid + 2 * sd
    df["bb_low"] = mid - 2 * sd
    return df


# ---------- strategies: each returns +1 (buy), -1 (sell), 0 (nothing) ----------
def s_trend(df):
    s = pd.Series(0, index=df.index)
    s[(df["ema20"] > df["ema50"]) & (df["close"] > df["ema50"])] = 1
    s[(df["ema20"] < df["ema50"]) & (df["close"] < df["ema50"])] = -1
    return s


def s_macd(df):
    s = pd.Series(0, index=df.index)
    s[df["macd"] > df["macd_sig"]] = 1
    s[df["macd"] < df["macd_sig"]] = -1
    return s


def s_breakout(df):
    hh = df["high"].rolling(20).max().shift(1)
    ll = df["low"].rolling(20).min().shift(1)
    raw = pd.Series(np.nan, index=df.index)
    raw[df["close"] > hh] = 1
    raw[df["close"] < ll] = -1
    return raw.ffill(limit=2).fillna(0).astype(int)   # stays active 3 days


def s_rsi(df):
    s = pd.Series(0, index=df.index)
    s[df["rsi"] < 30] = 1
    s[df["rsi"] > 70] = -1
    return s


def s_bollinger(df):
    s = pd.Series(0, index=df.index)
    s[df["close"] < df["bb_low"]] = 1
    s[df["close"] > df["bb_up"]] = -1
    return s


def s_pullback(df):
    s = pd.Series(0, index=df.index)
    s[(df["close"] > df["sma100"]) & (df["rsi"] < 40)] = 1
    s[(df["close"] < df["sma100"]) & (df["rsi"] > 60)] = -1
    return s


STRATEGIES = {
    "Trend following (EMA 20/50)": s_trend,
    "MACD momentum": s_macd,
    "20-day breakout": s_breakout,
    "RSI reversal (30/70)": s_rsi,
    "Bollinger band reversal": s_bollinger,
    "Pullback in trend": s_pullback,
}


# ---------- historical evidence ----------
def evaluate(df, sig):
    """How did this strategy do each time it newly fired, holding HORIZON days?"""
    entries = (sig != 0) & (sig != sig.shift(1))
    fwd = df["close"].shift(-HORIZON) / df["close"] - 1
    pnl = (fwd * sig - COST)[entries].dropna()
    n = len(pnl)
    if n < MIN_TRADES:
        return {"n": n, "win": np.nan, "avg": np.nan, "ok": False}
    h = n // 2
    w1 = (pnl.iloc[:h] > 0).mean()
    w2 = (pnl.iloc[h:] > 0).mean()
    ok = (w1 >= MIN_WIN) and (w2 >= MIN_WIN) and (pnl.mean() > 0)
    return {"n": n, "win": (pnl > 0).mean(), "avg": pnl.mean(), "ok": bool(ok)}


# ---------- scan ----------
def scan(pair):
    df = add_indicators(get_data(pair))
    price, atr = df["close"].iloc[-1], df["atr"].iloc[-1]
    name = pair.replace("=X", "")
    print(f"\n=== {name}  |  last close {price:.4f}  |  date {df.index[-1].date()} ===")

    supported_votes = []
    log_rows = []
    for label, fn in STRATEGIES.items():
        sig = fn(df).fillna(0).astype(int)
        today = int(sig.iloc[-1])
        stats = evaluate(df, sig)
        if today == 0:
            continue
        side = "BUY" if today > 0 else "SELL"
        verdict = "SUPPORTED" if stats["ok"] else "unproven"
        if stats["n"] >= MIN_TRADES:
            ev = f"past win rate {stats['win']:.0%} over {stats['n']} signals, avg {stats['avg']*100:+.2f}%/trade"
        else:
            ev = f"only {stats['n']} past signals, too few to judge"
        print(f"  [{verdict:9}] {label}: {side}  ({ev})")
        if stats["ok"]:
            supported_votes.append(today)
        log_rows.append({"date": df.index[-1].date(), "pair": name, "strategy": label,
                         "side": side, "price": round(price, 5), "verdict": verdict})

    if not supported_votes:
        print("  -> Suggested move: NO TRADE (no strategy with a proven history is active)")
    else:
        net = sum(supported_votes)
        if net == 0:
            print("  -> Suggested move: NO TRADE (supported strategies disagree)")
        else:
            d = 1 if net > 0 else -1
            stop = price - d * 1.5 * atr
            print(f"  -> Suggested move: {'BUY' if d > 0 else 'SELL'} | "
                  f"stop-loss near {stop:.4f} (1.5x ATR) | plan to exit after ~{HORIZON} days")
    return log_rows


def main():
    all_rows = []
    for p in PAIRS:
        try:
            all_rows += scan(p)
        except Exception as e:
            print(f"\n{p}: failed ({e})")

    if all_rows:
        new = pd.DataFrame(all_rows)
        if os.path.exists(LOG_FILE):
            new = pd.concat([pd.read_csv(LOG_FILE), new]).drop_duplicates(
                subset=["date", "pair", "strategy"])
        new.to_csv(LOG_FILE, index=False)
        print(f"\nSignals logged to {LOG_FILE} (check them later to see how they really did).")

    print("\nNOTE: 'SUPPORTED' means it worked historically, not that it will work next.")
    print("With 6 strategies x 6 pairs tested, some pass by pure luck. Paper trade first.")


if __name__ == "__main__":
    main()
