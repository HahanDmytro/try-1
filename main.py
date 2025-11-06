from flask import Flask, jsonify
import threading
from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import os

app = Flask(__name__)

@app.route("/")
def status():
    return jsonify(state)

# === CONFIG ===
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "XRPUSDT", "DOGEUSDT"]
INTERVAL = 60        # 1h candles
LIMIT = 720          # number of candles
FEE = 0.001          # 0.1%
START_BALANCE = 100.0
LOG_FILE = "multi_trades.json"

# === Initialize API ===
session = HTTP(testnet=False)

# === Prepare log file ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# === Function to append a trade to log ===
def save_trade(symbol, trade_type, price, balance):
    with open(LOG_FILE, "r") as f:
        trades = json.load(f)

    trades.append({
        "symbol": symbol,
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "type": trade_type,
        "price": round(price, 8),
        "balance": round(balance, 8)
    })

    with open(LOG_FILE, "w") as f:
        json.dump(trades, f, indent=4)

# === Function to fetch data for one symbol ===
def fetch_data(symbol):
    resp = session.get_mark_price_kline(
        category="linear",
        symbol=symbol,
        interval=INTERVAL,
        limit=LIMIT
    )
    graphic = resp["result"]["list"]
    df = pd.DataFrame(graphic, columns=["timestamp","open","high","low","close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# === Indicator calc ===
def add_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df.dropna().reset_index(drop=True)

# === State for each symbol ===
state = {}
for sym in SYMBOLS:
    state[sym] = {
        "balance": START_BALANCE,
        "position": 0.0,
        "last_signal": None
    }

# === Simulate all coins ===
def simulate_all():
    for sym in SYMBOLS:
        df = fetch_data(sym)
        df = add_indicators(df)
        st = state[sym]
        bal = st["balance"]
        pos = st["position"]

        if len(df) < 2:
            continue

        macd, signal, rsi, close, ema50 = df.iloc[-1][["MACD","Signal","RSI","close","EMA50"]]
        prev_macd, prev_signal = df.iloc[-2][["MACD","Signal"]]
        ts = df.iloc[-1]["timestamp"]

        # === BUY ===
        if prev_macd < prev_signal and macd > signal and rsi < 55 and close > ema50 and bal > 0:
            pos = (bal * (1 - FEE)) / close
            st["balance"] = 0.0
            st["position"] = pos
            st["last_signal"] = "BUY"
            save_trade(sym, "BUY", close, pos * close)
            print(f"{sym} | ✅ BUY at {close:.5f}")

        # === SELL ===
        elif prev_macd > prev_signal and macd < signal and rsi > 45 and close < ema50 and pos > 0:
            new_bal = (pos * close) * (1 - FEE)
            st["balance"] = new_bal
            st["position"] = 0.0
            st["last_signal"] = "SELL"
            save_trade(sym, "SELL", close, new_bal)
            print(f"{sym} | ❌ SELL at {close:.5f}")

        # === Display formatted output ===
        current_value = st["balance"] if st["position"] == 0 else st["position"] * close
        print("-" * 80)
        print(f"💰 {datetime.now()} | Symbol: {sym} | Value: ${current_value:.2f} | Price: {close:.5f} | RSI: {rsi:.2f}")
        print("-" * 80)

        # Update state
        state[sym] = st

def run_trading_loop():
    while True:
        try:
            simulate_all()
        except Exception as e:
            print("Error:", e)
        time.sleep(3600)

if __name__ == "__main__":
    # Start trading loop in a background thread
    t = threading.Thread(target=run_trading_loop, daemon=True)
    t.start()

    # Start Flask server
    app.run(host="0.0.0.0", port=10000)
