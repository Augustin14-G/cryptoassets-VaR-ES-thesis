import yfinance as yf
import pandas as pd
import numpy as np

btc = yf.download("BTC-USD", start="2018-01-01", end="2024-12-31")
eth = yf.download("ETH-USD", start="2018-01-01", end="2024-12-31")

if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

if isinstance(eth.columns, pd.MultiIndex):
    eth.columns = eth.columns.get_level_values(0)

btc = btc[["Close"]]
eth = eth[["Close"]]

btc["Return"] = np.log(btc["Close"] / btc["Close"].shift(1))
eth["Return"] = np.log(eth["Close"] / eth["Close"].shift(1))

btc = btc.dropna()
eth = eth.dropna()

btc[["Return"]].to_csv("btc_returns_methodology.csv", index_label="Date")
eth[["Return"]].to_csv("eth_returns_methodology.csv", index_label="Date")
