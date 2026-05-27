# Import packages
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera

# Download data
btc = yf.download("BTC-USD", start="2018-01-01", end="2024-12-31")
eth = yf.download("ETH-USD", start="2018-01-01", end="2024-12-31")

# Keep only Close prices
btc=btc[["Close"]]
eth=eth[["Close"]]

# Compute log returns
btc["Return"]=np.log(btc["Close"]/btc["Close"].shift(1))
eth["Return"]=np.log(eth["Close"]/eth["Close"].shift(1))

# Remove missing values
btc = btc.dropna()
eth = eth.dropna()

# Function for descriptive statistics
def get_stats(series):
    jb_stat, jb_pvalue = jarque_bera(series)
    return { 
        "Mean": series.mean(),
        "Std Dev": series.std(),
        "Skewness":skew(series),
        "Kurtosis": kurtosis(series, fisher=False), # normal = 3
        "JB Statistic": jb_stat,
        "JB p-value" : jb_pvalue
    }

# Compute statistics
btc_stats = get_stats(btc["Return"])
eth_stats = get_stats(eth["Return"])

 # table
stats_df = pd.DataFrame([btc_stats, eth_stats], index=["Bitcoin", "Ethereum"])
stats_df_clean = stats_df.copy()

stats_df_clean["Mean"] = stats_df_clean["Mean"].round(6)
stats_df_clean["Std Dev"] = stats_df_clean["Std Dev"].round(4)
stats_df_clean["Skewness"] = stats_df_clean["Skewness"].round(3)
stats_df_clean["Kurtosis"] = stats_df_clean["Kurtosis"].round(2)
stats_df_clean["JB Statistic"] = stats_df_clean["JB Statistic"].round(2)
stats_df_clean["JB p-value"] = stats_df_clean["JB p-value"].apply(lambda x: f"{x:.2e}")

# results
pd.set_option('display.max_columns', None)
print(stats_df_clean)

# Save results
stats_df_clean.to_csv("descriptive_stats.csv")
btc.to_csv("BTC_returns.csv")
eth.to_csv("ETH_returns.csv")

