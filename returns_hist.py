# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load Data
btc = pd.read_csv("BTC_returns.csv", index_col=0)
eth = pd.read_csv("ETH_returns.csv", index_col=0)

# Bitcoin Histogram + Normal
plt.figure()

plt.hist(btc["Return"], bins=50, density =True, color="lightgray", edgecolor="Black")

mu_btc= btc["Return"].mean()
sigma_btc = btc["Return"].std()

x = np.linspace(btc["Return"].min(), btc["Return"].max(), 1000)

plt.plot(x, norm.pdf(x, mu_btc, sigma_btc), color= "Black", linewidth=2)

plt.title("Bitcoin Returns Distribution vs Normal Distribution")
plt.xlabel("Returns")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("btc_histogram.png")
plt.show()

# Ethereum Histogram + Normal
plt.figure()

plt.hist(eth["Return"], bins=50, density=True, color="lightgray", edgecolor="Black")

mu_eth = eth["Return"].mean()
sigma_eth = eth["Return"].std()

x = np.linspace(eth["Return"].min(), eth["Return"].max(), 1000)

plt.plot(x, norm.pdf(x, mu_eth, sigma_eth), color= "Black", linewidth=2)

plt.title("Ethereum Returns Distribution vs Normal Distribution")
plt.xlabel("Returns")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("eth_histogram.png")
plt.show()