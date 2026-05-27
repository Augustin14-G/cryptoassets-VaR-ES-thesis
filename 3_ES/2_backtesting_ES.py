# Import packages
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load data
btc = pd.read_csv("btc_returns_methodology.csv", parse_dates=["Date"], index_col="Date")
eth = pd.read_csv("eth_returns_methodology.csv", parse_dates=["Date"], index_col="Date")

# Extract returns
btc_returns = btc["Return"]
eth_returns = eth["Return"]

# Parameters
alphas = [0.1, 0.05, 0.01]
WINDOW = 250

# Parametric VaR function
def var_parametric(returns, alpha):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1-alpha)
    return -mu + sigma * z

# Historical VaR function
def var_historical(returns, alpha):
    losses = -returns
    return np.quantile(losses, 1 - alpha)

# Monte Carlo VaR function
def var_monte_carlo(returns, alpha, n_sim=10000, dt=1, seed=42):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal (n_sim)
    simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    return np.quantile(simulated_losses, 1-alpha)

# Monte Carlo t VaR function
def var_monte_carlo_t(returns, alpha, nu=4, n_sim = 10000, dt=1, seed=42):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=nu, size=n_sim) #shocks
    z = z / np.sqrt(nu / (nu-2)) # Scale to unit variance
    simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    return np.quantile(simulated_losses, 1-alpha)


# Parametric ES function
def es_parametric(returns, alpha):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - alpha)
    return -mu + sigma * (norm.pdf(z) / alpha)

# Historical ES function
def es_historical(returns, alpha):
    losses = -returns
    var_alpha = np.quantile(losses, 1-alpha)
    tail_losses = losses[losses > var_alpha]

    if len(tail_losses) == 0:
        return np.nan
    
    return tail_losses.mean()

# Monte Carlo ES function
def es_monte_carlo(returns, alpha, n_sim=10000, dt=1, seed=42):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)
    simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    var_alpha = np.quantile(simulated_losses, 1 - alpha)
    tail_losses = simulated_losses[simulated_losses > var_alpha]

    if len(tail_losses) == 0:
        return np.nan
    
    return tail_losses.mean()

# Monte Carlo t ES function
def es_monte_carlo_t(returns, alpha, nu=4, n_sim=10000, dt=1, seed=42):
    mu=returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=nu, size=n_sim)
    z = z / np.sqrt(nu / (nu-2))  # Scale to unit variance
    simulated_returns = (mu-0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    var_alpha = np.quantile(simulated_losses, 1-alpha)
    tail_losses = simulated_losses[simulated_losses > var_alpha]

    if len(tail_losses) == 0:
        return np.nan
    
    return tail_losses.mean()


# Rolling VaR
def rolling_var(returns, alpha, method, window=WINDOW, base_seed=42):
    n = len(returns)
    var_series = pd.Series(np.nan, index=returns.index, dtype=float)

    for t in range(window, n):
        window_returns=returns.iloc[t-window:t]

        if method == "Parametric":
            var_series.iloc[t] = var_parametric(window_returns, alpha)
        elif method == "Historical":
            var_series.iloc[t] = var_historical(window_returns, alpha)
        elif method == "Monte Carlo (Normal)":
            var_series.iloc[t] = var_monte_carlo(window_returns, alpha, seed= base_seed + t)
        elif method == "Monte Carlo (Student-t)":
            var_series.iloc[t] = var_monte_carlo_t(window_returns, alpha, nu=4, seed=base_seed +t)
        else : raise ValueError("Method error")
    
    return var_series

# Rolling ES
def rolling_es(returns, alpha, method, window = WINDOW, base_seed = 42):
    n = len(returns)
    es_series = pd.Series(np.nan, index = returns.index, dtype = float)

    for t in range(window, n):
        window_returns = returns.iloc[t - window:t]

        if method == "Parametric":
            es_series.iloc[t] = es_parametric(window_returns, alpha)
        elif method == "Historical":
            es_series.iloc[t] = es_historical(window_returns, alpha)
        elif method == "Monte Carlo (Normal)":
            es_series.iloc[t] = es_monte_carlo(window_returns, alpha, seed= base_seed + t)
        elif method == "Monte Carlo (Student-t)":
            es_series.iloc[t] = es_monte_carlo_t(window_returns, alpha, nu=4, seed=base_seed +t)
        else : raise ValueError("Method error")

    return es_series


# ES Backtesting 
def es_backtest(returns, var_series, es_series, alpha): 
    losses = -returns
    valid = var_series.notna() & es_series.notna()

    losses = losses.loc[valid]
    var_series = var_series.loc[valid]
    es_series = es_series.loc[valid]

    N = len(losses)
    exceed = losses > var_series
    if exceed.sum() == 0:
        return np.nan, 0, round(N * alpha, 0)
    
    z_stat = (1 / (len(losses) * alpha)) * np.sum((losses[exceed] / es_series[exceed])) - 1

    return z_stat, int(exceed.sum()), round(N * alpha, 0)


# ES backtesting table
def es_backtesting_table(returns, asset_name):
    results = []

    for alpha in alphas:
        conf_label = f"{int((1 - alpha) * 100)}%"

        for method in ["Parametric", "Historical", "Monte Carlo (Normal)", "Monte Carlo (Student-t)"]:
            var_series = rolling_var(returns, alpha, method)
            es_series = rolling_es(returns, alpha, method)

            z_stat, n_exceed, n_expected = es_backtest(returns, var_series, es_series, alpha)
            

            results.append({
                "Asset": asset_name,
                "Confidence Level": conf_label,
                "Model": method,
                "ES Test Statistic": round(z_stat, 4) if not pd.isna(z_stat) else np.nan,
                "Expected Tail Exceedances": n_expected,
                "Observed Tail Exceedances": n_exceed,
                "Reject ES H0 at 5%": "Yes" if not pd.isna(z_stat) and z_stat > 0.7 else "No"
            })
    return pd.DataFrame(results)


es_backtesting_results = pd.concat([
    es_backtesting_table(btc_returns, "Bitcoin"),
    es_backtesting_table(eth_returns, "Ethereum")
], ignore_index=True)

print(es_backtesting_results.to_string(index=False))
es_backtesting_results.to_csv("es_backtesting_results.csv", index=False)


# ES backtesting graphs      
def plot_es_backtesting(returns, asset_name, method="Parametric", alpha=0.01, window=WINDOW):
    losses = -returns
    var_series = rolling_var(returns, alpha, method, window)
    es_series = rolling_es(returns, alpha, method, window)
    oos_start = returns.index[window]

    valid = var_series.notna()
    exceed = losses[valid] > var_series[valid]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(losses.index, losses, color="black", linewidth=0.7, alpha=0.4, label="_nolegend_")
    ax.plot(var_series.index, var_series, color="black", linestyle="--", linewidth=1.5, label= f"VaR {int((1 - alpha) * 100)}%")
    ax.plot(es_series.index, es_series, color="grey", linestyle="-", linewidth=1.5, label= f"ES {int((1 - alpha) * 100)}%")

 
    ax.set_xlim(left=oos_start)
    ax.set_title(f"{asset_name} — {method} Rolling VaR and ES "
                 f"({int((1 - alpha) * 100)}%, W={window})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Realized loss (-returns)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, framealpha=1, edgecolor="black", fancybox=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    safe_asset  = asset_name.lower().replace(" ", "_")
    safe_method = method.lower().replace(" ", "_")
    filename = f"{safe_asset}_{safe_method}_var_es_{int((1 - alpha) * 100)}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

for asset_name, returns in [("Bitcoin", btc_returns), ("Ethereum", eth_returns)]:
    for method in ["Parametric", "Historical", "Monte Carlo (Normal)", "Monte Carlo (Student-t)"]:
        plot_es_backtesting(returns, asset_name, method, alpha=0.01)
    



    
