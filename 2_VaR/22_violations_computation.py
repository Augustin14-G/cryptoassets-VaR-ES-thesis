# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# Load data 
btc = pd.read_csv("btc_returns_methodology.csv", parse_dates=["Date"], index_col="Date")
eth = pd.read_csv("eth_returns_methodology.csv", parse_dates=["Date"], index_col="Date")

# Extract returns
btc_returns = btc["Return"]
eth_returns = eth["Return"]

# Confidence levels 
alphas = [0.10, 0.05, 0.01]

# Rolling Window
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
    

# Violation functions
def compute_violations(returns, var_series) :
    losses = -returns
    valid = var_series.notna()
    violations = pd.Series(False, index=returns.index)
    violations.loc[valid] = losses.loc[valid] > var_series.loc[valid]
    return violations

def count_violations(returns, var_value):
    return compute_violations(returns, var_value).sum()

# Violation table
results = []
def violation_table(returns, asset_name, window=WINDOW):
    results = []
    n_oos = len(returns) - window

    for alpha in alphas : 
        var_p = rolling_var(returns, alpha, "Parametric", window)
        var_h = rolling_var(returns, alpha, "Historical", window)
        var_mc = rolling_var(returns, alpha, "Monte Carlo (Normal)", window)
        var_mc_t = rolling_var(returns, alpha, "Monte Carlo (Student-t)", window)

        results.append({
        "Asset": asset_name,
        "Confidence Level": f"{int((1 - alpha) * 100)}%",
        "Out-of-Sample Obs." : n_oos,
        "Expected Violations": round(alpha * n_oos, 0),
        "Parametric violations": count_violations(returns, var_p),
        "Historical Violations": count_violations(returns, var_h),
        "Monte Carlo (Normal) Violations": count_violations(returns, var_mc),
        "Monte Carlo (Student-t) Violations": count_violations(returns, var_mc_t)
        })

    return results

all_results = (
violation_table(btc_returns, "Bitcoin") + 
violation_table(eth_returns, "Ethereum")
)

violations_df = pd.DataFrame(all_results)
print(violations_df)
violations_df.to_csv("violations_results.csv", index=False)


# Plot Graph of Rolling VaR with violations
def plot_multi_rolling_var(returns, asset_name, method="Parametric", window=WINDOW):
    losses = -returns

    linestyles = {
        0.10: "--",
        0.05: "-.",
        0.01: "-"
    }

    plt.figure(figsize=(12, 6))

    losses_oos = losses.iloc[window:]

    plt.plot(
        losses_oos.index,
        losses_oos,
        color="black",
        linewidth = 0.7,
        alpha = 0.6
    )

    colors = {
        0.10: "0.6",
        0.05: "0.4",
        0.01: "0.0"
    }

    for alpha in alphas:
        var_series = rolling_var(returns, alpha, method, window)

        plt.plot(
            var_series.index,
            var_series,
            linestyle=linestyles[alpha],
            linewidth=1.5,
            color= colors[alpha],
            label=f"VaR {int((1 - alpha) * 100)}%"
        )

    plt.title(f"{asset_name} — {method} Rolling VaR", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=12, fontweight="bold")
    plt.ylabel("Realized loss (-returns)", fontsize=14, fontweight="bold")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend(fontsize=10, framealpha=1, edgecolor="black", fancybox=False)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    safe_asset = asset_name.lower().replace(" ", "_")
    safe_method = method.lower().replace(" ", "_")
    plt.savefig(f"{safe_asset}_{safe_method}_rolling_var_bw.png", dpi=300)

    plt.show()

for asset_name, returns in [("Bitcoin", btc_returns), ("Ethereum", eth_returns)]:
    for method in ["Parametric", "Historical", "Monte Carlo (Normal)", "Monte Carlo (Student-t)"]:
        plot_multi_rolling_var(returns, asset_name, method)





    
