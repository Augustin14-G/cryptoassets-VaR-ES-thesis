# Import packages
import numpy as np
import pandas as pd
from scipy.stats import norm, t

# Load data 
btc = pd.read_csv("btc_returns_methodology.csv", index_col=0)
eth = pd.read_csv("eth_returns_methodology.csv", index_col=0)

# Extract returns
btc_returns = btc["Return"]
eth_returns = eth["Return"]


# Confidence levels 
alphas = [0.10, 0.05, 0.01]


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
def var_monte_carlo(returns, alpha, n_sim=100000, dt=1, seed=42):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal (n_sim)
    simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    return np.quantile(simulated_losses, 1-alpha)

# Estimate nu from data — just to report, not used for simulation
btc_nu = t.fit(btc_returns)[0]
eth_nu = t.fit(eth_returns)[0]
print(f"Estimated nu for Bitcoin: {btc_nu:.2f}")
print(f"Estimated nu for Ethereum: {eth_nu:.2f}")

# degree of freedom for student t
NU = 4

# Monte Carlo t VaR function
def var_monte_carlo_t(returns, alpha, nu=NU, n_sim = 100000, dt=1, seed=42):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=nu, size=n_sim) #shocks
    z = z / np.sqrt(nu / (nu-2)) # Scale to unit variance
    simulated_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    simulated_losses = -simulated_returns
    return np.quantile(simulated_losses, 1-alpha)


# Computation table
def compute_var(returns, asset_name): 
    results = []

    for alpha in alphas:
        results.append({
            "Asset": asset_name,
            "Confidence Level": f"{int((1 - alpha) * 100)}%",
            "Parametric VaR": var_parametric(returns, alpha),
            "Historical VaR": var_historical(returns, alpha),
            "Monte Carlo VaR (Normal)": var_monte_carlo(returns, alpha),
            "Monte Carlo VaR (Student-t)": var_monte_carlo_t(returns, alpha)
        })

    return pd.DataFrame(results)

btc_var = compute_var(btc_returns, "Bitcoin")
eth_var = compute_var(eth_returns, "Ethereum")
var_results = pd.concat([btc_var, eth_var], ignore_index=True)

# Convert results to % and round
cols = ["Parametric VaR", "Historical VaR", "Monte Carlo VaR (Normal)", "Monte Carlo VaR (Student-t)"]
var_results[cols] = (var_results[cols] * 100).round(3)


print(var_results.to_string(index=False))
var_results.to_csv("var_results.csv", index=False)
    
