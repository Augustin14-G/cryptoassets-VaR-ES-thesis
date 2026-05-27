# Import packages
import pandas as pd
import numpy as np
from scipy.stats import norm, t

# Load data
btc = pd.read_csv("btc_returns_methodology.csv", parse_dates=["Date"], index_col="Date")
eth = pd.read_csv("eth_returns_methodology.csv", parse_dates=["Date"], index_col="Date")

# Extract returns
btc_returns = btc["Return"]
eth_returns = eth["Return"]

# Parameters
alphas = [0.1, 0.05, 0.01]

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
def es_monte_carlo(returns, alpha, n_sim=100000, dt=1, seed=42):
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
def es_monte_carlo_t(returns, alpha, nu=4, n_sim=100000, dt=1, seed=42):
    mu=returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=nu, size=n_sim)
    z = z / np.sqrt(nu / (nu-2))  # Scale to unit variance
    simulated_returns = (mu-0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    simumated_losses = -simulated_returns
    var_alpha = np.quantile(simumated_losses, 1-alpha)
    tail_losses = simumated_losses[simumated_losses > var_alpha]

    if len(tail_losses) == 0:
        return np.nan
    
    return tail_losses.mean()

# Computation table
def compute_es_table(returns, asset_name):
    results = []

    for alpha in alphas:
        results.append({
            "Asset": asset_name,
            "Confidence Level": f"{int((1 - alpha) * 100)}%",
            "Parametric ES": es_parametric(returns, alpha),
            "Historical ES": es_historical(returns, alpha),
            "Monte Carlo ES (Normal)": es_monte_carlo(returns, alpha),
            "Monte Carlo ES (Student-t)": es_monte_carlo_t(returns, alpha, nu=4)
        })

    return pd.DataFrame(results)
    

es_results = pd.concat([
    compute_es_table(btc_returns, "Bitcoin"),
    compute_es_table(eth_returns, "Ethereum")], ignore_index=True)
    
# Convert results to % and round
cols = ["Parametric ES", "Historical ES", "Monte Carlo ES (Normal)", "Monte Carlo ES (Student-t)"]
es_results[cols] = (es_results[cols] * 100).round(3)


print(es_results.to_string(index=False))
es_results.to_csv("es_results.csv", index=False)
    
    
    

    





