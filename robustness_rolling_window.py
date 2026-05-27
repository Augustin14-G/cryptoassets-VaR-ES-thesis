import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.stats import chi2

# Load data 
btc = pd.read_csv("btc_returns_methodology.csv", parse_dates=["Date"], index_col="Date")
eth = pd.read_csv("eth_returns_methodology.csv", parse_dates=["Date"], index_col="Date")

# Extract returns
btc_returns = btc["Return"]
eth_returns = eth["Return"]

# Confidence levels 
alphas = [0.10, 0.05, 0.01]

# Rolling Window --> We put 125
WINDOW = 125

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
violations_df.to_csv("violations_results_125.csv", index=False)

# Kupiec test 
def kupiec_test(violations, alpha, N=None):
    x = int(violations.sum())

    if N is None:
        N = int(violations.notna().sum())

    p_hat = x / N

    if x == 0 or x == N:
        return np.nan, np.nan

    log_L0 = (N - x) * np.log(1 - alpha) + x * np.log(alpha)
    log_L1 = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)

    LR = -2 * (log_L0 - log_L1)

    p_value = 1 - chi2.cdf(LR, df=1)

    return LR, p_value

# Independence test 
def independence_test(violations):
    v = violations.dropna().astype(int)

    if len(v)<2:
        return np.nan, np.nan
    
    v_prev = v.shift(1).dropna().astype(int)
    v_curr = v.iloc[1:].astype(int)

    n00 = ((v_prev == 0) & (v_curr == 0)).sum()
    n01 = ((v_prev == 0) & (v_curr == 1)).sum()
    n10 = ((v_prev == 1) & (v_curr == 0)).sum()
    n11 = ((v_prev == 1) & (v_curr == 1)).sum()

    denom_0 = n00 + n01
    denom_1 = n10 + n11
    denom_all = n00 + n01 + n10 + n11

    if denom_0 == 0 or denom_1 == 0 or denom_all == 0 :
        return np.nan, np.nan
    
    p_hat = (n01 + n11) / denom_all
    p_hat0 = n01 / denom_0
    p_hat1 = n11 / denom_1

    if p_hat in [0, 1] or p_hat0 in [0, 1] or p_hat1 in [0, 1]:
        return np.nan, np.nan

    log_L0 = (n00 + n10) * np.log(1-p_hat) + (n01 + n11) * np.log(p_hat)
    log_L1 = n00 * np.log(1 - p_hat0) + n01 * np.log(p_hat0) \
           + n10 * np.log(1 - p_hat1) + n11 * np.log(p_hat1)
    
    LR = -2 * (log_L0 - log_L1)
    p_value = 1 - chi2.cdf(LR, df=1)

    return LR, p_value

# Joint test function
def joint_test(LR_pof, LR_ind):
    if np.isnan(LR_pof) or np.isnan(LR_ind):
        return np.nan, np.nan
    
    LR_cc = LR_pof + LR_ind
    p_value = 1 - chi2.cdf(LR_cc, df=2)

    return LR_cc, p_value

# Joint test table
def joint_test_table(returns, asset_name):
    results = []
    n_oos = len(returns) - WINDOW
    oos_index = returns.index[WINDOW:]

    for alpha in alphas:
        var_p  = rolling_var(returns, alpha, "Parametric")
        var_h  = rolling_var(returns, alpha, "Historical")
        var_mc = rolling_var(returns, alpha, "Monte Carlo (Normal)")
        var_mc_t = rolling_var(returns, alpha, "Monte Carlo (Student-t)")

        conf_label = f"{int((1 - alpha) * 100)}%"

        for model, var_series in [
            ("Parametric",  var_p),
            ("Historical",  var_h),
            ("Monte Carlo (Normal)", var_mc),
            ("Monte Carlo (Student-t)", var_mc_t),
        ]:
            v = compute_violations(returns, var_series).loc[oos_index]

            LR_pof, pval_pof = kupiec_test(v, alpha, N=n_oos)
            LR_ind, pval_ind = independence_test(v)
            LR_cc,  pval_cc  = joint_test(LR_pof, LR_ind)


            results.append({
                "Asset":            asset_name,
                "Confidence Level": conf_label,
                "Model":            model,
                "Kupiec LR":        round(LR_pof,  4) if not np.isnan(LR_pof)  else np.nan,
                "Kupiec p-value":   round(pval_pof, 4) if not np.isnan(pval_pof) else np.nan,
                "Reject Kupiec H0":    "Yes" if not np.isnan(pval_pof) and pval_pof < 0.05 else "No",
                "Ind. LR":       round(LR_ind,  4) if not np.isnan(LR_ind)  else np.nan,
                "Ind. p-value":  round(pval_ind, 4) if not np.isnan(pval_ind) else np.nan,
                "Reject Ind. H0":   "Yes" if not np.isnan(pval_ind) and pval_ind < 0.05 else "No",
                "Joint LR":            round(LR_cc,   4) if not np.isnan(LR_cc)   else np.nan,
                "Joint p-value":       round(pval_cc,  4) if not np.isnan(pval_cc)  else np.nan,
                "Reject Joint H0":        "Yes" if not np.isnan(pval_cc) and pval_cc < 0.05 else "No",
            })

    return pd.DataFrame(results)


joint_results = pd.concat([
    joint_test_table(btc_returns, "Bitcoin"),
    joint_test_table(eth_returns, "Ethereum")
], ignore_index=True)

print(joint_results.to_string(index=False))
joint_results.to_csv("joint_test_results_125.csv", index=False)


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
es_backtesting_results.to_csv("es_backtesting_results_125.csv", index=False)

    

