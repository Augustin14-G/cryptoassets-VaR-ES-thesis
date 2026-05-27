# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Violation function
def compute_violations(returns, var_series):
    losses = -returns
    violations = pd.Series(np.nan, index=returns.index, dtype=float)

    valid = var_series.notna()
    violations.loc[valid] = (losses.loc[valid] > var_series.loc[valid]).astype(int)

    return violations


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

# Kupiec table
def kupiec_table(returns, asset_name):
    results = []
    n_oos = len(returns) - WINDOW
    oos_index = returns.index[WINDOW:]

    for alpha in alphas:
        var_p  = rolling_var(returns, alpha, "Parametric")
        var_h  = rolling_var(returns, alpha, "Historical")
        var_mc = rolling_var(returns, alpha, "Monte Carlo (Normal)")
        var_mc_t = rolling_var(returns, alpha, "Monte Carlo (Student-t)")

        v_p  = compute_violations(returns, var_p).loc[oos_index]
        v_h  = compute_violations(returns, var_h).loc[oos_index]
        v_mc = compute_violations(returns, var_mc).loc[oos_index]
        v_mc_t = compute_violations(returns, var_mc_t).loc[oos_index]

        conf_label = f"{int((1 - alpha) * 100)}%"

        for model, v, LR, pval in [
            ("Parametric",   v_p,  *kupiec_test(v_p,  alpha, N=n_oos)),
            ("Historical",   v_h,  *kupiec_test(v_h,  alpha, N=n_oos)),
            ("Monte Carlo (Normal)",  v_mc, *kupiec_test(v_mc, alpha, N=n_oos)),
            ("Monte Carlo (Student-t)", v_mc_t, *kupiec_test(v_mc_t, alpha, N=n_oos)),
        ]:
            results.append({
                "Asset":            asset_name,
                "Confidence Level": conf_label,
                "Model":            model,
                "LR Statistic":     round(LR, 4)   if not np.isnan(LR) else np.nan,
                "p-value":          round(pval, 4) if not np.isnan(pval) else np.nan,
                "Reject H0 at 5%":   "Yes" if not np.isnan(pval) and pval < 0.05 else "No"
            })

    return pd.DataFrame(results)


kupiec_results = pd.concat([
    kupiec_table(btc_returns, "Bitcoin"),
    kupiec_table(eth_returns, "Ethereum")
], ignore_index=True)

print(kupiec_results.to_string(index=False))
kupiec_results.to_csv("kupiec_results.csv", index=False)


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

#Independence table
def independence_table(returns, asset_name):
    results = []
    oos_index = returns.index[WINDOW:]

    for alpha in alphas:
        var_p  = rolling_var(returns, alpha, "Parametric")
        var_h  = rolling_var(returns, alpha, "Historical")
        var_mc = rolling_var(returns, alpha, "Monte Carlo (Normal)")
        var_mc_t = rolling_var(returns, alpha, "Monte Carlo (Student-t)")

        v_p  = compute_violations(returns, var_p).loc[oos_index]
        v_h  = compute_violations(returns, var_h).loc[oos_index]
        v_mc = compute_violations(returns, var_mc).loc[oos_index]
        v_mc_t = compute_violations(returns, var_mc_t).loc[oos_index]

        conf_label = f"{int((1 - alpha) * 100)}%"

        for model, v, LR, pval in [
            ("Parametric",  v_p,  *independence_test(v_p)),
            ("Historical",  v_h,  *independence_test(v_h)),
            ("Monte Carlo (Normal)", v_mc, *independence_test(v_mc)),
            ("Monte Carlo (Student-t)", v_mc_t, *independence_test(v_mc_t)),
        ]:
            results.append({
                "Asset": asset_name,
                "Confidence Level": conf_label,
                "Model": model,
                "LR Statistic": round(LR, 4) if not np.isnan(LR) else np.nan,
                "p-value": round(pval, 4) if not np.isnan(pval) else np.nan,
                "Reject H0 at 5%": "Yes" if not np.isnan(pval) and pval < 0.05 else "No"
            })

    return pd.DataFrame(results)

independence_results = pd.concat([
    independence_table(btc_returns, "Bitcoin"),
    independence_table(eth_returns, "Ethereum")
], ignore_index=True)

print(independence_results.to_string(index=False))
independence_results.to_csv("independence_results.csv", index=False)


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
joint_results.to_csv("joint_test_results.csv", index=False)
