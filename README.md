# cryptoassets-VaR-ES-thesis
This repository contains the Python code for the master's thesis:
"A Comparative Evaluation of Parametric, Historical, and Monte Carlo 
Value-at-Risk Models and Expected Shortfall in Measuring the Market 
Risk of Cryptoassets"

## Structure
1_data
- 1_stats_data.py - Descriptive statistics of cryptoassets returns
- 2_returns_hist.py - Histograms of returns
- 3_data_prep_methodology.py - Data preparation for the main analysis

2_VaR
- 1_VaR_computation.py - Computation of full-sample VaR estimates
- 2_violations_computation.py - Rolling VaR estimation, violation analysis and graphs
- 3_backtesting_VaR.py - Kupiec, Christoffersen independence and Joint tests 

3_ES
- 1_ES_computation.py - Computation of full-sample ES estimates
- 2_backtesting_ES.py - Acerbi-Szekely ES backtesting procedure and graphs

4_robustness
- robustness_rolling_window.py - robustness check using a 125-day rolling window

## Requirements
python 3.14.0 with the following packages : 
- yfinance
- numpy
- pandas
- matplotlib
- scipy
  
## Data
Daily Bitcoin (BTC-USD) and Ethereum (ETH-USD) price data were downloaded from Yahoo Finance using the yfinance library, covering the period from January 2018 to December 2024. 
