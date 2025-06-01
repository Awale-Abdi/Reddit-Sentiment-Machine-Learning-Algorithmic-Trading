<h1 align="center">Algorithmic Trading with Reddit Sentiment & Machine Learning: A Python Strategy Framework</h1>

## 📊 Overview

This project develops a full-cycle algorithmic trading strategy combining **social media sentiment analysis**, **machine learning (XGBoost & Decision Tree Classifiers)**, and **quantitative backtesting** using Pfizer Inc. (PFE) stock as a case study. The framework simulates how hedge funds and fintechs could use Reddit sentiment data and predictive modeling to make informed long/short decisions in equity markets.

Originally completed as part of the *Algorithmic Trading in Python* course taught by Professor Michael Rolleigh, this proposal-based project was independently built in VS Code and includes all relevant model code, backtests, and visualizations.

## 🧠 Methodology

- **Sentiment Analysis**: Scraped 1,000 Reddit posts from `r/wallstreetbets` using the Reddit API. Extracted sentiment polarity via `TextBlob`, aggregated into a daily sentiment index.
- **Feature Engineering**: Created 25+ features including moving averages, RSI, MACD, ROC, Bollinger Bands, lagged returns, and Reddit sentiment scores.
- **Models Tested**: 
  - Decision Tree Classifier (with hyperparameter tuning)
  - XGBoost Classifier (with grid search and feature selection)
- **Trading Strategy Logic**:
  - Go long on positive prediction, hold cash on negative.
  - Monthly rebalancing logic built with custom `bt` framework strategy class.

## 🔁 Backtesting

- **Benchmark Strategy**: Buy and Hold on PFE stock.
- **Backtest Engine**: `bt` Python library; 1,000 simulation loops per strategy.
- **Performance Metrics**:
  - Total Return
  - CAGR
  - Max Drawdown
  - Sharpe, Sortino, Calmar Ratios

### 📈 Backtest Summary

| Strategy         | Total Return | CAGR  | Max Drawdown | Sharpe Ratio |
|------------------|--------------|-------|---------------|---------------|
| Buy & Hold (PFE) | -26.84%      | -11%  | 57%           | -0.31         |
| XGBoost Model    | +3.18%       | 1.19% | 24%           | 0.16          |

Even with **limited Reddit data and no paid tools**, the ML model **outperformed** the benchmark on several metrics across multiple loops.

## 📁 Project Structure

- `Datasets/`  
  - `PFE_stock_data.csv` – Historical Pfizer stock data (downloaded via Yahoo Finance)  
  - `reddit_sentiment_daily.csv` – Aggregated daily sentiment scores from Reddit posts (via API)
 
- `Outputs/`  
  - `algo_trading_proposal.pdf` – 1,500-word write-up with results, visuals, and references  
  - `final_script_tree.py` – Full pipeline using Decision Tree Classifier  
  - `final_script_xgboost.py` – Optimized pipeline using XGBoost  
  - `writeup_visualizations.py` – Python code to generate financial performance charts and histograms  


## 🧩 Challenges & Limitations

- **Reddit API Access**: Free tier limited to 1,000 posts; no access to Twitter sentiment data.
- **Hardware Constraints**: Model runs limited to 1,000 loops vs. millions in institutional settings.
- **Simplified NLP**: Used `TextBlob` instead of custom financial sentiment classifiers.
- **Binary Classification**: Only predicted Up/Down instead of % change or multi-class.

> 🔍 *"Despite model accuracy hovering around 51%, the project proves that with more data, compute power, and financial domain expertise, this framework could be refined into a deployable quant trading system."*

## 🧠 Tools & Libraries

- `Reddit API`, `TextBlob` – Sentiment scraping and NLP
- `yfinance` – Historical stock data
- `pandas`, `ta`, `matplotlib` – Data analysis, indicators, and visualization
- `scikit-learn`, `xgboost` – ML modeling
- `bt` – Strategy definition and backtesting

## 🎓 Academic Context

Developed as part of the *Algorithmic Trading in Python* course (2024) at Hult International Business School with Prof. Michael Rolleigh. The goal was to simulate a real-world proposal that could be scaled by a hedge fund or fintech.

## 🔗 References

- [Tran et al., 2024 – ML in Vietnamese Stock Market](https://doi.org/10.1057/s41599-024-02807-x)
- [HAARMK Infotech – ML Stock Forecasting on Medium](https://medium.com/@haarmkinfotech/introduction-bc6ecaf22f8b)
- [Reddit API Updates](https://www.redditinc.com/blog/apifacts)

---

### 📧 Contact

For questions or collaboration:

- Email: **awaleiabdi@outlook.com**
- LinkedIn: [Awale Abdi](https://www.linkedin.com/in/awale-abdi/)
