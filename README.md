<h1 align="center">Algorithmic Trading with Reddit Sentiment & Machine Learning: A Python Strategy Framework</h1>

## 📊 Overview

This project develops a full-cycle algorithmic trading strategy combining **Reddit sentiment analysis**, **machine learning (Decision Tree & XGBoost classifiers)**, and **quantitative backtesting** using Pfizer Inc. (PFE) stock data. It simulates how hedge funds and fintech firms might use alternative data and predictive modeling to make long/short trade decisions.

The framework was built independently in **VS Code** as part of the *Algorithmic Trading in Python* course at Hult International Business School. It earned a **score of 90/100**, placing in the **top 30% of the class**, and led to a **recommendation letter offer** from Professor Michael Rolleigh based on the technical quality and clarity of the work.

## 🧠 Methodology

- **Sentiment Analysis**: Extracted ~1,000 Reddit posts mentioning "Pfizer" from `r/wallstreetbets`. Used `TextBlob` to generate sentiment scores, then aggregated by day.
- **Feature Engineering**: Created 25+ features including moving averages, RSI, MACD, ROC, Bollinger Bands, lagged returns, and sentiment scores.
- **Models Used**:  
  - `DecisionTreeClassifier` with grid search tuning  
  - `XGBoostClassifier` with optimized hyperparameters and feature selection
- **Strategy Logic**:  
  - Go long on positive prediction; hold cash on negative  
  - Monthly rebalancing using custom `bt` framework strategy

## 🔁 Backtesting

- **Benchmark Strategy**: Buy-and-hold for PFE
- **Backtest Engine**: `bt` Python library; 1,000 simulation loops
- **Performance Metrics**: Total Return, CAGR, Drawdowns, Sharpe, Sortino, Calmar Ratios

### 📈 Sample Results Summary

| Strategy         | Total Return | CAGR  | Max Drawdown | Sharpe Ratio |
|------------------|--------------|-------|---------------|---------------|
| Buy & Hold (PFE) | -26.84%      | -11%  | 57%           | -0.31         |
| XGBoost Model    | +3.18%       | 1.19% | 24%           | 0.16          |

Even with limited Reddit data and no paid tools, the ML model **outperformed** the benchmark in several key backtest loops, showing promising proof-of-concept performance.

## 📁 Project Structure

- `Datasets/`  
  - `PFE_stock_data.csv` – Historical Pfizer stock data (via Yahoo Finance)  
  - `reddit_sentiment_daily.csv` – Aggregated daily sentiment scores (via Reddit API)

- `Outputs/`  
  - `algo_trading_proposal.pdf` – 1,500-word write-up with results, visuals, and references  
  - `final_script_tree.py` – Full pipeline using Decision Tree Classifier  
  - `final_script_xgboost.py` – Optimized pipeline using XGBoost  
  - `writeup_visualizations.py` – Python code to generate performance charts and histograms  

## 🧩 Challenges & Limitations

- **API Constraints**: Reddit API limited to 1,000 posts; no access to Twitter/X data
- **Hardware Limits**: Simulations limited to 1,000 backtests vs. millions in industry
- **Simple NLP**: Used `TextBlob` instead of custom sentiment classifiers
- **Binary Output**: Predictions limited to Up/Down (classification); no regression or % change

> *Still, even with minimal data and basic modeling, the strategy often beat a buy-and-hold benchmark — validating this framework as a potential foundation for more scalable, real-world systems.*

## 🧠 Tools & Libraries

- `TextBlob`, `requests`, `Reddit API` – Sentiment data scraping
- `pandas`, `ta`, `matplotlib` – Data analysis and feature creation
- `yfinance` – Historical stock data
- `scikit-learn`, `xgboost` – ML modeling
- `bt` – Custom strategy logic and portfolio backtesting

## 🏆 Academic Recognition

- Scored **90/100** on this project in the *Algorithmic Trading in Python* course at Hult International Business School
- Ranked in the **top 30%** of the class (grade: A)
- Personally offered a **recommendation letter** by Professor Michael Rolleigh in recognition of the project's technical quality, end-to-end execution, and clear documentation

## 🔗 References

- Tran, P. et al. (2024). [ML in Vietnamese Stock Markets](https://doi.org/10.1057/s41599-024-02807-x)  
- HAARMK Infotech. (2023). [ML for Stock Forecasting on Medium](https://medium.com/@haarmkinfotech/introduction-bc6ecaf22f8b)  
- [Reddit API Documentation](https://www.redditinc.com/blog/apifacts)  
- [Meta Graph API](https://developers.facebook.com)  
- [Twitter API (X)](https://developer.x.com/en/docs/twitter-api/getting-started/about-twitter-api)

---

### 📧 Contact

For questions or collaborations:

- Email: **awaleiabdi@outlook.com**  
- LinkedIn: [Awale Abdi](https://www.linkedin.com/in/awale-abdi/)
