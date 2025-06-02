<h1 align="center">Algorithmic Trading with Reddit Sentiment & Machine Learning: A Python Strategy Framework</h1>

## 📊 Overview

This project proposes an algorithmic trading framework combining **Reddit sentiment analysis**, **machine learning (Decision Tree & XGBoost classifiers)**, and **quantitative backtesting**, applied to Pfizer Inc. (PFE) stock data.

Built independently in **VS Code** as part of the *Algorithmic Trading in Python* course at Hult International Business School, this proof-of-concept earned the highest possible grade, placing in the **top 25%** of the class. It also earned the author a **recommendation letter offer** from Professor Michael Rolleigh for technical clarity and originality.

> ⚠️ **Disclaimer:** This model is not profitable or production-ready. It is an academic demonstration built with limited data and compute power. The goal is to show how a better-resourced team could develop and scale such a system.

## 🧠 Methodology

- **Sentiment Analysis**: Scraped ~1,000 posts from `r/wallstreetbets` mentioning Pfizer using the Reddit API. Used `TextBlob` for polarity scoring, aggregated daily.
- **Feature Engineering**: Created 25+ features (e.g., MA, RSI, ROC, MACD, Bollinger Bands, lagged returns, and sentiment).
- **Modeling**:  
  - `DecisionTreeClassifier` (baseline)  
  - `XGBoostClassifier` (with hyperparameter tuning via GridSearchCV)  
- **Strategy Logic**:  
  - Go long on a positive prediction; hold cash on negative  
  - Monthly rebalancing via custom strategy using the `bt` Python library

## 🔁 Backtesting

- **Benchmark**: Buy-and-hold on PFE stock
- **Test Set**: Final 20% of time-series data, unseen by the models
- **Simulations**: 1,000 backtest loops for both ML strategy and benchmark

### 📉 Key Results

- **Accuracy**:  
  - Decision Tree: ~51.43%  
  - XGBoost: ~51.43% (negligible improvement)  

- **Returns**:  
  - ML models **slightly outperformed** the benchmark in select loops  
  - XGBoost strategy showed marginally better risk-adjusted metrics  
  - **Still near-random** performance overall, as expected with limited data

> 📊 Despite underwhelming performance, the ML strategy’s ability to edge past a declining benchmark in certain cases suggests potential when scaled with richer data, compute, and domain expertise.

## 📁 Project Structure

- `Data Sources (Live)`  
  - **Reddit API** – Scraped sentiment data from `r/wallstreetbets` mentioning Pfizer  
  - **Yahoo Finance API** (`yfinance`) – Historical Pfizer (PFE) stock data (2010–2024)

- `Outputs/`  
  - `algo_trading_proposal.pdf` – 1,500-word academic write-up with results, visualizations, and references  
  - `final_script_tree.py` – Pipeline using Decision Tree Classifier  
  - `final_script_xgboost.py` – Pipeline using XGBoost  
  - `writeup_visualizations.py` – Charts comparing strategy metrics  

## 🧩 Challenges & Limitations

- **Reddit API Limits**: Capped at 1,000 posts (no access to Twitter/X)  
- **Basic Sentiment Scoring**: `TextBlob` not finance-optimized  
- **Compute Constraints**: Limited loops, no deep model tuning  
- **Binary Targets**: No regression or probabilistic outputs

## 🏆 Academic Recognition

- Received the **highest possible grade** in *Algorithmic Trading in Python* (Hult International Business School, Summer 2024)
- Ranked in the **top 25% of the class**
- Personally offered a **recommendation letter** from Professor Michael Rolleigh (AI/ML & Finance) for technical quality, project leadership, and classroom excellence

> *“Awale was an incredibly impressive student… He clearly demonstrated mastery of the material and its application by earning the highest possible grade. I would not hesitate to hire Awale as a research assistant and would be delighted to have him as a colleague.”*  
> — **Professor Michael Rolleigh**, Hult International Business School


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
