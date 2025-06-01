"""Author of this code is Awale Abdi from Professor Michael 
Rolleigh's class."""

# Making the necessary imports
import requests
import datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
import bt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import ta
from joblib import Parallel, delayed
import xgboost as xgb

# Setting up the Sentiment Analysis Function
def search_sub_reddit(subreddit, search_term):
    url = f"https://oauth.reddit.com/r/{subreddit}/search"
    params = {'q': search_term, 'sort': 'top', 'limit': 100}
    headers = {'Authorization': 'bearer ' + token, 'User-Agent': 'Dapp by Classworkman'}
    response = requests.get(url, headers=headers, params=params)
    post_data = []

    if response.status_code == 200:
        posts = response.json()['data']['children']
        for post in posts:
            title = post['data']['title']
            text = post['data']['selftext']
            combined_text = title + " " + text
            if search_term.lower() in combined_text.lower():
                timestamp = datetime.datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%d')
                sentiment = TextBlob(combined_text).sentiment.polarity
                post_data.append({'timestamp': timestamp, 'sentiment': sentiment, 'text': combined_text})
    return pd.DataFrame(post_data, columns=['timestamp', 'sentiment', 'text'])

# Getting and processing the sentiment data from Reddit
REDDIT_USERNAME = "Classworkman"
REDDIT_PASSWORD = "orancio76"
APP_ID = "0nYkdEc7r7USBrh5E-CRuQ"
APP_SECRET = "weUnQjCm2m7NTiii4Gvs8IVzsxgpKw"
API_BASE_URL = 'https://www.reddit.com'
ACCESS_TOKEN_URL = API_BASE_URL + '/api/v1/access_token'
OAUTH_BASE_URL = 'https://oauth.reddit.com'
USER_AGENT = f'Dapp by {REDDIT_USERNAME}'

auth = requests.auth.HTTPBasicAuth(APP_ID, APP_SECRET)
data = {'grant_type': 'password', 'username': REDDIT_USERNAME, 'password': REDDIT_PASSWORD}
headers = {'user-agent': USER_AGENT}
response = requests.post(ACCESS_TOKEN_URL, data=data, headers=headers, auth=auth)
token_data = response.json()

token = 'bearer ' + token_data['access_token']

subreddits = ['wallstreetbets']
companies = ['Pfizer']
sentiment_df = pd.concat([search_sub_reddit(subreddit, company) for subreddit in subreddits for company in companies])
sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
daily_sentiment = sentiment_df.groupby(sentiment_df['timestamp'].dt.date)['sentiment'].mean()

# Fetching the historical data for Pfizer Inc. from Yahoo Finance
data = yf.download('PFE', start='2010-01-01', end='2024-01-01')

# Merging the sentiment data with the stock data so that it can be integrated into our later machine learning model
data = data.merge(daily_sentiment, left_index=True, right_index=True, how='left').fillna(0)

# Creating the features for the Machine Learning Model
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()
data['20_MA'] = data['Close'].rolling(window=20).mean()
data['100_MA'] = data['Close'].rolling(window=100).mean()
data['Price_Change'] = data['Close'].pct_change()
data['Volatility'] = data['Price_Change'].rolling(window=50).std()

# Adding in Momentum Indicators
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()
data['MACD_Signal'] = ta.trend.MACD(data['Close']).macd_signal()
data['ROC'] = ta.momentum.ROCIndicator(data['Close'], window=12).roc()

# Adding in Bollinger Bands
bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
data['Bollinger_High'] = bb.bollinger_hband()
data['Bollinger_Low'] = bb.bollinger_lband()

# Adding in Lagged Returns
for lag in range(1, 11):
    data[f'Lag_{lag}'] = data['Close'].pct_change(periods=lag)

# Creating the target variable: 1 if price will go up, 0 if price will go down
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Dropping rows with NaN values
data.dropna(inplace=True)

# Defining the features and target
features = ['20_MA', '50_MA', '100_MA', '200_MA', 'Price_Change', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'ROC', 'Bollinger_High', 'Bollinger_Low', 'sentiment'] + [f'Lag_{lag}' for lag in range(1, 11)]
X = data[features]
y = data['Target']

# Splitting data into training and testing sets based on time periods
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Selecting features using mutual_info_classif
selector = SelectKBest(mutual_info_classif, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
selected_features = selector.get_support(indices=True)
X_train = X_train.iloc[:, selected_features]
X_test = X_test.iloc[:, selected_features]

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
grid_search_xgb = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)
best_params_xgb = grid_search_xgb.best_params_
xgb_model = xgb.XGBClassifier(**best_params_xgb, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'Improved XGBoost Accuracy: {accuracy_xgb}')
print(classification_report(y_test, y_pred_xgb))
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f'Cross-validated accuracy (XGBoost): {cv_scores_xgb.mean()}')

# Finally, creating a trading strategy function based on the predictions of the XGBoost model
data['Predictions'] = np.nan
test_index = X_test.index
data.loc[test_index, 'Predictions'] = y_pred_xgb

class MLAlgo(bt.Algo):
    def __call__(self, target):
        preds = data.loc[target.now, 'Predictions']
        if preds == 1:
            target.temp['selected'] = target.universe.columns
        else:
            target.temp['selected'] = []
        return True

# Defining the Machine Learning strategy
Strat = bt.Strategy('ML_PFE', [bt.algos.RunMonthly(), MLAlgo(), bt.algos.WeighEqually(), bt.algos.Rebalance()])

# Defining the benchmark strategy (Buy and Hold)
benchmark_strategy = bt.Strategy('Buy and Hold', [bt.algos.RunOnce(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()])

# Function for running a single loop backtest per strategy
def run_ml_iteration():
    ml_test = bt.Backtest(Strat, data[['Close']][split_point:])
    ml_res = bt.run(ml_test)
    ml_cagr = ml_res.stats.loc['cagr'].values[0]
    ml_total_return = ml_res.stats.loc['total_return'].values[0]
    return ml_cagr, ml_total_return, ml_res

def run_benchmark_iteration():
    benchmark_test = bt.Backtest(benchmark_strategy, data[['Close']][split_point:])
    benchmark_res = bt.run(benchmark_test)
    benchmark_cagr = benchmark_res.stats.loc['cagr'].values[0]
    benchmark_total_return = benchmark_res.stats.loc['total_return'].values[0]
    return benchmark_cagr, benchmark_total_return, benchmark_res

# Running the loop in parallel for faster performance
num_iterations = 1000
ml_results = Parallel(n_jobs=-1)(delayed(run_ml_iteration)() for _ in range(num_iterations))
benchmark_results = Parallel(n_jobs=-1)(delayed(run_benchmark_iteration)() for _ in range(num_iterations))

# Find the best performing ML model and benchmark model
best_ml_result = max(ml_results, key=lambda x: (x[0], x[1]))
best_ml_cagr, best_ml_total_return, best_ml_res = best_ml_result

best_benchmark_result = max(benchmark_results, key=lambda x: (x[0], x[1]))
best_benchmark_cagr, best_benchmark_total_return, best_benchmark_res = best_benchmark_result

print(f'Best ML Strategy CAGR: {best_ml_cagr}')
print(f'Best ML Strategy Total Return: {best_ml_total_return}')

print(f'Best Benchmark CAGR: {best_benchmark_cagr}')
print(f'Best Benchmark Total Return: {best_benchmark_total_return}')

# Displaying the best ML strategy results
best_ml_res.plot()
best_ml_res.display()
best_ml_res.plot_security_weights()
best_ml_res.plot_histogram()

# Displaying the best benchmark results
best_benchmark_res.plot()
best_benchmark_res.display()
best_benchmark_res.plot_security_weights()
best_benchmark_res.plot_histogram()

# Plotting the combined equity graphs
plt.figure(figsize=(14, 7))
plt.plot(best_ml_res.prices, label='ML_PFE')
plt.plot(best_benchmark_res.prices, label='Buy and Hold')
plt.title('Equity Progression of XGBoost ML Strategy vs. Benchmark')
plt.legend()
plt.show()
