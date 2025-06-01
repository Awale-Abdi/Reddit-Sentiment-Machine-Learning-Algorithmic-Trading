"""Author of this code is Awale Abdi from Professor Michael 
Rolleigh's class."""

# Getting the necessary imports done
import requests
import datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
import bt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import ta
from joblib import Parallel, delayed

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

# Creating the features for our Machine Learning Model
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

# Hyperparameter tuning for our Decision Tree
param_grid_tree = {'max_depth': [5, 10, 15, 20, 25, 30], 'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['gini', 'entropy']}
grid_search_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_tree, cv=5, n_jobs=-1, verbose=2)
grid_search_tree.fit(X_train, y_train)
best_params_tree = grid_search_tree.best_params_
tree_model = DecisionTreeClassifier(**best_params_tree, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f'Improved Decision Tree Accuracy: {accuracy_tree}')
print(classification_report(y_test, y_pred_tree))
cv_scores_tree = cross_val_score(tree_model, X_train, y_train, cv=5)
print(f'Cross-validated accuracy (Decision Tree): {cv_scores_tree.mean()}')

# Finally, creating a trading strategy function based on the predictions of the Decision Tree model
data['Predictions'] = np.nan
test_index = X_test.index
data.loc[test_index, 'Predictions'] = y_pred_tree

class MLAlgo(bt.Algo):
    def __init__(self):
        super(MLAlgo, self).__init__()
        self.last_month = None  # Initialize last_month attribute

    def __call__(self, target):
        if target.now.month != self.last_month:
            self.last_month = target.now.month
            preds = data.loc[target.now, 'Predictions']
            if preds == 1:
                target.temp['selected'] = target.universe.columns
            else:
                target.temp['selected'] = []
        return True

# Defining the benchmark strategy (Buy and Hold)
benchmark_strategy = bt.Strategy('Buy and Hold', [bt.algos.RunOnce(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()])

# Setting up counters for performance comparisons
num_iterations = 10  # Reduced to 10 for testing purposes
benchmark_cagr = 0.07  # Conservative benchmark CAGR
benchmark_total_return = 0.10  # Conservative benchmark Total Return
exceed_count = 0

# Defining the Machine Learning strategy
Strat = bt.Strategy('ML_PFE', [bt.algos.RunMonthly(), MLAlgo(), bt.algos.WeighEqually(), bt.algos.Rebalance()])

# Function for running a single loop backtest per strategy
def run_single_iteration():
    ml_test = bt.Backtest(Strat, data[['Close']][split_point:])
    ml_res = bt.run(ml_test)
    
    benchmark_test = bt.Backtest(benchmark_strategy, data[['Close']][split_point:])
    benchmark_res = bt.run(benchmark_test)
    
    ml_cagr = ml_res.stats.loc['cagr'].values[0]
    ml_total_return = ml_res.stats.loc['total_return'].values[0]
    
    if ml_cagr > benchmark_cagr and ml_total_return > benchmark_total_return:
        return 1, ml_res.stats, benchmark_res.stats
    return 0, ml_res.stats, benchmark_res.stats

# Running the loop in parallel for faster performance once I raise the number of loops
results = Parallel(n_jobs=-1)(delayed(run_single_iteration)() for _ in range(num_iterations))

# Calculating percentages for when the Machine Learning Model beat the Benchmark 
exceed_count = sum(result[0] for result in results)
exceed_percentage = (exceed_count / num_iterations) * 100
not_exceed_percentage = 100 - exceed_percentage

print(f'Exceed benchmark: {exceed_percentage}%')
print(f'Do not exceed benchmark: {not_exceed_percentage}%')

# Printing the metrics for the last loop
print("ML Strategy Metrics (last iteration):")
print(results[-1][1])
print("\nBenchmark Strategy Metrics (last iteration):")
print(results[-1][2])

# Backtesting and displaying the results of the Machine Learning strategy
ml_test = bt.Backtest(Strat, data[['Close']][split_point:])
ml_res = bt.run(ml_test)
ml_res.plot()
ml_res.display()
ml_res.plot_security_weights()
ml_res.plot_histogram()

# Backtesting and displaying the results of the Benchmark strategy
benchmark_test = bt.Backtest(benchmark_strategy, data[['Close']][split_point:])
benchmark_res = bt.run(benchmark_test)
benchmark_res.plot()
benchmark_res.display()
benchmark_res.plot_security_weights()
benchmark_res.plot_histogram()