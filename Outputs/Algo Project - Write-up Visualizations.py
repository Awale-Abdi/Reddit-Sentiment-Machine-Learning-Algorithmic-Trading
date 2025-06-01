"""Author of this code is Awale Abdi from Professor Michael 
Rolleigh's class."""

# Importing Matplotlib for my visualizations
import matplotlib.pyplot as plt

# First I'm going to visualize the results of
## the benchmark
# Hardcoding the data in need of visualizing
metrics = {
    "Total Return": -26.84,
    "CAGR": -11.18,
    "Max Drawdown": 57.34,
    "Daily Sharpe Ratio": -0.31,
    "Daily Sortino Ratio": -0.55,
    "Calmar Ratio": -0.19,
    "Daily Volatility (ann.)": 26.63,
    "Best Day": 10.85,
    "Worst Day": -6.72,
    "Best Month": 22.84,
    "Worst Month": -13.82,
    "Best Year": -13.23,
    "Worst Year": -43.81
}

# Creatin the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(metrics.keys(), metrics.values(), color='red')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Financial Performance Metrics In The Red')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Saving the plot to a file
plt.savefig('/mnt/data/financial_performance_metrics_all_red.png')

# Displaying the plot
plt.show()


# Now we do the same for the ML Strategy
# Hardcoding the data in need of visualizing
metrics = {
    "Total Return": 3.18,
    "Daily Sharpe": 0.16,
    "Daily Sortino": 0.26,
    "CAGR": 1.19,
    "Max Drawdown": -24.18,
    "Calmar Ratio": 0.05,
    "Monthly Sharpe": 0.21,
    "Monthly Sortino": 0.41,
    "Monthly Mean (ann.)": 2.53,
    "Monthly Vol (ann.)": 12.26,
    "Best Month": 13.16,
    "Worst Month": -7.73,
    "Avg. Drawdown": -4.21,
    "Avg. Drawdown Days": 80.00,
    "Avg. Up Month": 4.59,
    "Avg. Down Month": -0.84,
    "Win Year %": 50.00,
    "Win 12m %": 47.62
}

# Creating the bar chart
plt.figure(figsize=(12, 7))
bars = plt.bar(metrics.keys(), metrics.values(), color='lightgreen')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Financial Performance Metrics In The Green')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Saving the plot to a file
plt.savefig('financial_performance_metrics_all_green.png')

# Displaying the plot
plt.show()

