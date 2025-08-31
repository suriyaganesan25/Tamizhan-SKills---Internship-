import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset (CSV with columns: datetime, vehicle_count)
df = pd.read_csv('traffic_data.csv', parse_dates=['datetime'])

# Extract date and hour for aggregation
df['date'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# Aggregate vehicle counts by hour
hourly_traffic = df.groupby('hour')['vehicle_count'].sum().reset_index()

# Plot traffic density over 24 hours
plt.figure(figsize=(10,5))
sns.lineplot(data=hourly_traffic, x='hour', y='vehicle_count')
plt.title('Traffic Density by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Total Vehicle Count')
plt.xticks(range(0,24))
plt.tight_layout()
plt.show()

# Identify anomalies: e.g., hours where vehicle count deviates significantly from mean
mean_count = hourly_traffic['vehicle_count'].mean()
std_count = hourly_traffic['vehicle_count'].std()
threshold_upper = mean_count + 2 * std_count
threshold_lower = mean_count - 2 * std_count

anomalies = hourly_traffic[(hourly_traffic['vehicle_count'] > threshold_upper) | (hourly_traffic['vehicle_count'] < threshold_lower)]
print("Anomalous hours:\n", anomalies)

# Optional: Predict peak hours using regression on hourly data
X = hourly_traffic[['hour']]
y = hourly_traffic['vehicle_count']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Plot actual vs predicted for test set
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Traffic Vehicle Count: Actual vs Predicted')
plt.xlabel('Hour of Day')
plt.ylabel('Vehicle Count')
plt.legend()
plt.tight_layout()
plt.show()

print("Regression coefficients:", model.coef_)
print("Regression intercept:", model.intercept_)
