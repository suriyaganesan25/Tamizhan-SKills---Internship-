import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("sales_data.csv")

print("First 5 rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'])

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

X = df[['quantity', 'day', 'month', 'year']]
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

results = pd.DataFrame({'Actual Revenue': y_test, 'Predicted Revenue': y_pred})
print("\nSample Predictions:")
print(results.head())

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

future_data = pd.DataFrame({
    'quantity': [50, 70, 90],
    'day': [15, 20, 25],
    'month': [8, 8, 8],
    'year': [2025, 2025, 2025]
})
future_predictions = model.predict(future_data)

print("\nForecast for Upcoming Periods:")
print(pd.DataFrame(future_data).assign(Predicted_Revenue=future_predictions))
