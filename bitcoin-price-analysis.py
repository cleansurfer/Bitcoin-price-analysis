import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv("btcusd_1-min_data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------
print(df.head())
print(df.describe())

plt.figure(figsize=(10,5))
plt.plot(df['Timestamp'], df['Close'])
plt.title("Bitcoin Closing Price Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Price")
plt.show()

# Moving average
df['MA_30'] = df['Close'].rolling(window=30).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Timestamp'], df['Close'], label="Close Price")
plt.plot(df['Timestamp'], df['MA_30'], label="30-Day MA")
plt.legend()
plt.title("BTC Price with Moving Average")
plt.show()

# -----------------------------
# 3. Simple Prediction Model
# -----------------------------
df['Target'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Close']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# -----------------------------
# 4. Visualization of Predictions
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted BTC Prices")
plt.show()