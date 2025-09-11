import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("Samples/Flight.csv")


df['date'] = pd.to_datetime(df['month'] + ' ' + df['year'].astype(str))
df = df.sort_values('date').reset_index(drop=True)

features = df[['passengers', 'revenue ($)']].values.astype(float)


scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)


lookback = 12
target_dates = df['date'][lookback:].reset_index(drop=True)

def create_sequences(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])      
        y.append(data[i + lookback])        
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, lookback=lookback)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_dates = target_dates[train_size:].reset_index(drop=True)


model = Sequential([
    LSTM(64, activation='tanh', input_shape=(lookback, 2)),
    Dense(2) 
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)


y_pred_scaled = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred_scaled)

y_test_pass = y_test_inv[:, 0]
y_test_rev  = y_test_inv[:, 1]
y_pred_pass = y_pred_inv[:, 0]
y_pred_rev  = y_pred_inv[:, 1]


print("Passenger Metrics:")
print(f"  MSE : {mean_squared_error(y_test_pass, y_pred_pass):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_pass, y_pred_pass)):.2f}")
print(f"  MAE : {mean_absolute_error(y_test_pass, y_pred_pass):.2f}")
print(f"  R²  : {r2_score(y_test_pass, y_pred_pass):.4f}")

print("\nRevenue Metrics:")
print(f"  MSE : {mean_squared_error(y_test_rev, y_pred_rev):.2f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_rev, y_pred_rev)):.2f}")
print(f"  MAE : {mean_absolute_error(y_test_rev, y_pred_rev):.2f}")
print(f"  R²  : {r2_score(y_test_rev, y_pred_rev):.4f}")


last_sequence = scaled[-lookback:].reshape((1, lookback, 2))
next_pred_scaled = model.predict(last_sequence)
next_pred_inv = scaler.inverse_transform(next_pred_scaled)

next_passengers_inv = next_pred_inv[0, 0]
next_revenue_inv    = next_pred_inv[0, 1]

next_month = df['date'].max() + pd.DateOffset(months=1)
print(f"\nForecast for {next_month.strftime('%B %Y')}:")
print(f"  Predicted Passengers: {next_passengers_inv:.0f}")
print(f"  Predicted Revenue  : ${next_revenue_inv:,.0f}")



plt.figure(figsize=(14,6))
plt.plot(test_dates, y_test_pass, label='True Passengers')
plt.plot(test_dates, y_pred_pass, label='Predicted Passengers')
plt.title("Passenger Forecast (Test Set)")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
plt.plot(test_dates, y_test_rev, label='True Revenue')
plt.plot(test_dates, y_pred_rev, label='Predicted Revenue')
plt.title("Revenue Forecast (Test Set)")
plt.xlabel("Date")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


last_12_scaled = scaled[-12:]
last_12_inv = scaler.inverse_transform(last_12_scaled)
last_12_pass = last_12_inv[:, 0]

plt.figure(figsize=(10,5))
months_rel = list(range(1, 13))

plt.plot(months_rel, last_12_pass, label='Last 12 Months (Passengers)')

plt.plot(13, next_passengers_inv, 'ro', 
         label=f'Predicted Next Month ({next_month.strftime("%b %Y")})', 
         markersize=8)

plt.title("AirPassengers — Last 12 Months + Next Month Prediction (Passengers)")
plt.xlabel("Months (relative to last 12)")
plt.ylabel("Passengers")
plt.xticks(range(1,14))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)  
plt.tight_layout()
plt.show()
