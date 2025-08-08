import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the AEMO data from the CSV file
df = pd.read_csv('NEMPRICEANDDEMAND_QLD1_202508082035.csv', parse_dates=['Settlement Date']) # Data from NEM data dashboard, price and demand, donwloaded on the 08/08/2025

# The 'Scheduled Demand (MW)' column is used for the forecast
# Features are extracted from the 'Settlement Date' to capture time-based trends
df['month'] = df['Settlement Date'].dt.month
df['day_of_week'] = df['Settlement Date'].dt.dayofweek # Monday=0, Sunday=6
df['hour'] = df['Settlement Date'].dt.hour # Energy demand varies greatly by hour
df['minute'] = df['Settlement Date'].dt.minute # Minute is added to handle 5-minute intervals

# A 'lag' feature uses the demand from the previous 5-minute period to predict the next
df['lag_1_period'] = df['Scheduled Demand (MW)'].shift(1)

# The first row with a NaN value from the lag feature is dropped
df.dropna(inplace=True)

# The features (X) and target (y) are defined
X = df[['month', 'day_of_week', 'hour', 'minute', 'lag_1_period']]
y = df['Scheduled Demand (MW)']

# The data is split into 80% for training and 20% for testing
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# A Random Forest Regressor model is created to capture complex patterns
model = RandomForestRegressor(n_estimators=100, random_state=42)

# The model is trained on the training data
model.fit(X_train, y_train)
print("Advanced Random Forest model training complete.")

# Predictions are made on the test data
y_pred = model.predict(X_test)

# The Mean Absolute Error (MAE) is calculated to evaluate performance
mae = mean_absolute_error(y_test, y_pred)
print(f"\nAdvanced Model Performance (Random Forest):")
print(f"Mean Absolute Error (MAE): {mae:.2f} MW")

# A forecast for the next 5 5-minute periods is created
last_known_demand = y_test.iloc[-1]
last_known_date = df['Settlement Date'].iloc[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(minutes=5), periods=5, freq='5min')
future_data = pd.DataFrame({
    'month': future_dates.month,
    'day_of_week': future_dates.dayofweek,
    'hour': future_dates.hour,
    'minute': future_dates.minute,
    # The last known demand value is used for the future lags
    'lag_1_period': last_known_demand
})

# The trained model is used to predict future demand
future_forecast = model.predict(future_data)

print("\n--- 5-Period Energy Demand Forecast (5-minute intervals) ---")
for i, date in enumerate(future_dates):
    print(f"Date: {date.strftime('%Y-%m-%d %H:%M')}, Predicted Demand: {future_forecast[i]:.2f} MW")
