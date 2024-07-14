import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import pandas_ta as ta
import joblib
from datetime import datetime

# Ensure the end date is always up-to-date
end_date = datetime.now().strftime('%Y-%m-%d')

# Download SPY data
spy = yf.download('SPY', start='2010-01-01', end=end_date)

# Display the last few rows of data to check the current ETF level
print(spy.tail())

# Drop rows with missing values
spy.dropna(inplace=True)

# Create additional features
spy['Return'] = spy['Close'].pct_change()
spy['Volatility'] = spy['Close'].rolling(window=21).std()
spy['MA_10'] = spy['Close'].rolling(window=10).mean()
spy['MA_50'] = spy['Close'].rolling(window=50).mean()
spy['Day_of_Week'] = spy.index.dayofweek
spy['Month'] = spy.index.month
spy['Lag_1'] = spy['Close'].shift(1)
spy['Lag_2'] = spy['Close'].shift(2)

# Add more technical indicators using pandas_ta
spy['RSI'] = ta.rsi(spy['Close'], length=14)
macd = ta.macd(spy['Close'], fast=12, slow=26, signal=9)
spy['MACD'] = macd['MACD_12_26_9']
spy['MACD_signal'] = macd['MACDs_12_26_9']
spy['MACD_hist'] = macd['MACDh_12_26_9']
bbands = ta.bbands(spy['Close'], length=20)
spy['BB_upper'] = bbands['BBU_20_2.0']
spy['BB_middle'] = bbands['BBM_20_2.0']
spy['BB_lower'] = bbands['BBL_20_2.0']

# Drop rows with missing values after adding new features
spy.dropna(inplace=True)

# Shift the 'Close' price to create the target variable
spy['Target'] = spy['Close'].shift(-1)
spy.dropna(inplace=True)

# Define features and target
features = ['Close', 'Return', 'Volatility', 'MA_10', 'MA_50', 'Day_of_Week', 'Month', 'Lag_1', 'Lag_2', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower']
target = 'Target'

X = spy[features]
y = spy[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_mae = -np.mean(cv_scores)
print(f'Cross-Validated MAE: {cv_mae}')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize the Grid Search
grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Train the model with the best parameters
best_model = XGBRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Save the model for future use
joblib.dump(best_model, 'best_model.pkl')

# Predict the next day's price
last_row = pd.DataFrame([spy[features].iloc[-1].values], columns=features)
next_day_price = best_model.predict(last_row)
print(f'Predicted next day price: {next_day_price[0]}')
    