
import pandas as pd
import numpy as np
from arch import arch_model

# Generate dummy data
np.random.seed(42)
returns = pd.Series(np.random.normal(0, 1, 100), name='log_ret')

# Fit model
am = arch_model(returns, vol='Garch', p=1, o=0, q=1)
res = am.fit(disp='off')

# Forecast with horizon 5
forecasts = res.forecast(horizon=5)

print("\nForecasts Variance structure:")
print(forecasts.variance.head())
print(forecasts.variance.tail())

print("\nColumns:", forecasts.variance.columns)

# Check access
last_row = forecasts.variance.iloc[-1]
print("\nLast row (Forecasts from T):")
print(last_row)

print(f"\nValue for h=1: {last_row['h.1']}")
print(f"Value for h=5: {last_row['h.5']}")
