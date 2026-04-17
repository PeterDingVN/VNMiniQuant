 # MACD with:
# Fast Length = 55
# Slow Length = 233
# Signal Line = 50

import pandas as pd

# Example dataframe with Close prices
df = pd.DataFrame({
    "Close": [100, 102, 101, 103, 105, 107, 106, 108, 110, 112]
})

# Calculate EMAs
df["EMA_fast_55"] = df["Close"].ewm(span=55, adjust=False).mean()
df["EMA_slow_233"] = df["Close"].ewm(span=233, adjust=False).mean()

# MACD Line
df["MACD"] = df["EMA_fast_55"] - df["EMA_slow_233"]

# Signal Line (EMA of MACD)
df["Signal_50"] = df["MACD"].ewm(span=50, adjust=False).mean()

# Histogram
df["Histogram"] = df["MACD"] - df["Signal_50"]

# Show result
# print(df)