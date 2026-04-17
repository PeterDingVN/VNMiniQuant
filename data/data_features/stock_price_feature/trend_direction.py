import pandas as pd

# Example: load your data (must contain a 'Close' column)
# df = pd.read_csv("data.csv")

# Sample dataframe
df = pd.DataFrame({
    "Close": [100, 102, 101, 103, 105, 107, 106, 108, 110, 112]
})

# Calculate EMA55 and EMA233
df["EMA55"] = df["Close"].ewm(span=55, adjust=False).mean()
df["EMA233"] = df["Close"].ewm(span=233, adjust=False).mean()

# Show result
# print(df)



#



    
    
