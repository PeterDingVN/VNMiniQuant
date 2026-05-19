import pandas as pd

class ema_crossover_strategy:

    def __init__(self, config:dict):
        self.config = config


    def run(self,
        df: pd.DataFrame):
    
        fast_period = self.config["fastperiod"]
        slow_period = self.config["slowperiod"]

        df = df.copy()

        # Calculate EMAs
        df["ema_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()

        # Raw signal
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1

        # Crossover entries only
        df["position"] = df["signal"].diff()

        return df[['time', 'close', 'position']]
