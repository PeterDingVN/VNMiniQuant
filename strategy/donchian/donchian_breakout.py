import pandas as pd
import numpy as np

class Donchian:
    def __init__(self, config):
        self.cfg = config

    def run(self, df: pd.DataFrame):
        df_ = df.copy()
        pos = self.donchian_breakout(ohlc=df_)
        df_['position'] = pos

        return df_[['time', 'close', 'position']]

    def donchian_breakout(self, ohlc: pd.DataFrame):

        lookback = self.cfg.lookback
        # input df is assumed to have a 'close' column
        upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
        lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
        signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
        signal.loc[ohlc['close'] > upper] = 1
        signal.loc[ohlc['close'] < lower] = -1
        signal = signal.ffill()
        return signal

