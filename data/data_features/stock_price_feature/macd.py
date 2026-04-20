import pandas as pd
from config import SHORT_EMA, LONG_EMA, SIGNAL, EMA_START, SIGNAL_START
from .ema import EMA


class MACD(EMA):

    @staticmethod
    def add_macd(data: pd.DataFrame, fast_len:int=SHORT_EMA, 
                 slow_len:int=LONG_EMA, signal_len:int=SIGNAL,
                 ema_start_at:int=EMA_START, 
                 signal_start_at:int=SIGNAL_START):

        EMA._col_validate(data)
        df = data.copy()

        fast = (
            df['close']
            .ewm(span=fast_len, adjust=False, min_periods=ema_start_at)
            .mean())
        slow = (
            df['close']
            .ewm(span=slow_len, adjust=False, min_periods=ema_start_at)
            .mean())

        macd = fast - slow
        
        signal = macd.ewm(
            span=signal_len,
            adjust=False, min_periods=signal_start_at).mean()

        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = macd - signal

        return df