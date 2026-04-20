import pandas as pd
from config import LONG_EMA, SHORT_EMA, EMA_START

class EMA:

    @staticmethod
    def _col_validate(df: pd.DataFrame):
        if 'close' not in df.columns:
            raise ValueError("'close' column is missing from DataFrame")
        

    @staticmethod
    def add_ema(df: pd.DataFrame, short_ema:int=SHORT_EMA, long_ema: int=LONG_EMA, 
                ema_start_at:int=EMA_START):

        EMA._col_validate(df)
        data = df.copy()

        for ema_line in [short_ema, long_ema]:
            col = f"ema_{ema_line}"

            data[col] = data['close'].ewm(span=ema_line, adjust=False, min_periods=ema_start_at).mean()

        return data
