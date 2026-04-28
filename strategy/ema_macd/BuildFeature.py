from config import EmaMacdCfg
import pandas as pd

class EmaMacdFeatures:

    @staticmethod
    def build_features(df: pd.DataFrame, ema_start:int= EmaMacdCfg.EMA_START, 
                                               signal_start:int = EmaMacdCfg.SIGNAL_START):
        df = df.copy()

        df = EMA.add_ema(df, ema_start_at=ema_start)
        df = MACD.add_macd(df, ema_start_at=ema_start, 
                                signal_start_at=signal_start)

        return df


class EMA:

    @staticmethod
    def _col_validate(df: pd.DataFrame):
        if 'close' not in df.columns:
            raise ValueError("'close' column is missing from DataFrame")
        

    @staticmethod
    def add_ema(df: pd.DataFrame, short_ema:int=EmaMacdCfg.SHORT_EMA, long_ema: int=EmaMacdCfg.LONG_EMA, 
                ema_start_at:int=EmaMacdCfg.EMA_START):

        EMA._col_validate(df)
        data = df.copy()

        for ema_line in [short_ema, long_ema]:
            col = f"ema_{ema_line}"

            data[col] = data['close'].ewm(span=ema_line, adjust=False, min_periods=ema_start_at).mean()

        return data
    
class MACD(EMA):

    @staticmethod
    def add_macd(data: pd.DataFrame, fast_len:int=EmaMacdCfg.SHORT_EMA, 
                 slow_len:int=EmaMacdCfg.LONG_EMA, signal_len:int=EmaMacdCfg.SIGNAL,
                 ema_start_at:int=EmaMacdCfg.EMA_START, 
                 signal_start_at:int=EmaMacdCfg.SIGNAL_START):

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

# ------------------------------------------
# -------------- TEST CASE -----------------
# ------------------------------------------

if __name__ == '__main__':
    from data import AccessData
    all_data = AccessData(symbol=['AGR']).access_data()[0]['data']

    trans_data = EmaMacdFeatures().build_features(all_data)
    print(trans_data.head(20))

# CMD: python -m strategy.ema_macd.BuildFeature
