from data.data_features import EMA, MACD
from config import EMA_START, SIGNAL_START
import pandas as pd

class FeatureBuilderNonML:

    def __init__(self):
        self.ema = EMA()
        self.macd = MACD()

    def build_features(self, df: pd.DataFrame, ema_start:int= EMA_START, 
                                               signal_start:int = SIGNAL_START):
        df = df.copy()

        df = self.ema.add_ema(df, ema_start_at=EMA_START)
        df = self.macd.add_macd(df, ema_start_at=EMA_START, 
                                signal_start_at=SIGNAL_START)

        return df

# ------------------------------------------
# -------------- TEST CASE -----------------
# ------------------------------------------

if __name__ == '__main__':
    from data import AccessData
    all_data = AccessData(symbol=['AGR']).access_data()[0]['data']

    trans_data = FeatureBuilderNonML().build_features(all_data)
    print(trans_data.head(20))

# CMD: python -m data.data_features.all_features.non_ml_based.non_ml_feature_build 