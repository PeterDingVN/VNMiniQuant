from .BuildFeature import EmaMacdFeatures
from .SignalGen import generate_signals
from config import EmaMacdCfg
import pandas as pd


class EmaMacdStrategy:
    
    def __init__(self, config: EmaMacdCfg):
        self.cfg = config

    def run(self, df: pd.DataFrame)-> pd.DataFrame:

        # Copy data
        df_cp = df.copy()

        # Build features
        df_fe = EmaMacdFeatures.build_features(df_cp)

        # Add signal
        df_signal = generate_signals(df_fe, start_sig=self.cfg.SIGNAL_START)

        return df_signal
        