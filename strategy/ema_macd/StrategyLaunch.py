from .BuildFeature import EmaMacdFeatures
from .SignalGen import generate_signals
from config import EmaMacdCfg, AssetType
from strategy_backtest import TransactionCost
import pandas as pd
import numpy as np


class EmaMacdStrategy:
    
    def __init__(self, config: EmaMacdCfg):
        self.cfg = config

    def run(self, df: pd.DataFrame, cost_rate: float=AssetType.cost_type['stock'])-> pd.DataFrame:

        # Copy data
        df_cp = df.copy()

        # Build features
        df_fe = EmaMacdFeatures.build_features(df_cp)

        # Add signal
        df_signal = generate_signals(df_fe, start_sig=self.cfg.SIGNAL_START)

        df_signal['position'] = df_signal['final_signal']


        return df_signal
        