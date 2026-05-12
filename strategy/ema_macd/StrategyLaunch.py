from .BuildFeature import EmaMacdFeatures
from .SignalGen import generate_signals
from config import EmaMacdCfg, COST_RATE
from strategy_backtest import TransactionCost
import pandas as pd
import numpy as np


class EmaMacdStrategy:
    
    def __init__(self, config: EmaMacdCfg):
        self.cfg = config

    def run(self, df: pd.DataFrame, cost_rate: float=COST_RATE)-> pd.DataFrame:

        # Copy data
        df_cp = df.copy()

        # Build features
        df_fe = EmaMacdFeatures.build_features(df_cp)

        # Add signal
        df_signal = generate_signals(df_fe, start_sig=self.cfg.SIGNAL_START)

        # Add transaction cost
        pos_cost_pct = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df_signal['final_signal'])
        trans_cost = pos_cost_pct * df_signal['close']

        # Final output data (add col for PnL)
        pt_ret = (df_signal['close'] - df_signal['close'].shift(1)).fillna(0)
        df_signal['point_ret'] = pt_ret * df_signal['final_signal'] - trans_cost
        df_signal['position'] = df_signal['final_signal']


        return df_signal
        