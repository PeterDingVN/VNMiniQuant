from .BuildFeature import EmaMacdFeatures
from .SignalGen import generate_signals
from config import EmaMacdCfg
from strategy_backtest import TransactionCost
import pandas as pd
import numpy as np


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

        # Add transaction cost
        cost_col = TransactionCost().transaction_cost_arr(df_signal['final_signal'])
        df_signal['trans_cost'] = cost_col

        # Final output data
        df_signal['log_return'] = df_signal['log_return'] + df_signal['trans_cost']
        df_signal['real_return'] = df_signal['log_return'] * abs(df_signal['final_signal'].shift(1))  
        df_signal['strat_ret'] = np.cumsum(df_signal['real_return'])

        return df_signal
        